# 05 — The TimesFM hybrid

The differentiator. Use a **time-series foundation model** as a feature generator and let a
tree fuse its signal with business covariates the model can't see.

## How TimesFM runs (a Hugging Face serving Space)

TimesFM 2.5 isn't on PyPI (PyPI caps at 2.0.1) and pulls torch plus an ~800 MB checkpoint, so
**this repo never runs it directly**. It's deployed once as a standalone GPU **Hugging Face
Space** — the self-contained bundle in [`space/`](../../space) (`app.py` + `Dockerfile`) — which
exposes a single `POST /forecast`. The pipeline is a pure HTTP client: it never imports
`timesfm` or `torch` and never downloads the checkpoint.

The forecaster is an **injected port**, so wiring it in is a drop-in: point the pipeline at the
Space and the CV uses `RemoteTimesFMForecaster` (which needs only `requests`). Leave the endpoint
unset and the CV simply runs the tree/baseline models (`cli._timesfm_specs` returns nothing
without an endpoint).

```bash
export FORECAST_TIMESFM_ENDPOINT="https://<your-space>.hf.space/forecast"
export FORECAST_TIMESFM_ENDPOINT_TOKEN="<bearer if the Space is private>"
python -m forecasting find-best-model --metric rmsle --horizon 14
```

The Space compiles with a `max_horizon` (`TIMESFM_MAX_HORIZON`, default 64) that covers every
run horizon, and the client requests `max(horizon_days, final_horizon_days)` (= 39), so one
endpoint serves both the 14-day CV folds and the 39-day final pred-vs-real forecast. Why a
Space and not HF's
*serverless* Inference API: TimesFM's `forecast()` isn't a standard pipeline task, so it needs a
dedicated GPU Space/Endpoint. A single batched forecast over all 829 stores is **~6 min on CPU
vs seconds on the Space's GPU** — which is what makes the year-long CV practical.

**Two ways, both in the CV** (ranked by `find-best-model --metric ... --horizon ...`):
- `timesfm_zeroshot` (implemented) — TimesFM standalone. `recursive = False`: `fit` snapshots
  each store's daily-reindexed history; `predict` runs one batched native n-day forecast and
  reads off each (store, date) by calendar-day offset.
- `timesfm_hybrid` (implemented) — TimesFM **whole-window** signal + engineered features →
  XGBoost (below). TimesFM forecasts the horizon once at the cutoff; only the lags recurse.

## What TimesFM is

[TimesFM](https://github.com/google-research/timesfm) is Google Research's pretrained,
decoder-only foundation model for time-series forecasting. We use **2.5** (200M params,
PyTorch, `google/timesfm-2.5-200m-pytorch`): context up to 16k, horizon up to 1k, point +
10 quantile outputs. 200M is small (BERT-base ≈ 110M) — Google shrank it
from 500M (v2.0) for efficiency.

## The hybrid pipeline (window once at cutoff)

TimesFM forecasts a **block** better than one step at a time, so it forecasts the whole horizon
**once, at the cutoff, over actuals only** — and then leaves the recursive loop entirely. Only
the engineered lags recurse. For each in-window day +k:

```
                          ┌─▶ step +1 ┐
[ history ≤ cutoff ] ─▶ [ TimesFM 2.5, whole horizon ] ─▶ │   …       │  (computed ONCE, up front)
                          └─▶ step +k ┘
                                   │  tfm_point, tfm_q0..q9 for day +k
                                   ▼
[ engineered covariates for day +k ] ─────────────────────▶ [ XGBoost ] ─▶ ŷ(+k)
   (calendar, holidays, reservations, store stats, lags)         │
        ▲──────────  ŷ(+k) appended → next day's lags only  ◀────┘
```

`Predictor` calls `prepare_window(history, horizon)` once (TimesFM runs here), then per step
`augment(features, …)` just indexes the precomputed window at offset k — **no TimesFM call in
the loop**. The fed-back predictions flow into the lags only, never back into TimesFM.

The **leakage formula**, both halves safe by construction:

```
ŷ(k) = tree( lags/rollings(k)      [recursive: own predictions for unknown in-window days]
           + TimesFM_window(k) )   [step k of the single cutoff-origin forecast over history ≤ cutoff]
```

Lags never read in-window actuals; the window forecast uses only history ≤ cutoff. TimesFM
digests trend & seasonality; XGBoost learns how the **covariates modulate** that baseline —
how a holiday or a reservation surge bends it.

## The signal (and the stretch)

- **The signal:** TimesFM's **whole-window `forecast()` output — per horizon step, the point +
  10 quantile-head values** (`tfm_point`, `tfm_q0..tfm_q9`), indexed by days-ahead. Fully
  documented API.
- **Stretch (Phase 14):** swap in the raw **decoder embedding** via a forward hook on the
  PyTorch module. Richer, but undocumented and version-fragile — off the critical path.
- Covariates enter through the concatenation, so we don't need TimesFM's XReg path. **LoRA
  fine-tuning** is a further stretch.

## The compute trap and how we bound it (`adapters/models/timesfm/features.py`)

A naive hybrid runs TimesFM per horizon step inside both the training and the inference loops —
hours of Transformer calls. The window design plus three bounds keep it cheap:

1. **One call per origin, not per step.** Each origin's window forecast covers the whole
   horizon, so inference is **1 TimesFM call per forecast origin** (was ~horizon), and training
   is **1 call per training origin**.
2. **Training uses the window signal over *actual* origins.** `training_signal` forecasts the
   window once at each origin `c` and emits rows at `date = c + step`, so the head trains on the
   *same offset-k feature it sees at inference* (the old offset-1 mismatch is gone). The window
   for `c` depends only on history ≤ c, so it's identical across CV folds/configs →
   **memoised by origin** (in-memory, `TimesFMFeatureGenerator._memo`), and origins are
   **subsampled to the most recent `timesfm_train_cutoffs`** to bound size. The Transformer
   never runs inside a tree `fit`.
3. **Durable per-series cache at the client.** A TimesFM forecast is a pure function of
   `(trimmed history, horizon)`, so `RemoteTimesFMForecaster` content-addresses each series
   (SHA-256) and caches the result on disk (`timesfm_cache_dir`, default `.cache/timesfm`,
   gitignored; `npz`). The same `(store, cutoff)` series recurs across folds, across steps 1 & 3,
   across **both** TimesFM models, and across reruns — so each unique forecast hits the GPU
   **once, ever**. Only cache-misses in a batch are POSTed; a fully-cached batch makes no HTTP
   call at all. Set `timesfm_cache_dir=""` to disable.

(Lags at *train* time stay on actual history — the standard recursive-strategy training; the
tree's own in-window predictions don't exist yet, so actuals are the only well-defined choice.)

## What the ablation shows

`xgboost` (engineered features) vs `timesfm_hybrid` (same features + TimesFM signal),
same base learner. The RMSLE gap is **TimesFM's marginal contribution** — the headline number
for the presentation, and an honest one because the only thing that changed is the signal.
