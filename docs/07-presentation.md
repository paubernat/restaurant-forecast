# 07 · Presentation

Live technical-test presentation. **Slot: ~15-20 min talk + Q&A. Language: English.**
Each slide: **tier** · time · talking points · visual · eval criterion. Tiers —
**DEEP** (differentiator, spend time) / **MED** / **SKIM** (table-stakes, one slide, don't narrate).

Eval criteria: (1) problem framing · (2) validation & metric rigor · (3) code quality /
reproducibility · (4) communication via graphics · (5) technical depth in Q&A.

**The results are real** — produced by the committed pipeline; the figures live in
`artifacts/report/index.html` (open it during slide 10). Run config that produced them:
`weighted_mae` (over-ordering penalised 2×), horizon 14, holdout 39, stride 18 / 20 folds,
feature threshold 0.98.

| # | Slide | Tier | Time |
|---|-------|------|------|
| 1 | Framing: demand per center → auto-ordering | SKIM | 0:45 |
| 2 | Why the Recruit dataset | MED | 1:00 |
| 3 | Repo & architecture (hexagonal) | SKIM | 0:45 |
| 4 | The data, honestly (closed days + EDA) | MED | 1:30 |
| 5 | Feature engineering (7 families) | MED | 1:30 |
| 6 | Metrics & why | DEEP | 1:30 |
| 7 | Temporal validation + recursive contract | **DEEPEST** | 2:30 |
| 8 | 4-step selection pipeline + tree params | MED | 1:30 |
| 9 | TimesFM: what / zero-shot / hybrid / ablation | DEEP | 2:30 |
| 10 | Results & graphics | **MEAT** | 2:30 |
| 11 | Scaling to the real fleet | MED | 1:15 |
| 12 | Packaging & ops (Docker / K8s / MLflow) | SKIM | 0:45 |
| 13 | Close + what's next | — | 0:30 |

**Core ≈ 17.5 min.** Squeeze to 15: merge 8→9, cut 3 to a 15-sec aside. Given 25+: 2nd results
slide + a live `render-report` / MLflow-UI demo.

---

## Slide 1 · Framing — SKIM · 0:45 · → crit 1
- Problem: forecast **daily demand per center** to feed **automatic supplier orders**.
- Gstock's real case (centers × products → auto-orders); the Recruit dataset is a reduced-scale
  proxy with the same shape.
- Set the *why* before any model talk.
- **Visual:** one-line problem statement + a single demand curve.

## Slide 2 · Why Recruit — MED · 1:00 · → crit 1
- Natively **multi-series** (~829 stores, daily Jan 2016 → Apr 2017) = "comensales por centro."
- Real signals: reservations, holidays, genre/area.
- Built-in **RMSLE benchmark** + a defined test window.
- Best fit for a foundation model (needs many real seasonal series).
- Rejected alts (one line): Hotel Booking = classification / ~2 series; Restaurant Revenue =
  137 rows, not a series.
- **Visual:** tiny 3-row comparison table (dataset → kept/dropped).

## Slide 3 · Repo & architecture — SKIM · 0:45 · → crit 3
- Hexagonal: **pure domain** (features / metrics / validation, no IO) · **adapters** (IO + model
  libs) · **one CLI** composition root. 5 models behind one `Model` port.
- `AGENTS.md` encodes the rules (no leakage, recursive eval contract, log-domain safety).
- **Visual: 📸 architecture screenshot/diagram** — the `src/forecasting/` folder tree
  (`domain/` · `domain/ports/` · `adapters/` · `application/` · `entrypoints/cli.py`). Source for
  the capture: `README.md` "Architecture (hexagonal)" block, or the tree itself. Show it, move on.

## Slide 4 · The data, honestly — MED · 1:30 · → crit 1, 2
- 8 CSVs (AIR point-of-sale, HPG reservations, relation, store/date info) joined to a canonical
  `store_id / date / visitors`.
- The honest gotcha: **closed-day rows are ABSENT** (~15% of the calendar) — not zeros, *missing*.
- Decision: reindex each store to a full daily calendar, **fill `visitors=0` AND set
  `is_closed=1`** — the model separates "closed" from "open but zero," and lags stay
  calendar-aligned.
- **Visual:** 1-2 EDA charts (`notebooks/01_eda.ipynb`) — weekly seasonality + a holiday spike.

## Slide 5 · Feature engineering — MED · 1:30 · → crit 1
- **35 features, 7 families:** **calendar** (dow, weekend, month, day-of-month, sin/cos
  day-of-year), **holiday** (is_holiday, golden_week, day_before/after), **closed-flag**,
  **lags** at multiples of 7 (1,7,14,21,28,35), **rolling mean+median** (7,14,28,35),
  **reservations** (count, visitors, lead-time — a *leading* indicator), **store stats**
  (genre, area_prefecture/ward, per-store & per-store×dow means).
- Two things to emphasise:
  - **log1p target** — count data, RMSLE-safe.
  - **leakage discipline** — lags shifted, rolling uses `shift(1)` (window ends *yesterday*),
    store aggregates computed on the **train slice only**.
- **Visual:** the 7-family table + a "shifted lag" mini-diagram.

## Slide 6 · Metrics & why — DEEP · 1:30 · → crit 2
- **RMSLE (primary / ranking):** count data, symmetric in log space, fair across store sizes,
  matches the Recruit benchmark, safe at 0 via log1p.
- **MAE:** the human-readable companion (average miss in visitors).
- **`weighted_mae` (the business metric):** asymmetric cost. For *this* run we penalise
  **over-ordering (waste) 2×** vs under-ordering — a perishable-goods stance (waste is the
  dominant cost). It's a **knob** (`under_weight`/`over_weight`): flip it for a stockout-averse
  business. Selection ran on `weighted_mae`, so the chosen model minimises the *business* cost,
  not raw log-error.
- All predictions clipped ≥0 before any metric.
- **Visual:** metric table + the asymmetric-cost intuition (waste vs stockout).

## Slide 7 · Temporal validation + recursive contract — DEEPEST · 2:30 · → crit 2
The rigor showcase — spend the most time here.
- **Holdout carved off first:** the last `final_horizon_days` (**39** = the length of the
  official Recruit test window) of *labeled* data. Labels end 2017-04-22, so it's
  **2017-03-15 → 04-22** (Golden Week sits in the unlabeled official scoring window). CV never
  touches it; only the final deployable model is refit *including* it.
- **Rolling-origin, expanding-window CV** on the earlier history: a split is **three dates, not
  row indices** (`train_end < valid_start ≤ valid_end`), **no shuffle**. Each fold trains on
  **all** history up to its cutoff (the train set grows fold to fold) and validates the next
  `horizon` (14) days.
- **The stride trick (the detail to land):** consecutive fold origins step by a stride that is
  **coprime with 7** — this run used **18** (→ **20 folds spread across the whole year**); the
  default is 9. Why coprime: if the step were a multiple of 7, every fold would start on the
  **same weekday** and `horizon_offset=1` would always be that weekday → **by-horizon error
  confounded with day-of-week**. Coprime ⇒ origins **rotate through every weekday**. Honest cost:
  stride < horizon overlaps windows → correlated folds (variance slightly understated).
- **Recursive multi-step = the eval contract:** day+1 uses **real** lags/rolling (history ≤
  cutoff); for day+2…+14 the in-window lags/rolling are filled with the model's **own
  predictions**, fed back step by step — **never future actuals, never null**. Mirrors
  production. Genuine warm-up NaNs (first ~35 days/store) are handled **natively by the trees**
  (LightGBM/XGBoost learn a default split direction — no imputation).
- **Seasonal stratification:** ≥1 fold trains through the 2016 Golden Week (so the trees learn
  the spike); error is reported **by season** (and **by prefecture**), not hidden in an average.
- **Visual:** rolling-origin split diagram (overlapping, stride-18) + the error-by-horizon curve.

## Slide 8 · 4-step selection pipeline + tree params — MED · 1:30 · → crit 1, 2
- Pipeline (`ModelSelector` — one file per step):
  1. **Rank** — CV all 5 models on the **full** feature set with **default params** → rank by
     the metric. Picks who drives step 2.
  2. **Feature selection** — on the step-1 winner (or best tree if a naive won): keep features by
     **cumulative tree gain to 0.98** (`FeatureSelector`) → **17 of 35** features this run.
  3. **Grid search** — re-CV **all** models on the **selected** features, each over a small grid
     (the grid always contains the step-1 default). **The winner is decided here** and can differ
     from step 1.
  4. **Report + deploy** — reuse step-3's retained fold predictions (**no extra CV pass**) for the
     cross-model report, then **refit the winner on ALL data** (cv + holdout) → the shipped model.
- Honest **seasonal-naive (lag_7)** baseline everything must beat.
- Tree knobs (name knob + effect): LGBM/XGB tune **lr (0.05↔0.1)** = speed/fit and
  **depth/leaves (6↔8 / 31↔63)** = capacity/overfit; log1p target, native categoricals.
- Brief asked for **2** models — we built **5** spanning heuristic → tabular ML → foundation model.
- **Visual:** 4-step pipeline diagram.

## Slide 9 · TimesFM — DEEP · 2:30 · → crit 2, 5
The centerpiece.
- **What:** Google **TimesFM 2.5**, 200M-param decoder-only foundation model for time series,
  pretrained, outputs point + 10 quantiles, **no training**. Context fed: the most-recent
  ~448 days per series (bucketed to fixed compiled lengths — see Q&A).
- **Use A — zero-shot:** forecast the whole horizon directly, no fit, no covariates.
- **Use B — hybrid (centerpiece):** TimesFM's **whole-horizon window signal** (`tfm_point` +
  `tfm_q0..q9`, one row per days-ahead step) becomes a feature for **XGBoost** alongside the
  engineered covariates. TimesFM forecasts the window **once at the cutoff** (out of the recursive
  loop); only the engineered lags recurse. *"TimesFM digests trend/seasonality; XGBoost learns how
  the covariates modulate it."*
- **The clean ablation — `xgboost` vs `timesfm_hybrid`:** same base learner, the only difference
  is the TimesFM signal, so the gap **is** TimesFM's marginal value (not a leaderboard number).
- **Engineering (½ bullet):** the transformer never runs inside the recursive loop; each
  whole-window forecast is content-addressed to a **per-series disk cache** (`.cache/timesfm`) so
  a forecast hits the GPU **once, ever** — across folds, steps, both TimesFM models and reruns.
  Served from a GPU **Hugging Face Space** via `FORECAST_TIMESFM_ENDPOINT`; the pipeline needs only
  `requests` (no torch, no checkpoint). The cache let a full re-run finish in ~20 min with **0 GPU
  calls**.
- **Visual:** hybrid data-flow (history → TimesFM whole-window signal → XGBoost → recursive lag
  feedback) **+ 📸 a screenshot of the live Hugging Face Space** (the deployed GPU server). The
  Space code is `space/app.py` (+ `space/Dockerfile`); the client adapter is
  `src/forecasting/adapters/models/timesfm/remote.py`; probe it with
  `scripts/check_timesfm_endpoint.py` (or `make check-endpoint`). Endpoint set via
  `FORECAST_TIMESFM_ENDPOINT` in `.env` (template: `.env.example`).

## Slide 10 · Results & graphics — MEAT · 2:30 · → crit 4
Open `artifacts/report/index.html` live.

**Final holdout (39 days, 2017-03-15→04-22), sorted by the business metric `weighted_mae`:**

| Model | RMSLE | MAE | weighted_MAE |
|---|--:|--:|--:|
| **timesfm_hybrid** | **0.514** | **7.30** | **9.93** |
| timesfm_zeroshot | 0.618 | 7.71 | 9.99 |
| xgboost *(deployed)* | 0.520 | 7.33 | 10.02 |
| lightgbm | 0.534 | 7.41 | 10.14 |
| seasonal_naive *(baseline)* | 0.857 | 9.88 | 14.47 |

**The honest story (this is the differentiator — lead with it):**
- Everything **crushes the naive baseline** (~31% better weighted_mae) — the floor is real.
- **The deployed winner is `xgboost`, but on the holdout `timesfm_hybrid` is actually best.** Not
  a contradiction: **selection happens on CV, never on the holdout** (you can't peek at your
  out-of-sample set). On CV the trees and the hybrid are **tied** (~9.97 weighted_mae) and xgboost
  ranked first; the holdout is the *honest* check, not the selector. Selecting on the holdout would
  be the exact leakage we designed against.
- **The ablation (xgboost → timesfm_hybrid):** holdout weighted_mae **10.02 → 9.93** (−0.9%),
  RMSLE **0.520 → 0.514** (−1.2%); on CV the gap is ~0. So **TimesFM's marginal value here is small
  and positive** — real but not a game-changer on this data/horizon. That's the point of an
  ablation: it *measures* the value, it doesn't assume it.
- Nuance worth a sentence: `timesfm_zeroshot` has the worst RMSLE of the non-naive models (0.618)
  yet a competitive `weighted_mae` (9.99) — it errs in a way the asymmetric business cost likes.
- **The charts back each claim** (all in `artifacts/report/`, bundled in `index.html`):
  - `holdout_scores.png` — the leaderboard above, all 3 metrics, 2 rows (with / without the naive
    so the close models are readable).
  - `pred_vs_real.png` — aggregate + 2 sample stores, one row per model family.
  - `error_by_horizon.png` & `seasonal.png` — 2 rows (with/without naive); error grows with
    days-ahead, broken down by season.
  - `error_by_prefecture.png` — where each model wins/loses geographically.
  - `feature_importance_*.png` — `is_closed` dominates; the TimesFM signal columns rank mid-pack
    in the hybrid (consistent with the small ablation delta).

## Slide 11 · Scaling — MED · 1:15 · → crit (scaling, explicitly valued)
- Already scaled in shape: **one global model** with store_id / genre / area as features
  (cross-learning, not one-model-per-series).
- TimesFM zero-shot handles **cold-start** for brand-new centers.
- Real Gstock = centers × products = millions of intermittent series → hierarchical/grouped
  forecasts + reconciliation (MinT), sparse-demand handling, **batch** (not live) ordering.
- The `DataSource` port = swap CSV for the warehouse in one file.
- **Visual:** single-global-model vs per-series + a hierarchy sketch.

## Slide 12 · Packaging & ops — SKIM · 0:45 · → crit 3
- `docker compose up` runs CV → selection → plots → MLflow with **zero manual steps**; CPU-only
  image, data baked in. TimesFM called over HTTP (no model in the image).
- K8s **Job** (one-off train/select) + **CronJobs** (weekly retrain, nightly batch predict — the
  auto-ordering shape).
- **MLflow** via an injected `ExperimentTracker` **port** (domain logs without importing MLflow;
  `sqlite:///mlflow.db`). Every run is tagged with a per-CV-run **`cv_run_id`** (uuid) and logs
  **horizon + folds**; `report-summary` carries the winner + holdout metrics; plots + `selection.json`
  + `best_model.pkl` are attached.
- **Reproducibility wins to name-drop:** a per-run **`logs/<run_id>.log`**, a persisted
  **`report.pkl`** + a **`render-report`** command (re-render every chart in seconds, no model
  run, no GPU), and **retry-with-backoff** on the TimesFM endpoint so a transient blip never tears
  down an hours-long run.
- **Visual:** one ops diagram or an MLflow-UI screenshot.

## Slide 13 · Close + what's next — 0:30
- On the shelf: probabilistic outputs → service-level ordering, weather/promo covariates,
  hierarchical reconciliation, TimesFM fine-tuning (the ablation says there's a small signal to
  grow).
- Close on the brief's own line: *"We don't chase a perfect model — we reason, structure, and
  communicate."*

---

## Depth strategy
- **Deep (most points):** 7 (validation/recursive), 6 (metrics), 9 (TimesFM + ablation),
  10 (results) → criteria 2, 4, 5.
- **Medium:** 2, 4, 5, 8, 11.
- **Skim, one slide max:** 1, 3, 12. Resist spending 5 min on hexagonal architecture and Docker.

---

## Asset & path reference (where everything lives)

**📸 Screenshots to capture for the deck**
- **Architecture:** the `src/forecasting/` tree / the README "Architecture (hexagonal)" block.
- **Hugging Face deployment:** the live Space page (the GPU server running `space/app.py`); the
  `/health` response or a `make check-endpoint` run also works as proof-of-life.

**Code — by concept (slide → path)**
- Architecture / rules → `AGENTS.md`, `src/forecasting/` (`domain/`, `domain/ports/`, `adapters/`,
  `application/`, `entrypoints/cli.py`).
- Config & knobs (metric weights, threshold, stride, horizon) → `src/forecasting/config.py`.
- Data adapter (8-CSV join, closed-day policy) → `src/forecasting/adapters/data/recruit_csv.py`.
- Features (7 families, leakage discipline) → `src/forecasting/domain/services/features/`
  (`builder.py`, `selector.py`).
- Metrics (RMSLE / MAE / weighted_mae) → `src/forecasting/domain/services/evaluation/metrics.py`.
- Temporal validation (rolling-origin, holdout, stride) → `src/forecasting/domain/validation.py`.
- Recursive multi-step contract → `src/forecasting/domain/services/predictor.py`.
- 4-step selection pipeline → `src/forecasting/domain/services/model_selection/`
  (`step1_ranking.py` · `selector`/step2 · `step3_grid_search.py` · `step4_report.py` · `main.py`).
- Models (5 behind one port) → `src/forecasting/adapters/models/` (`seasonal_naive.py`,
  `lightgbm_model.py`, `xgboost_model.py`, `timesfm/{zeroshot,hybrid,features,remote}.py`).
- Plotting / report → `src/forecasting/adapters/plotting.py`.
- MLflow tracker (port + adapter) → `src/forecasting/domain/ports/tracker.py`,
  `src/forecasting/adapters/mlflow_tracker.py`.

**TimesFM / Hugging Face**
- Space server → `space/app.py`, `space/Dockerfile`.
- Client adapter (HTTP + per-series cache) → `src/forecasting/adapters/models/timesfm/remote.py`.
- Probe → `scripts/check_timesfm_endpoint.py` (`make check-endpoint`).
- Endpoint config → `.env` (template `.env.example`): `FORECAST_TIMESFM_ENDPOINT` (+ `_TOKEN`).
- Forecast cache → `.cache/timesfm/` (content-addressed `.npz` per series).

**Results & artifacts (open during slide 10)**
- The report → **`artifacts/report/index.html`** (+ PNGs: `holdout_scores`, `pred_vs_real`,
  `seasonal`, `error_by_horizon`, `error_by_prefecture`, `feature_importance_*`; CSVs:
  `by_horizon_*`, `by_season_*`).
- Persisted report (for `render-report`) → `artifacts/report.pkl`.
- Winner + metrics → `artifacts/selection.json`; deployable model → `artifacts/best_model.pkl`.
- Per-run log → `logs/<run_id>.log`; MLflow store → `mlflow.db` (`mlflow ui` → http://127.0.0.1:5000).

**Run / ops commands**
- Full run (what produced the results): `FORECAST_OVER_WEIGHT=2.0 FORECAST_UNDER_WEIGHT=1.0
  FORECAST_CV_STRIDE_DAYS=18 python -m forecasting find-best-model --metric weighted_mae`.
- Re-render charts only (no model run): `python -m forecasting render-report`.
- One-command demo: `docker compose up` (Docker) → `Dockerfile`, `docker-compose.yml`, `Makefile`.
- K8s → `k8s/train-job.yaml`, `k8s/retrain-cronjob.yaml`, `k8s/predict-cronjob.yaml`.
- EDA charts → `notebooks/01_eda.ipynb`.
- Supporting docs → `docs/00-overview.md` … `docs/06-scaling.md`, `docs/adr/`.

---

## Q&A prep (appendix — not presented)
- **Winner is xgboost but hybrid is better on the holdout — why ship xgboost?** Selection is by
  CV (you must not select on the held-out set). On CV they tie and xgboost ranks first; the
  holdout is the unbiased check, not the chooser. Picking on the holdout = leakage.
- **So TimesFM didn't help?** It helped a little — holdout weighted_mae −0.9%, RMSLE −1.2% vs
  plain xgboost; ~0 on CV. The ablation reports the marginal value honestly instead of assuming it.
- **RMSLE over MAPE?** MAPE blows up near zero-demand days; RMSLE is symmetric in log space and
  matches the benchmark.
- **Why penalise over-ordering, not stockout?** Business choice for perishables (waste-dominant);
  it's a config knob (`under_weight`/`over_weight`) — flip for a stockout-averse business.
- **Recursive vs direct per-horizon?** One model, honest error propagation, matches production
  (re-forecast daily).
- **How is leakage prevented?** Lags shifted, rolling `shift(1)`, store aggregates fold-train-only,
  holdout carved first, selection on CV only.
- **Why stride coprime with 7 (this run 18)?** A step that's a multiple of 7 locks every fold to
  one weekday → by-horizon error confounded with day-of-week. Coprime ⇒ origins rotate through all
  weekdays. Cost: overlap → correlated folds. Configurable (`cv_stride_days`).
- **Recursive lags — null mid-forecast?** No. Day+1 real history; day+2…+14 feed the model's own
  predictions back; warm-up NaNs handled natively by trees.
- **Feature-selection threshold?** Cumulative tree gain to **0.98** (17 of 35 features here);
  generous on purpose — step 3 re-CVs on the reduced set so a bad cut shows up. Configurable.
- **How much context does TimesFM see?** Up to ~448 recent days per series, bucketed to fixed
  compiled context lengths (multiples of the 32-pt patch) to keep inference batchable and avoid the
  short-input NaN issue. Short series use all they have; long ones use the most-recent ~448 days.
- **TimesFM cost in production?** Content-addressed per-series cache (disk today, shared feature
  store at scale) + GPU endpoint; nightly CronJob, not a live API. A full re-run was 0 GPU calls.
- **What's logged to MLflow?** `report-{model}` (overall + per-season + per-prefecture +
  rmsle_h1..14 + CSVs + importance PNG) and `report-summary` (winner params, **horizon/folds**,
  holdout metrics, cross-model PNGs, `selection.json`, `best_model.pkl`) — all tagged `cv_run_id`.

---

## Self-check before the talk
- **Time:** read aloud against a timer; core ≤ ~18 min; trim bullets, not slides.
- **Coverage:** every eval criterion hit by ≥1 deep slide (1→s1/2, 2→s6/7, 3→s3/12, 4→s10, 5→s9).
- **Results gate:** slide 10 — open `index.html`, lead with the naive-beat + the
  selection-vs-holdout nuance + the ablation delta. This is the section that draws questions.
- **Honesty:** the xgboost-vs-hybrid story is a strength, not a weakness — it shows you understand
  *why* you don't select on the holdout. Rehearse it.
