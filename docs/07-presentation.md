# 07 · Presentation Outline

Working structure for the live technical-test presentation. **Slot: ~15-20 min
talk + Q&A. Language: English.** This is a structure doc — slide tooling
(Marp/pptx) comes later.

Each slide: **tier** · time budget · talking points · required visual · eval
criterion. Tiers — **DEEP** (differentiator, spend time) / **MED** / **SKIM**
(table-stakes, one slide max, don't narrate).

Eval criteria: (1) problem framing · (2) validation & metric rigor · (3) code
quality / reproducibility · (4) communication via graphics · (5) technical
depth in Q&A.

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

**Core ≈ 17.5 min.** Squeezed to 15: merge 8→9, cut 3 to a 15-sec aside. Given
25+: add a 2nd results slide and a live `docker compose up` / MLflow-UI demo.

---

## Slide 1 · Framing — SKIM · 0:45 · → crit 1
- Problem: forecast **daily demand per center** to feed automatic supplier
  orders.
- One line on Gstock's real case (centers × products → auto-orders), then the
  Recruit dataset as a reduced-scale proxy.
- Sets the *why* before any model talk.
- **Visual:** one-line problem statement + a single demand curve.

## Slide 2 · Why Recruit — MED · 1:00 · → crit 1
- Natively **multi-series** (~829 stores, Jan 2016–May 2017) = maps to
  "comensales por centro."
- Real feature signals: reservations, holidays, genre/area.
- Built-in **RMSLE benchmark** + defined holdout.
- Best fit for a foundation model (needs many real seasonal series).
- Rejected alts, one line: Hotel Booking = classification / ~2 daily series;
  Restaurant Revenue = 137 rows, not a series.
- **Visual:** tiny 3-row comparison table (dataset → why kept/dropped).

## Slide 3 · Repo & architecture — SKIM · 0:45 · → crit 3
- Hexagonal: **pure domain** (features / metrics / validation, no IO) vs
  **adapters** (IO + model libs) vs **one CLI** composition root.
- `AGENTS.md` encodes the rules (no leakage, recursive eval contract, log-domain
  safety).
- Headline: **"5 models behind one `Model` port."**
- **Visual:** folder tree (domain / ports / adapters / application).
- Don't dwell — show it, move on.

## Slide 4 · The data, honestly — MED · 1:30 · → crit 1, 2
- 8 CSVs (AIR point-of-sale, HPG reservations, relation, store info, date info)
  joined to canonical `store_id / date / visitors`.
- The honest gotcha: **~44k closed-day rows are ABSENT** (~15% of the calendar,
  coverage ~0.85) — not zeros, *missing*.
- Decision: reindex each store to a full daily calendar, **fill `visitors=0`
  AND set `is_closed=1`** — model distinguishes "closed" from "open but zero,"
  and lags stay calendar-aligned.
- **Visual:** 1-2 EDA charts from `notebooks/01_eda.ipynb` — weekly seasonality
  + Golden Week spike.

## Slide 5 · Feature engineering — MED · 1:30 · → crit 1
- 7 families: **calendar** (dow, weekend, month, day-of-month payday proxy,
  sin/cos day-of-year), **holiday** (+Golden Week window), **closed-flag**,
  **lags at multiples of 7** (1,7,14,21,28,35), **rolling mean+median**
  (7,14,28,35), **reservations** (leading indicator: count, visitors,
  lead-time), **store stats** (genre, area, per-store & per-store×dow means).
- Emphasize two things:
  - **(a) log1p target** — count data, RMSLE-safe.
  - **(b) leakage discipline** — lags shifted, rolling uses `shift(1)`, store
    aggregates built fold-train-only.
- On the shelf for v2: weather, location clustering, promo/price, outlier
  capping, extra rolling stats.
- **Visual:** the 7-family table + a "shifted lag" mini-diagram.

## Slide 6 · Metrics & why — DEEP · 1:30 · → crit 2
- **RMSLE primary:** count data, symmetric in log space, matches the Recruit
  benchmark, safe at 0 via log1p.
- MAE for interpretation (the human-readable companion).
- The domain flex: **`weighted_mae`** — under-predict (stockout) costs **2×**,
  over-predict (waste) costs **1×** → aligns the metric with the ordering
  business.
- All predictions clipped ≥0 before any metric.
- **Visual:** metric table + the asymmetric-cost intuition (stockout vs waste).

## Slide 7 · Temporal validation + recursive contract — DEEPEST · 2:30 · → crit 2
The rigor showcase — spend the most time here.
- **Holdout carved off first:** the last `final_horizon_days` (**39** = the length
  of the official Recruit test window) of labeled data — labels end 2017-04-22, so
  it's 2017-03-15→04-22 (Golden Week sits in the unlabeled scoring window). CV never
  touches it; only the final deployable model is refit *including* it.
- **Rolling-origin, expanding-window CV** on the earlier history: a split is
  **three dates, not row indices** (`train_end < valid_start ≤ valid_end`), **no
  shuffle**. Each fold trains on **all** history up to its cutoff (so the train
  set *grows* fold to fold) and validates the next `horizon` (14) days.
- **The stride trick (the detail to land):** consecutive fold origins step
  **9 days** (`cv_stride_days`), not 14. 9 is **coprime with 7**, so the forecast
  origin **rotates through every weekday**. If the step equalled the horizon (a
  multiple of 7), every fold would start on the *same* weekday → `horizon_offset=1`
  would always be that weekday and the **by-horizon error would be confounded with
  day-of-week**. Cost of the overlap (be honest): folds reuse target days →
  correlated → variance slightly understated, + more folds to compute.
  `n_folds = cv_window_days // cv_stride_days` (≈40 over the last year; `--folds`).
- **Recursive multi-step = the eval contract:** day+1 uses **real** lags/rolling
  (history ≤ cutoff); for day+2…+14 the in-window lags/rolling are filled with the
  model's **own predictions**, fed back step by step — **never future actuals,
  never null**. Mirrors production (no tomorrow's actuals today). The genuine
  warm-up NaNs (first ~35 days/store, no past yet) are handled **natively by the
  trees** (LightGBM/XGBoost learn a default split direction — no imputation).
- **Seasonal stratification:** ≥1 fold trains through the 2016 Golden Week (so the
  trees learn the spike); error is reported **by season**, not hidden in an average
  — "the difference between a metric and an operational insight." (A finer
  holiday-window cut is available via the `golden_week` feature if wanted.)
- **Visual:** rolling-origin split diagram (overlapping, stride-9) + error-by-
  horizon curve.

## Slide 8 · 4-step selection pipeline + tree params — MED · 1:30 · → crit 1, 2
- Pipeline (`ModelSelector` — one file per step, classes throughout):
  1. **Rank** — CV all 5 models on the **full** feature set with their **default
     params** (no grid yet) → rank by the metric. Picks who drives step 2.
  2. **Feature selection** — on the step-1 winner, **or the best tree if a naive
     model won** (naive has no importances): keep features by **cumulative tree
     gain to 95%** (`FeatureSelector`).
  3. **Grid search** — re-CV **all** models on the **selected** features, each over
     a small grid (the grid always **contains the step-1 default**, so that exact
     config is re-tried on the reduced set). **The final winner is decided here —
     it may differ from step 1.**
  4. **Report + deploy** — reuse step-3's retained fold predictions (**no extra CV
     pass**) for the cross-model report (→ slide 10), then **refit the winning
     config on ALL data** (cv + holdout) → the deployable model we ship.
- Honest **seasonal-naive (lag_7)** baseline everything must beat.
- Tree knobs, crisp — name the knob + what it does: both LGBM/XGB use **300
  trees, lr 0.05, subsample/colsample 0.8**, log1p target, native categoricals;
  we tune **lr (0.05↔0.1)** = speed vs fit and **depth/leaves (6↔8 / 31↔63)** =
  capacity vs overfit.
- Note: brief asked for **2** models — we built **5** spanning heuristic →
  tabular ML → deep foundation model.
- **Visual:** 4-step pipeline diagram.

## Slide 9 · TimesFM — DEEP · 2:30 · → crit 2, 5
The centerpiece.
- **What:** Google **TimesFM 2.5**, 200M-param decoder-only foundation model for
  time series, pretrained, outputs point + 10 quantiles, no training needed.
- **Use A — zero-shot:** forecast the whole horizon directly, no fit. Free but
  naive (no covariates).
- **Use B — hybrid (centerpiece):** TimesFM's **whole-horizon window signal**
  (`tfm_point` + `tfm_q0..q9`, one row per days-ahead step) becomes a feature for
  **XGBoost** alongside engineered covariates. TimesFM forecasts the window **once
  at the cutoff** (out of the recursive loop); only the engineered lags recurse.
  *"TimesFM digests trend/seasonality; XGBoost learns how covariates modulate it."*
- **The clean ablation:** **`xgboost` vs `timesfm_hybrid`** — same base learner,
  only difference is the TimesFM signal → the RMSLE gap **is** TimesFM's marginal
  value (not a leaderboard number).
- **Engineering aside (½ bullet):** Transformer never runs inside the recursive
  loop — the whole-window signal is forecast **once per origin**, memoised
  in-memory and content-addressed to a **per-series disk cache** (`.cache/timesfm`)
  so each unique forecast hits the GPU once, ever. TimesFM is served from a dedicated
  GPU **Hugging Face Space**; the pipeline calls it via `FORECAST_TIMESFM_ENDPOINT`
  and needs only `requests` (no torch, no checkpoint). Honest framing: "no GPU
  budget for full sweeps, so the design offloads it cleanly — ~6 min/call on CPU
  vs seconds on the Space."
- **Visual:** hybrid data-flow (history → TimesFM whole-window signal → XGBoost →
  recursive lag feedback).

## Slide 10 · Results & graphics — MEAT · 2:30 · → crit 4
The comparison the brief explicitly rewards.
- **Leaderboard table:** RMSLE / MAE / weighted_mae per model (naive → trees → TimesFM
  zero-shot → hybrid).
- **Pred vs real** (1-2 representative stores).
- **Residuals.**
- **Error-by-horizon** (where each model degrades over the 14-day horizon).
- **Feature importance** (where the TimesFM signal ranks in the hybrid).
- **By-season** error split.
- Land the **ablation delta** as the headline.
- ⚠ **Depends on plots being generated — see Dependencies.**

## Slide 11 · Scaling — MED · 1:15 · → crit (scaling, explicitly valued)
- Already lowkey scaled: **one global model** with store_id / genre / area as
  features (cross-learning, not one-model-per-series).
- TimesFM zero-shot handles **cold-start** for brand-new centers.
- Real Gstock = centers × products = millions of intermittent series →
  hierarchical/grouped forecasts + reconciliation (MinT), sparse-demand
  handling, **batch** (not live) ordering.
- The `DataSource` port = swap CSV for warehouse in one file.
- **Visual:** single-global-model vs per-series, + a hierarchy sketch.

## Slide 12 · Packaging & ops — SKIM · 0:45 · → crit 3
- `docker compose up` runs train → CV → plots → MLflow with **zero manual
  steps**.
- TimesFM served from a Hugging Face Space; the pipeline calls it over HTTP (no model bundled in the image).
- K8s **Job** for one-off train, **CronJob nightly @ 02:00** for batch predict
  (the auto-ordering shape).
- **MLflow** experiment tracking via an injected `ExperimentTracker` **port** (so
  the pure domain logs without importing MLflow; `sqlite:///mlflow.db`, browse with
  `mlflow ui`). What lands there: a run per **(model × config)** in steps 1 & 3,
  the **feature-selection** run (+ `selected_features.json`), and `report-{model}`
  + `report-summary` with **per-season / per-horizon / holdout** metrics and the
  artifacts (`selection.json`, `best_model.pkl`, plots). Full breakdown in Q&A.
- **Visual:** one ops diagram or an MLflow-UI screenshot.

## Slide 13 · Close + what's next — 0:30
- On the shelf: probabilistic outputs → service-level ordering, weather/promo
  covariates, hierarchical reconciliation, TimesFM fine-tuning.
- Close on the brief's own line: *"We don't chase a perfect model — we reason,
  structure, and communicate."*

---

## Depth strategy (where to go deep)
- **Deep (win the most points):** 7 (validation/recursive), 6 (metrics),
  9 (TimesFM + ablation), 10 (results) → criteria 2, 4, 5.
- **Medium:** 2, 4, 5, 8, 11.
- **Skim, one slide max:** 1, 3, 12. Resist the common failure of spending 5 min
  on hexagonal architecture and Docker.

### How the ~17 originally-listed topics map
- Scaffolding / AGENTS.md → slide 3.
- Init analysis / nulls + data prep / 0-imputation → merged into slide 4.
- Naive/XGB/LGBM params + 4-step selection pipeline → merged into slide 8.
- TimesFM (what / solo / combined) + HuggingFace server → merged into slide 9
  (HF offload is a half-bullet, not its own section).
- "Runs results, graphics" (listed twice) → slide 10.
- Docker deployment + scheduling → slide 12.

---

## Dependencies & risks
1. **Results plots don't exist yet (top risk).** Slide 10 needs pred-vs-real,
   residuals, error-by-horizon, feature importance, normal-vs-GW split. Needs
   Phase 10 finished (`src/forecasting/adapters/plotting.py` — confirm it emits
   these) + a full eval run. `mlflow.db` already holds partial run metrics.
   Decide: generate real plots, or present leaderboard + a subset and point to
   "full plot suite in the repo."
2. **TimesFM availability.** The hybrid's signal comes from the GPU Hugging Face
   Space (`FORECAST_TIMESFM_ENDPOINT`); if it's unreachable, the ablation number
   can't be computed — present the methodology with the partial MLflow number and
   say so honestly in Q&A.
3. **Notebook stub.** `notebooks/02_model_experiments.ipynb` is empty; fill it
   only if you want results to come from a runnable notebook (reproducibility
   points). Optional for the talk.

---

## Q&A prep (appendix — not presented)
- **RMSLE over MAPE?** MAPE blows up near zero-demand days; RMSLE is symmetric
  in log space and matches the benchmark.
- **Recursive vs direct per-horizon?** One model, honest error propagation,
  matches production where you re-forecast daily.
- **How exactly is leakage prevented?** Lags shifted, rolling `shift(1)`, store
  aggregates fold-train-only, holdout carved first.
- **Why not ARIMA/Prophet?** Per-series, ignores cross-series signal; trees +
  global model + foundation model cover more ground.
- **TimesFM delta ~0 or negative?** Valid result — the ablation *measures*
  marginal value, doesn't assume it; honesty is the point per the brief.
- **Brand-new restaurant, no history?** TimesFM zero-shot cold-start; trees fall
  back to genre/area/dow priors.
- **TimesFM cost in production?** Content-addressed per-series cache (disk today,
  shared feature store at scale) + GPU endpoint; nightly CronJob, not a live API.
- **Why stride 9 in the CV (not step = horizon)?** Step = horizon is a multiple of
  7 → every fold starts on the same weekday and the by-horizon error gets
  confounded with day-of-week. Stride 9 is coprime with 7, so origins rotate
  through all weekdays. Accepted cost: overlapping windows → correlated folds
  (variance slightly understated) + more folds. Configurable (`cv_stride_days`);
  set it = horizon for classic non-overlapping CV.
- **During the recursive forecast, what happens to lags/rolling — null?** No. Day+1
  uses real history; day+2…+14 feed the model's *own* predictions back into the
  in-window lags/rolling (never future actuals, never null). Separately, genuine
  warm-up NaNs (first ~35 days/store) are handled natively by the trees — no
  imputation, no leakage.
- **Does step 3 re-CV only the step-1 winner?** No — **all** models, on the
  selected features, each over its grid. The final winner comes from step 3 and
  can differ from step 1 (the feature set changed). The deployable model is that
  winner refit on **all** data, holdout included.
- **Feature-selection threshold?** Cumulative tree gain to **0.95** (`FeatureSelector`),
  deliberately generous — step 3 re-CVs on the reduced set, so a bad cut shows up
  there. Configurable via `feature_select_threshold`.
- **What exactly is logged to MLflow?** `step1-{model}` (one/model) and
  `step3-{model}` (one **per grid combo**): params + ranking metric + RMSLE/MAE/
  weighted_mae suite. `step2-feature-selection`: base model, n_selected, threshold
  + `selected_features.json`. `report-{model}`: overall + per-season +
  `rmsle_h1..h14` + CSVs + importance PNG. `report-summary`: winner params, holdout
  metrics, cross-model PNGs, `selection.json`, `best_model.pkl`. Domain logs the
  mechanical runs via the port; the use case logs the report runs.

---

## Self-check before the talk
- **Time:** read aloud against a timer; core ≤ ~18 min; trim bullets, not slides.
- **Coverage:** every eval criterion hit by ≥1 deep slide (1→s1/2, 2→s6/7,
  3→s3/12, 4→s10, 5→s9 + Q&A).
- **Mapping:** all ~17 listed topics present (merged per above) — nothing
  silently dropped.
- **Results gate:** slide 10 has real numbers/plots (or an explicit, honest
  placeholder) — the section most likely to draw questions.
