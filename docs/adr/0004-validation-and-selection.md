# ADR 0004 — Temporal validation & the model-selection pipeline

**Status:** accepted · **Date:** 2026-06-28

## Context

Forecasting demand to feed automatic ordering. The evaluation must be **honest about
production**: you never have tomorrow's actuals when you order today, and the metric you rank
on must reflect the ordering cost. A naive K-fold or a single train/test split would leak the
future and over-state accuracy. We also want a comparison across 5 models that is reproducible
and that isolates each modelling choice. Full reasoning in [`../04-evaluation.md`](../04-evaluation.md).

## Decision

A pure-domain validation + selection service (`domain/validation.py` +
`domain/services/model_selection/`), with four properties:

1. **Holdout carved off first.** The last `final_horizon_days` (39 = the 2017 Recruit window,
   incl. Golden Week) is set aside before any CV and never touched. Only the final deployable
   model is refit *including* it.
2. **Rolling-origin, expanding-window CV** on the earlier history. A split is three dates
   (`train_end < valid_start ≤ valid_end`), never row indices, never shuffled. Each fold trains
   on **all** history up to its cutoff (the train set grows fold to fold) and validates the next
   `horizon` (14) days.
3. **Stride coprime with 7 (default 9).** Consecutive fold origins step `cv_stride_days` days,
   not `horizon`. Because `horizon` is a multiple of 7, stepping by the horizon would start
   every fold on the **same weekday** and confound the by-horizon error with day-of-week. A
   stride coprime with 7 rotates the origin through every weekday. `n_folds =
   cv_window_days // cv_stride_days` (≈ the last year). The pure function defaults `stride_days`
   to `horizon` (classic non-overlapping CV); the **app default is 9**.
4. **Recursive multi-step as the eval contract.** For day+1 the lag/rolling features use real
   history ≤ cutoff; for day+2…+14 the in-window lags/rolling are filled with the model's **own
   predictions**, fed back step by step — never future actuals, never null. Genuine warm-up
   NaNs (first ~35 days/store) are left for the trees to handle natively.

The selection runs as a **4-step pipeline** (`ModelSelector`, one file per step):

1. **Rank** every model on the full feature set with default params (no grid) → pick the best.
2. **Feature-select** on that winner — or the best *tree* if a naive model won — by cumulative
   tree gain to `feature_select_threshold` (0.95).
3. **Grid-search** all models on the selected features (each grid contains its step-1 config);
   the final winner is decided here.
4. **Report + deploy** — reuse step-3 fold predictions for the cross-model report (no extra CV),
   then refit the winning config on all data → the deployable model.

Ranking metric is **RMSLE** (relative, count-appropriate, matches the benchmark), reported
alongside **MAE** (readable) and **weighted_mae** (asymmetric stockout-vs-waste cost).

## Consequences

- The reported error is a faithful n-day-ahead estimate; no leakage path survives (holdout
  first, per-fold `reference` for target aggregates, recursive lags).
- Stride 9 makes the by-horizon curve interpretable, at the cost of **overlapping (correlated)
  folds** — variance is slightly understated — and **more folds to compute** (≈40 at stride 9
  vs 26 at a non-overlapping stride=14). Both are tunable: raise `cv_stride_days`, or cap with
  `--folds`.
- The 4-step split keeps each phase a small, testable class; the winner can differ from step 1
  because the feature set changed between them.
- Every (model × config) run, the feature-selection run, and the report are logged through the
  injected `ExperimentTracker` (MLflow), so the whole comparison is browsable and reproducible.
