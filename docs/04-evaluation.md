# 04 — Evaluation

Implemented in `domain/services/evaluation/` (metrics) and `domain/validation.py` (temporal splits).

## Metrics

| Metric | Why |
|---|---|
| **RMSLE** (primary) | Matches the Recruit benchmark; right for **count** data; scores **relative** error in log space, so it doesn't let a few huge-volume stores dominate. The ranking metric |
| MAE | Interpretable in raw visitors ("off by N comensales") — the human-readable companion |
| **weighted_mae** | **Asymmetric** MAE: under-prediction (stockout / lost sales) and over-prediction (waste / spoilage) cost differently. `under_weight`/`over_weight` from config — the business view |

All three are computed and logged on every CV run. **Which one drives model selection is
configurable** (`find_best_model --metric ...`, default `rmsle`): RMSLE for a leaderboard-
comparable run, or `weighted_mae` when the business cost of running out is worse than the cost
of waste (the usual case for supplier ordering).

Why RMSLE over MAPE: MAPE explodes on the low-count / zero days that restaurants have all the
time. RMSLE via `log1p` is well-behaved there.

## Log-domain safety (a correctness requirement, not a nicety)

Counts include zeros and near-zeros, and trees/TimesFM can emit small **negatives**. Naively,
`RMSLE` would compute `log(0)` or `log(negative)` → a hard crash (`math domain error`). The
domain guards this in three places:

1. train trees in **`log1p`** space;
2. invert predictions with **`expm1`**;
3. **`clip_nonneg` (`max(0, ŷ)`) before any metric** — see `evaluation.clip_nonneg`.

This is covered by `tests/domain/test_evaluation.py` (negative prediction → finite RMSLE).

## Temporal validation

Both in `domain/validation.py`. **The holdout is carved off first**
(`final_holdout`) — the last `final_horizon_days` (**39**, the length of the official Recruit
test window) of *labeled* data (labels end 2017-04-22, so the holdout is 2017-03-15→04-22) —
and is reserved for the final pred-vs-real forecast only. Rolling-origin CV
(`rolling_origin_splits`) then runs on the *earlier* history alone, so it never touches the
holdout. The **CV horizon** is a separate, shorter operational window: `config.horizon_days`
(default **14**) or `--horizon`. The CV reads roughly the **last year** — the use case derives
`n_folds = cv_window_days // cv_stride_days` (~40 at the default stride 9; override with
`--folds`) — so the report can break error down seasonally.

- **Rolling-origin (expanding window):** train on `[start, cutoff]`, validate on the next
  `horizon` days, then advance the cutoff. Mimics production (you only ever know the past).
  Each split is a triple of **date boundaries** (`train_end`, `valid_start`, `valid_end`),
  so callers slice their frame by date — leakage-safe by construction.
- **Recursive multi-step scoring (the eval contract).** A fold doesn't predict its valid
  window in one shot — that would leak intra-window actuals into the lags. `domain/services/
  Predictor` forecasts the `horizon` days one at a time, feeding each prediction back into the
  next day's lags/rolling (and, for the hybrid, TimesFM's input). We score the predictions
  against the **observed** valid days (closed days, absent from the panel, aren't scored — you
  know your own closures, so predicting demand you'd never serve isn't the test).
- **Never** random K-fold on a time series — it leaks the future into training.
- Report **error-by-horizon** (RMSLE vs days-ahead): forecasts should degrade with distance,
  and the shape tells you where each model breaks down.

## The comparison report (step 4)

After selection, `domain/services/model_selection/step4_report.build_report` assembles a
cross-model `ComparisonReport` (numbers only; plots in `adapters/plotting.render_report`,
MLflow logging in the use case). No second CV pass — it **reuses the retained step-3 fold
predictions** (the last year of 14-day CV), and adds one final forecast per model. Per model:

- **Overall + by-season + by-prefecture + by-horizon**, all (the full RMSLE/MAE/weighted_mae
  suite) from the retained CV predictions. The seasonal view (`evaluation.season(date)`,
  N-hemisphere by month) is the "descomposición estacional" — *performance by season*, not a
  statsmodels decomposition, so no new dependency. By-prefecture groups stores by the first
  token of their address (the `area_prefecture` split). By-horizon is one row per days-ahead
  offset 1..14 (≈829 stores/offset, so it's robust).
- **The final 39-day forecast** — each model fit on the whole CV panel, rolled forward
  `final_horizon_days` over the untouched holdout: the headline **pred-vs-real**. Plus tree
  **feature importances**.

`adapters/plotting.render_report` turns that into the figure set **and an `index.html`** — a
single scannable report (forecast split by model family, the three breakdowns one metric per
panel, residuals, importances), written to `artifacts/report/` and logged to MLflow. It is the
headline artifact of a CV run: open `artifacts/report/index.html`.

The deployable winner is then re-fit on **all** data (incl. the holdout) and persisted
(`best_model.pkl` + `selection.json`) — the only model that ever sees the holdout.

## Golden Week (business-aware validation)

**Japan's Golden Week** (late Apr–early May) is a cluster of national holidays where restaurant
demand can multiply. The labeled data ends 2017-04-22, so the carved holdout (2017-03-15→04-22)
just misses it — Golden Week sits in the official scoring window, which has no public labels.
It still matters here for two reasons, both handled:

1. **Train must have seen the pattern.** At least one rolling fold includes a **prior**
   holiday period (2016's Golden Week). Otherwise the trees never learn the spike and would
   miss it on any future holiday — they'd predict a normal Monday and miss 3×.
2. **Report it separately.** The comparison report stratifies error **by season** (above) so
   the high-volatility windows don't hide inside an average — a model can look fine overall
   while being terrible exactly on the high-stakes days that drive ordering. (The `golden_week`
   / `is_holiday` / `day_before_holiday` / `day_after_holiday` feature columns from
   `domain/services/features/builder.py` remain available for a finer holiday-window cut if wanted.) Surfacing
   the breakdown is the difference between a metric and an operational insight.
