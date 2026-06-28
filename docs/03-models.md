# 03 — Models

Five models, all behind the same `Model` port (`fit` / `predict` / `save` / `load`), so the
use cases treat them uniformly and the comparison is apples-to-apples.

| Model | Type | Trained? | Uses covariates? | Role |
|---|---|---|---|---|
| `seasonal_naive` | heuristic | no | no | **Baseline** — the honest anchor (implemented) |
| `lightgbm` | gradient-boosted trees | yes | yes (engineered) | Strong global multi-series model (implemented) |
| `xgboost` | gradient-boosted trees | yes | yes (engineered) | Tree comparison + **hybrid base learner** (implemented) |
| `timesfm_zeroshot` | foundation model | no | no | Reference: frontier model, out of the box (implemented) |
| `timesfm_hybrid` | TimesFM signal → XGBoost | yes | yes (engineered) | **Centerpiece** — best of both (implemented) |

## Rationale

- **seasonal-naive** predicts each (store, date) as the same weekday's most recent value.
  Cheap, strong on weekly data, and a model that can't beat it isn't worth shipping.
- **LightGBM & XGBoost** are the workhorses for tabular multi-series demand: fast, handle
  categoricals, capture non-linear covariate interactions (holiday × weekend × reservations).
  We keep both — XGBoost is the hybrid's base, so `xgboost` vs `timesfm_hybrid` is a clean
  ablation. They tell a similar story; that's intentional (the ablation needs a matched base).
- **TimesFM zero-shot** is nearly free (it reuses the same remote forecaster the hybrid calls)
  and answers "what does a pretrained time-series foundation model give with zero training?".
- **TimesFM hybrid** is the interesting one — full design in
  [`05-timesfm-hybrid.md`](./05-timesfm-hybrid.md).

## Why these two "distintos" satisfy the brief

The brief asks for ≥2 distinct ML models (e.g. RF vs XGBoost/LightGBM/NN). We exceed it: two
gradient-boosting trees, a foundation model, and a hybrid — spanning heuristic, tabular ML,
and deep pretrained models. RandomForest was dropped in favour of the TimesFM story (RF and
LightGBM would tell nearly the same tabular tale); it can be added trivially if wanted.

## Model selection (`find_best_model`)

All five models are exercised by a single entrypoint,
`python -m forecasting find-best-model --metric rmsle --horizon 14`
(thin use case `application/find_best_model.py` → domain service
`domain/services/model_selection.ModelSelector`), which runs a **4-step** selection pipeline
(rank → feature-select → grid-search → report) — each step logged to MLflow. The CV is
**honest n-day-ahead**: each fold
trains on its history, then `domain/services/Predictor` rolls the forecast forward `--horizon`
days recursively (lags come from the model's own predictions, never the intra-window actuals;
the hybrid forecasts its TimesFM window once at the cutoff over history ≤ cutoff) and scores
against the observed valid days.

1. **CV, all features, all models.** Rolling-origin CV (`domain/validation.py`) of every model
   on the full v1 feature set; rank by a **selectable metric** (`--metric`, default `rmsle`,
   e.g. `weighted_mae`). Pick the best.
2. **Feature selection on the winner.** Refit the best tree model on the CV-train and keep the
   features making up the top **95%** of cumulative gain (config `feature_select_threshold`) —
   generous, drops only the dead tail. The selected list is logged as an artifact.
3. **CV, selected features, all models + a small grid.** Re-CV on the reduced feature set,
   sweeping a small hyperparameter grid per tree model. Each model's grid is
   `{simple_grid} ∪ {its step-1 config}`, so the winner's step-1 hyperparameters are **always
   re-tried on the reduced feature set** (no hyperparameter regression). The score itself can
   still move vs step 1, since the feature set changed — that delta *is* the cost/benefit of
   pruning. Pick the overall `(model, params)`.

Leakage stays honest: the final holdout (`final_holdout`) is carved off first and never
touched, and per fold the target-based store aggregates are rebuilt with `reference` = that
fold's training slice. `seasonal_naive` ignores the feature matrix, so it rides along as a
fixed baseline but sits out feature selection and the grid.

**4. Comparison report + register the winner.** From the retained step-3 fold predictions
(no extra CV pass) plus a final 39-day pred-vs-real forecast per model,
`domain/services/model_selection/step4_report.build_report` builds a cross-model
`ComparisonReport` — overall/by-season/by-horizon metrics, the headline forecast, and tree
feature importances (plots in `adapters/plotting`, logged to MLflow as `report-<model>` + a
`report-summary` run; see [`04-evaluation.md`](./04-evaluation.md)). The winning
`(model, params)` is then re-fit on
**all** data and persisted as `artifacts/best_model.pkl` + `artifacts/selection.json` (what a
future train-job reads); `find_best_model.run` returns the trained winner and the report.

## The headline result

The slide that matters is the **ablation**: `xgboost` (engineered features only) vs
`timesfm_hybrid` (same features + TimesFM signal). The RMSLE delta *is* TimesFM's marginal
value — a far more informative result than a raw leaderboard number. The report's overall and
seasonal bars put the two side by side.
