"""Model + feature selection pipeline (pure domain service).

  Step 1: rolling-origin CV of every model on the FULL feature set; pick the best by the
          chosen metric.
  Step 2: feature selection on the step-1 winner (cumulative tree importance, generous).
  Step 3: re-CV every model on the SELECTED features, sweeping a small per-model grid; each
          model's grid includes its step-1 config, so that exact config is always re-tried on
          the reduced feature set (the score can still shift — the feature set changed).
  Step 4: build the cross-model `ComparisonReport` (reusing step-3's retained fold
          predictions + a final holdout forecast per model), then fit the winner on ALL data.

Scoring is **honest n-day-ahead**: each fold trains on `date <= train_end`, then the
`Predictor` rolls the forecast forward `horizon` days (recursive multi-step), and we score
against the observed valid days. The model registry (factories + grids) and the tracker are
injected, so this service imports no adapter and no model library — ports only.

Leakage discipline: the final holdout (the last `final_horizon_days`) is carved off first and
never touched by CV; per fold the target-based store aggregates are rebuilt with `reference` =
that fold's train slice; lags in the horizon come from the model's own predictions, never
intra-window actuals; the window TimesFM forecast uses only history <= cutoff. Only the
returned deployable winner is fit on all data.

`ModelSelector` orchestrates; `ModelSpec` is the registry entry the CLI builds.
"""

from .main import ModelSelector
from .spec import ModelSpec

__all__ = ["ModelSelector", "ModelSpec"]
