"""Evaluation domain service — forecasting metrics and the helpers that apply them.

`get_metrics(y_true, y_pred)` is the entry point: the full suite (RMSLE, MAE, weighted_mae)
for one set of predictions. `selection_metric(name)` resolves the config's metric name to the
single callable CV ranks by; `season(date)` buckets results for the per-season report.
"""

from .metrics import (
    clip_nonneg,
    get_metrics,
    mae,
    rmsle,
    season,
    selection_metric,
    weighted_mae,
)

__all__ = [
    "clip_nonneg",
    "get_metrics",
    "mae",
    "rmsle",
    "season",
    "selection_metric",
    "weighted_mae",
]
