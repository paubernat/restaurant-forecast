"""Forecasting metrics with log-domain safety.

Three metrics, each a distinct job (see docs/04-evaluation.md):
  - RMSLE   — the ranking metric. RMSE in log space, so it scores *relative* error and is
              fair across stores of very different size; this is what CV selects on.
  - MAE     — the human-readable companion: average miss in visitors.
  - weighted_mae — the business view: under-prediction (stockout) costs more than over (waste).

All metrics route predictions through `clip_nonneg` first: trees and TimesFM can emit small
negatives, and RMSLE on a negative (or the log of 0) is a hard math error. We use log1p
throughout.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

Array = npt.NDArray[np.float64]


def clip_nonneg(y: Array) -> Array:
    """Demand cannot be negative; floor predictions at 0 before any metric/inverse."""
    return np.maximum(0.0, np.asarray(y, dtype=np.float64))


def mae(y_true: Array, y_pred: Array) -> float:
    y_pred = clip_nonneg(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmsle(y_true: Array, y_pred: Array) -> float:
    """Root Mean Squared Logarithmic Error using log1p (safe at 0)."""
    y_pred = clip_nonneg(y_pred)
    y_true = clip_nonneg(y_true)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def weighted_mae(y_true: Array, y_pred: Array, *, under_weight: float, over_weight: float) -> float:
    """Asymmetric MAE: under- and over-prediction cost differently.

    For supplier ordering, under-predicting demand (`y_pred < y_true`) is a stockout
    (lost sales) while over-predicting is waste — usually not equally bad. Weight the
    absolute residual by `under_weight` on the stockout side, `over_weight` otherwise.
    """
    y_pred = clip_nonneg(y_pred)
    y_true = clip_nonneg(y_true)
    err = np.abs(y_true - y_pred)
    w = np.where(y_pred < y_true, under_weight, over_weight)
    return float(np.mean(w * err))


def get_metrics(
    y_true: Array,
    y_pred: Array,
    *,
    under_weight: float = 1.0,
    over_weight: float = 1.0,
) -> dict[str, float]:
    """The full metric suite for one set of predictions: RMSLE (ranking), MAE (readable),
    weighted_mae (business cost)."""
    return {
        "rmsle": rmsle(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "weighted_mae": weighted_mae(
            y_true, y_pred, under_weight=under_weight, over_weight=over_weight
        ),
    }


_SEASONS = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}  # N-hemisphere; used for the report's seasonal performance breakdown (not a decomposition)


def season(date) -> str:
    """Map a date to its meteorological season (N-hemisphere) by month."""
    return _SEASONS[pd.Timestamp(date).month]


def selection_metric(name: str, *, under_weight: float = 1.0, over_weight: float = 1.0):
    """Resolve a metric name to the callable the CV ranks by (lower is better)."""
    if name == "weighted_mae":
        return lambda yt, yp: weighted_mae(
            yt, yp, under_weight=under_weight, over_weight=over_weight
        )
    funcs = {"rmsle": rmsle, "mae": mae}
    if name not in funcs:
        raise ValueError(f"Unknown selection metric: {name!r}")
    return funcs[name]
