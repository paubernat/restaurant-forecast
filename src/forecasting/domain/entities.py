"""Domain value objects shared across the core.

Deliberately thin: a multi-series demand panel is just a tidy DataFrame
(`store_id`, `date`, `visitors`, + features); these dataclasses carry the
typed results that flow between use cases and adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Canonical column names used across the domain (Recruit -> generic).
STORE = "store_id"
DATE = "date"
TARGET = "visitors"


@dataclass(frozen=True)
class Forecast:
    """Per-(store, date) predictions for one model over a horizon."""

    model_name: str
    frame: pd.DataFrame  # columns: store_id, date, y_pred (and optional quantile cols)


@dataclass(frozen=True)
class EvalResult:
    """Metrics for one model on one evaluation window.

    `metrics` is the overall suite over the model's last-year CV predictions; `by_segment`
    holds the per-season split (`season(date)`); `by_region` the per-prefecture split; and
    `by_horizon` holds error as a function of days-ahead (one row per offset 1..horizon). Each
    split carries the full metric suite (rmsle / mae / weighted_mae).
    """

    model_name: str
    metrics: dict[str, float]
    by_segment: dict[str, dict[str, float]] = field(default_factory=dict)
    by_region: dict[str, dict[str, float]] = field(default_factory=dict)
    by_horizon: pd.DataFrame | None = None


@dataclass(frozen=True)
class ComparisonReport:
    """The step-3.5 cross-model report (pure data; plots/MLflow live in the adapters).

    `results` maps model name -> its last-year CV `EvalResult`. `holdout_preds` maps model
    name -> the final `final_horizon`-day pred-vs-real frame (store_id, date, y_pred, y_true)
    — the headline forecast. `importances` maps tree model name -> {feature: gain}.
    """

    results: dict[str, EvalResult]
    holdout_preds: dict[str, pd.DataFrame]
    importances: dict[str, dict[str, float]]
    horizon: int
    final_horizon: int
