"""Temporal validation: time-respecting splits for honest evaluation.

Rolling-origin (expanding-window) CV + a final forward holdout mirroring the
competition's last ~39 days (Golden Week). Two non-negotiables:

  1. No shuffling: splits are by date only; never random K-fold on a time series.
  2. The holdout is carved off FIRST; rolling_origin_splits runs only on the
     earlier history (caller filters dates to <= holdout.train_end), so CV never
     touches the holdout.

A split is a triple of date boundaries, not row indices: leakage-safe because the
caller filters its DataFrame by date (train: date <= train_end; valid:
valid_start <= date <= valid_end).

The "normal days vs Golden Week" stratified error report is NOT here: it reuses
the feature columns already produced in features.py (`golden_week`, `is_holiday`,
…), computed at eval time. See docs/04-evaluation.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def _window(valid_end: pd.Timestamp, horizon_days: int) -> TemporalSplit:
    valid_start = valid_end - pd.Timedelta(days=horizon_days - 1)
    return TemporalSplit(valid_start - pd.Timedelta(days=1), valid_start, valid_end)


def final_holdout(dates: pd.Series, *, horizon_days: int) -> TemporalSplit:
    """Carve off the last `horizon_days` as the untouched final-model holdout."""
    return _window(pd.Timestamp(dates.max()), horizon_days)


def rolling_origin_splits(
    dates: pd.Series, *, n_folds: int, horizon_days: int, stride_days: int | None = None
) -> list[TemporalSplit]:
    """Expanding-window splits ending at successively later cutoffs (earliest-first).

    Fold i (walking back from the max date) validates on a `horizon_days` window; the train
    period expands as the cutoff advances. Consecutive cutoffs are `stride_days` apart.

    `stride_days` defaults to `horizon_days` → windows tile without overlap (each day scored
    once). A smaller stride overlaps the windows; choosing one **coprime with 7** (the app
    default is 9) rotates the forecast origin through every weekday, so the by-horizon error
    stops being locked to a single day-of-week (it costs correlated folds + more compute — see
    docs/04-evaluation.md).
    """
    stride = stride_days or horizon_days
    last = pd.Timestamp(dates.max())
    splits = [
        _window(last - pd.Timedelta(days=i * stride), horizon_days) for i in range(n_folds)
    ]
    return splits[::-1]  # earliest-first
