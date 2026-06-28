"""Temporal splits: time order respected, no train/valid leakage, Golden Week in train."""

from __future__ import annotations

import pandas as pd

from forecasting.domain import validation

# Mirrors the real Recruit AIR calendar (Jan 2016 -> 22 Apr 2017, daily).
DATES = pd.Series(pd.date_range("2016-01-01", "2017-04-22", freq="D"))
HORIZON = 39


def test_final_holdout_is_last_horizon_days():
    h = validation.final_holdout(DATES, horizon_days=HORIZON)
    assert h.valid_end == pd.Timestamp("2017-04-22")
    assert (h.valid_end - h.valid_start).days == HORIZON - 1
    assert h.train_end == h.valid_start - pd.Timedelta(days=1)


def test_rolling_origin_ordered_nonoverlapping_expanding():
    folds = validation.rolling_origin_splits(DATES, n_folds=5, horizon_days=HORIZON)
    assert len(folds) == 5

    # Earliest-first ordering; training window expands as the cutoff advances.
    train_ends = [f.train_end for f in folds]
    assert train_ends == sorted(train_ends)

    for f in folds:
        assert (f.valid_end - f.valid_start).days == HORIZON - 1
        # The load-bearing check: validation always strictly after training.
        assert f.train_end < f.valid_start

    # Consecutive validation windows don't overlap (step == horizon).
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert earlier.valid_end < later.valid_start


def test_stride_days_overlaps_and_steps_origins_by_stride():
    folds = validation.rolling_origin_splits(DATES, n_folds=5, horizon_days=HORIZON, stride_days=5)
    # Origins (valid_end) step by exactly `stride` days, not by the horizon.
    valid_ends = [f.valid_end for f in folds]
    steps = {(later - earlier).days for earlier, later in zip(valid_ends, valid_ends[1:], strict=False)}
    assert steps == {5}
    # stride < horizon => consecutive validation windows OVERLAP (origin rotates the weekday).
    assert any(earlier.valid_end >= later.valid_start
               for earlier, later in zip(folds, folds[1:], strict=False))


def test_some_fold_trains_through_2016_golden_week():
    # Otherwise the trees never see the holiday spike the holdout is full of.
    folds = validation.rolling_origin_splits(DATES, n_folds=5, horizon_days=HORIZON)
    assert any(f.train_end > pd.Timestamp("2016-05-05") for f in folds)
