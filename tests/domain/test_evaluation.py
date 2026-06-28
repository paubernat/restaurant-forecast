"""Metrics + log-domain safety (the one piece of real logic in the scaffold)."""

import numpy as np
import pandas as pd

from forecasting.domain.services import evaluation as ev


def test_perfect_prediction_is_zero_error():
    y = np.array([0.0, 3.0, 10.0, 250.0])
    m = ev.get_metrics(y, y.copy())
    assert m["rmsle"] == 0.0
    assert m["mae"] == 0.0
    assert m["weighted_mae"] == 0.0


def test_rmsle_safe_at_zero_and_negative_predictions():
    # log(0) and negative preds would crash a naive RMSLE; clip_nonneg + log1p guard it.
    y_true = np.array([0.0, 5.0])
    y_pred = np.array([-2.0, 4.0])  # negative prediction must be floored to 0
    val = ev.rmsle(y_true, y_pred)
    assert np.isfinite(val) and val >= 0.0


def test_clip_floors_negatives():
    assert np.array_equal(ev.clip_nonneg(np.array([-1.0, 2.0, -0.5])), np.array([0.0, 2.0, 0.0]))


def test_weighted_mae_penalises_underprediction_more():
    # Same |error| of 2, but under-prediction (stockout) should cost more than over (waste).
    y_true = np.array([10.0])
    under = ev.weighted_mae(y_true, np.array([8.0]), under_weight=2.0, over_weight=1.0)
    over = ev.weighted_mae(y_true, np.array([12.0]), under_weight=2.0, over_weight=1.0)
    assert under == 4.0 and over == 2.0
    assert under > over


def test_season_maps_month_to_n_hemisphere_season():
    assert ev.season(pd.Timestamp("2016-01-15")) == "winter"
    assert ev.season("2016-04-10") == "spring"
    assert ev.season("2016-07-01") == "summer"
    assert ev.season("2016-10-31") == "autumn"
    assert ev.season("2016-12-25") == "winter"


def test_selection_metric_resolves_names():
    yt, yp = np.array([10.0]), np.array([8.0])
    assert ev.selection_metric("rmsle")(yt, yp) == ev.rmsle(yt, yp)
    wm = ev.selection_metric("weighted_mae", under_weight=2.0, over_weight=1.0)
    assert wm(yt, yp) == ev.weighted_mae(yt, yp, under_weight=2.0, over_weight=1.0)
