"""Hybrid with the window-TimesFM design, using a fake forecaster (no TimesFM/torch needed).

The fake returns a per-step ramp so the tests can prove (a) the head consumes the window
training signal and (b) `augment` indexes the precomputed window by days-ahead offset. The
real foundation model is exercised in the CV run, not here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.adapters.models.timesfm.features import TimesFMFeatureGenerator
from forecasting.adapters.models.timesfm.hybrid import TimesFMHybrid
from forecasting.domain.entities import DATE, STORE
from forecasting.domain.services.features import FeatureBuilder
from forecasting.domain.services.predictor import Predictor


class _RampForecaster:
    """point[i, k] = last_value_i + (k + 1); quantiles broadcast it. Distinct per step k."""

    def forecast(self, series, horizon):
        last = np.array([float(np.asarray(s)[-1]) for s in series])  # (n,)
        steps = np.arange(1, horizon + 1)  # (H,)
        point = last[:, None] + steps[None, :]  # (n, H)
        quant = np.repeat(point[:, :, None], 10, axis=2)  # (n, H, 10)
        return point, quant


def test_hybrid_fits_on_window_signal_and_predicts_recursively(raw):
    gen = TimesFMFeatureGenerator(_RampForecaster(), max_train_cutoffs=20)
    model = TimesFMHybrid(gen, {"n_estimators": 10}, horizon=5)
    feat = FeatureBuilder(
        raw.visits, reservations=raw.reservations, stores=raw.stores, holidays=raw.holidays
    ).build()
    model.fit(feat)
    assert any(c.startswith("tfm_") for c in model.features_)  # window signal reached the head

    pred = Predictor(
        model,
        reservations=raw.reservations,
        stores=raw.stores,
        holidays=raw.holidays,
        reference=raw.visits,
    ).infer(raw.visits, horizon=3)

    assert len(pred) == raw.visits[STORE].nunique() * 3
    assert np.isfinite(pred["y_pred"]).all()
    assert (pred["y_pred"] >= 0).all()


def test_augment_indexes_the_window_by_days_ahead_offset(raw):
    gen = TimesFMFeatureGenerator(_RampForecaster(), max_train_cutoffs=20)
    model = TimesFMHybrid(gen, {"n_estimators": 10}, horizon=5)
    history = raw.visits
    cutoff = pd.to_datetime(history[DATE]).max()
    stores = history[STORE].unique()

    model.prepare_window(history, horizon=5)  # one forecast for the whole window
    f1 = model.augment(pd.DataFrame({STORE: stores, DATE: cutoff + pd.Timedelta(days=1)}), history)
    f3 = model.augment(pd.DataFrame({STORE: stores, DATE: cutoff + pd.Timedelta(days=3)}), history)

    # ramp: step k point = last + k, so offset 3 minus offset 1 = 2 for every store (no refetch).
    assert np.allclose(f3["tfm_point"].to_numpy() - f1["tfm_point"].to_numpy(), 2.0)
