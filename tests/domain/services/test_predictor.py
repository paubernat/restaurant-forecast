"""Recursive multi-step Predictor: feedback into lags, and the native one-shot path."""

from __future__ import annotations

import pandas as pd

from forecasting.domain.entities import DATE, STORE, TARGET
from forecasting.domain.services.predictor import Predictor


class _EchoLag1Plus1:
    """y_pred = lag_1 + 1. With feedback this ramps; without it, lag_1 would be NaN past +1."""

    name = "echo"

    def fit(self, train):  # noqa: D401
        pass

    def predict(self, features):
        return features.assign(y_pred=(features["lag_1"].fillna(0.0) + 1).to_numpy())


class _NativeConst:
    recursive = False
    name = "native"

    def __init__(self):
        self.calls = 0

    def fit(self, train):
        pass

    def predict(self, rows):
        self.calls += 1
        return rows.assign(y_pred=7.0)


class _PrepareSpy:
    """Exposes the optional prepare_window hook; records calls to prove it runs once up front."""

    name = "spy"

    def __init__(self):
        self.prepared = []
        self.predict_dates = []

    def fit(self, train):
        pass

    def prepare_window(self, history, horizon):
        self.prepared.append((history[DATE].max(), horizon))

    def predict(self, features):
        self.predict_dates.append(features[DATE].iloc[0])
        return features.assign(y_pred=1.0)


def test_prepare_window_called_once_before_the_loop():
    dates = pd.date_range("2016-01-01", periods=40, freq="D")
    history = pd.DataFrame({STORE: "s", DATE: dates, TARGET: 10.0})
    spy = _PrepareSpy()
    Predictor(spy, reference=history).infer(history, horizon=3)
    # Hook runs exactly once, on the full history, before any per-step predict.
    assert spy.prepared == [(dates.max(), 3)]
    assert len(spy.predict_dates) == 3  # then one predict per horizon step


def test_recursive_feeds_predictions_into_lags():
    dates = pd.date_range("2016-01-01", periods=40, freq="D")
    history = pd.DataFrame({STORE: "s", DATE: dates, TARGET: 10.0})
    pred = Predictor(_EchoLag1Plus1(), reference=history).infer(history, horizon=3)
    ys = pred.sort_values(DATE)["y_pred"].tolist()
    # +1 day: lag_1 = last actual (10) -> 11; then each step feeds back -> 12, 13.
    assert ys == [11.0, 12.0, 13.0]


def test_native_model_predicts_block_once():
    dates = pd.date_range("2016-01-01", periods=10, freq="D")
    history = pd.concat(
        [
            pd.DataFrame({STORE: "a", DATE: dates, TARGET: 5.0}),
            pd.DataFrame({STORE: "b", DATE: dates, TARGET: 8.0}),
        ]
    )
    model = _NativeConst()
    pred = Predictor(model, reference=history).infer(history, horizon=4)
    assert model.calls == 1  # one batched call, no per-step loop
    assert len(pred) == 2 * 4
    assert (pred["y_pred"] == 7.0).all()
    expected = {dates.max() + pd.Timedelta(days=k) for k in range(1, 5)}
    assert set(pred[DATE]) == expected
