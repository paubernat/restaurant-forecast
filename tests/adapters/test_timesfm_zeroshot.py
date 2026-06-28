"""Zero-shot adapter logic, with a fake forecaster (no TimesFM/torch needed).

Asserts the offset mapping (cutoff -> horizon step) and non-negative output. The real
TimesFM loader is exercised manually / in the CV run, not here — it's a heavy git+torch
dependency and a network download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.adapters.models.timesfm.zeroshot import TimesFMZeroShot
from forecasting.domain.entities import DATE, STORE, TARGET


class _RampForecaster:
    """Returns [100, 101, 102, ...] per series, so y_pred reveals the horizon offset used."""

    def forecast(self, series, horizon):
        point = np.array([[100.0 + h for h in range(horizon)] for _ in series])
        return point, None


def _panel(store, start, n, value):
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({STORE: store, DATE: dates, TARGET: float(value)})


def test_zeroshot_maps_offset_from_cutoff_and_clips():
    train = pd.concat([_panel("a", "2016-01-01", 30, 8), _panel("b", "2016-01-01", 30, 3)])
    cutoff = train[DATE].max()  # 2016-01-30
    first = cutoff + pd.Timedelta(days=1)
    valid = pd.concat([_panel("a", first, 5, 0), _panel("b", first, 5, 0)])

    model = TimesFMZeroShot(_RampForecaster())
    model.fit(train)
    out = model.predict(valid)

    # First valid day -> horizon step 0 -> 100; second -> 101; ...
    a = out[out[STORE] == "a"].sort_values(DATE)["y_pred"].to_numpy()
    assert list(a) == [100.0, 101.0, 102.0, 103.0, 104.0]
    assert (out["y_pred"] >= 0).all()


def test_zeroshot_unknown_store_falls_back_to_global_mean():
    train = _panel("a", "2016-01-01", 14, 10)
    model = TimesFMZeroShot(_RampForecaster())
    model.fit(train)
    valid = _panel("ghost", train[DATE].max() + pd.Timedelta(days=1), 3, 0)
    out = model.predict(valid)
    assert (out["y_pred"] == 10.0).all()  # global mean of train
