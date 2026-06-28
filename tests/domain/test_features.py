"""Feature engineering tests — the load-bearing one is no-leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.domain.entities import DATE, STORE, TARGET
from forecasting.domain.services import features


def _panel(store: str, start: str, n: int, base: int = 10) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({STORE: store, DATE: dates, TARGET: np.arange(base, base + n)})


def test_lags_and_rolling_have_no_leakage():
    panel = _panel("air_a", "2016-01-01", 20)  # visitors = 10,11,...,29
    feat = features.FeatureBuilder(panel).build().sort_values(DATE).reset_index(drop=True)

    # lag_1[t] is exactly visitors[t-1]; lag_7[t] is visitors[t-7].
    assert feat["lag_1"].iloc[5] == panel[TARGET].iloc[4]
    assert feat["lag_7"].iloc[10] == panel[TARGET].iloc[3]

    # roll_mean_7 at row 7 uses visitors[0:7] only (shift(1) then window) — never its own row.
    assert feat["roll_mean_7"].iloc[7] == panel[TARGET].iloc[0:7].mean()
    # First window-worth of rows are NaN (no past to look at) — proof we don't peek ahead.
    assert feat["roll_mean_7"].iloc[:7].isna().all()


def test_closed_days_reindexed_to_zero_with_flag():
    # Drop 2016-01-05 -> a closed day (absent row), span stays 01-01..01-10.
    panel = _panel("air_a", "2016-01-01", 10)
    panel = panel[panel[DATE] != pd.Timestamp("2016-01-05")]

    feat = features.FeatureBuilder(panel).build()
    gap = feat[feat[DATE] == pd.Timestamp("2016-01-05")].iloc[0]
    assert gap[TARGET] == 0.0 and gap["is_closed"] == 1
    assert (feat[feat[DATE] == pd.Timestamp("2016-01-04")]["is_closed"] == 0).all()
    assert len(feat) == 10  # gap filled back in


def test_calendar_features():
    # 2016-01-02 is a Saturday
    feat = features.FeatureBuilder(_panel("air_a", "2016-01-01", 14)).build()
    sat = feat[feat[DATE] == pd.Timestamp("2016-01-02")].iloc[0]
    assert sat["is_weekend"] == 1 and sat["dow"] == 5
    assert feat["doy_sin"].between(-1, 1).all() and feat["doy_cos"].between(-1, 1).all()


def test_store_aggregates_use_reference_not_future():
    # store_mean must come from `reference` (train), ignoring an inflated val row.
    train = _panel("air_a", "2016-01-01", 10, base=10)  # mean 14.5
    full = pd.concat([train, _panel("air_a", "2016-01-11", 1, base=1000)], ignore_index=True)

    feat = features.FeatureBuilder(full, reference=train).build()
    assert np.isclose(feat["store_mean"].iloc[0], train[TARGET].mean())
    # The 1000-visitor day did NOT leak into the aggregate.
    assert feat["store_mean"].iloc[0] < 20


def test_reservations_aggregated_and_defaulted():
    panel = _panel("air_a", "2016-01-01", 5)
    reservations = pd.DataFrame(
        {
            STORE: "air_a",
            "visit_date": ["2016-01-02", "2016-01-02", "2016-01-03"],
            "reserve_visitors": [3, 4, 5],
        }
    )
    feat = features.FeatureBuilder(panel, reservations=reservations).build()
    d2 = feat[feat[DATE] == pd.Timestamp("2016-01-02")].iloc[0]
    assert d2["reserve_visitors"] == 7 and d2["reserve_count"] == 2
    # Days with no reservation default to 0, not NaN.
    assert feat[feat[DATE] == pd.Timestamp("2016-01-01")]["reserve_visitors"].iloc[0] == 0.0
