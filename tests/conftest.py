"""Shared synthetic fixtures: a small multi-store panel + its feature matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.domain.ports.data_source import RawData
from forecasting.domain.services.features import FeatureBuilder

STORES = ["air_a", "air_b", "air_c"]


@pytest.fixture
def raw() -> RawData:
    dates = pd.date_range("2016-01-01", "2016-05-31", freq="D")  # 152 days
    rng = np.random.default_rng(0)
    rows = []
    for si, s in enumerate(STORES):
        for d in dates:
            base = 20 + 5 * si + 8 * np.sin(2 * np.pi * d.dayofweek / 7)
            rows.append((s, d, max(0, int(base + rng.normal(0, 2)))))
    visits = pd.DataFrame(rows, columns=["store_id", "date", "visitors"])
    # Drop ~5% of rows to exercise the closed-day reindex.
    visits = (
        visits.sample(frac=0.95, random_state=1)
        .sort_values(["store_id", "date"])
        .reset_index(drop=True)
    )
    reservations = pd.DataFrame(
        {
            "store_id": ["air_a", "air_b"],
            "visit_date": [pd.Timestamp("2016-02-01"), pd.Timestamp("2016-02-02")],
            "reserve_visitors": [5, 3],
            "lead_time_days": [2.0, 1.0],
        }
    )
    stores = pd.DataFrame(
        {
            "store_id": STORES,
            "genre": ["Izakaya", "Cafe", "Bar"],
            "area": ["Tokyo A-ku X", "Tokyo B-ku Y", "Osaka C-ku Z"],
            "lat": [35.6, 35.7, 34.7],
            "lon": [139.6, 139.7, 135.5],
        }
    )
    holidays = pd.DataFrame(
        {"date": [pd.Timestamp("2016-05-03"), pd.Timestamp("2016-05-04")], "is_holiday": [1, 1]}
    )
    return RawData(visits=visits, reservations=reservations, stores=stores, holidays=holidays)


@pytest.fixture
def feat(raw) -> pd.DataFrame:
    return FeatureBuilder(
        raw.visits, reservations=raw.reservations, stores=raw.stores, holidays=raw.holidays
    ).build()
