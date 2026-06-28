"""RecruitCsvSource: normalises the AIR/HPG CSVs onto the canonical schema."""

from __future__ import annotations

import pandas as pd
import pytest

from forecasting.adapters.data.recruit_csv import RecruitCsvSource
from forecasting.domain.services import features


def _write_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "air_store_id": ["air_a", "air_a"],
            "visit_date": ["2016-01-01", "2016-01-02"],
            "visitors": [10, 12],
        }
    ).to_csv(root / "air_visit_data.csv", index=False)
    pd.DataFrame(
        {
            "air_store_id": ["air_a"],
            "visit_datetime": ["2016-01-02 19:00:00"],
            "reserve_datetime": ["2016-01-01 19:00:00"],  # 1 day lead
            "reserve_visitors": [4],
        }
    ).to_csv(root / "air_reserve.csv", index=False)
    pd.DataFrame(
        {
            "hpg_store_id": ["hpg_x", "hpg_unmapped"],
            "visit_datetime": ["2016-01-02 18:00:00", "2016-01-02 18:00:00"],
            "reserve_datetime": ["2016-01-02 12:00:00", "2016-01-02 12:00:00"],
            "reserve_visitors": [3, 99],
        }
    ).to_csv(root / "hpg_reserve.csv", index=False)
    pd.DataFrame(
        {
            "air_store_id": ["air_a"],
            "air_genre_name": ["Izakaya"],
            "air_area_name": ["Tōkyō-to Setagaya-ku Taishidō"],
            "latitude": [35.6],
            "longitude": [139.6],
        }
    ).to_csv(root / "air_store_info.csv", index=False)
    pd.DataFrame({"air_store_id": ["air_a"], "hpg_store_id": ["hpg_x"]}).to_csv(
        root / "store_id_relation.csv", index=False
    )
    pd.DataFrame(
        {
            "calendar_date": ["2016-01-01", "2016-01-02"],
            "day_of_week": ["Friday", "Saturday"],
            "holiday_flg": [1, 1],
        }
    ).to_csv(root / "date_info.csv", index=False)


def test_load_normalises_to_canonical_schema(tmp_path):
    _write_fixture(tmp_path / "raw")
    raw = RecruitCsvSource(tmp_path / "raw").load()

    assert list(raw.visits.columns) == ["store_id", "date", "visitors"]
    assert pd.api.types.is_datetime64_any_dtype(raw.visits["date"])
    assert list(raw.stores.columns) == ["store_id", "genre", "area", "lat", "lon"]
    assert list(raw.holidays.columns) == ["date", "is_holiday"]


def test_hpg_mapped_to_air_and_lead_time_computed(tmp_path):
    _write_fixture(tmp_path / "raw")
    res = RecruitCsvSource(tmp_path / "raw").load().reservations

    # HPG reservation surfaces under its AIR id; the unmapped HPG store is dropped.
    assert set(res["store_id"]) == {"air_a"}
    assert (res["reserve_visitors"] == 99).sum() == 0  # hpg_unmapped gone
    air_row = res[res["reserve_visitors"] == 4].iloc[0]
    assert air_row["lead_time_days"] == pytest.approx(1.0)


def test_load_feeds_build_features(tmp_path):
    _write_fixture(tmp_path / "raw")
    raw = RecruitCsvSource(tmp_path / "raw").load()
    feat = features.FeatureBuilder(
        raw.visits, reservations=raw.reservations, stores=raw.stores, holidays=raw.holidays
    ).build()
    # The port output flows straight into the domain feature pipeline.
    for col in ("is_closed", "lag_1", "roll_mean_7", "reserve_visitors", "store_dow_mean", "genre"):
        assert col in feat.columns
    d2 = feat[feat["date"] == pd.Timestamp("2016-01-02")].iloc[0]
    assert d2["reserve_visitors"] == 7  # air 4 + hpg 3 on the same day


def test_missing_files_raise(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="Missing Recruit CSV"):
        RecruitCsvSource(tmp_path / "empty").load()
