"""DataSource adapter for the Recruit Restaurant CSVs.

Reads the bundled tables from data/raw/ and normalises them onto the canonical schema in
ports.data_source.RawData (store_id / date / visitors), hiding the AIR/HPG split:

  air_visit_data.csv                       -> visits       (store_id, date, visitors)
  air_reserve.csv + hpg_reserve.csv        -> reservations (store_id, visit_date,
                                              reserve_visitors, lead_time_days)  [row-level]
  air_store_info.csv                       -> stores       (store_id, genre, area, lat, lon)
  date_info.csv                            -> holidays     (date, is_holiday)

HPG reservations are mapped to their AIR store via store_id_relation.csv (inner join: HPG
rows with no AIR counterpart are dropped, since we only forecast AIR stores).

Missing-day policy (docs/01-data.md): closed days are ABSENT rows here and stay that way —
the daily reindex + fill-0 + `is_closed` flag happens downstream in the feature builder
(`domain/services/features/builder.py`), not in this adapter.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...domain.ports.data_source import DataSource, RawData

_REQUIRED = (
    "air_visit_data.csv",
    "air_reserve.csv",
    "hpg_reserve.csv",
    "air_store_info.csv",
    "store_id_relation.csv",
    "date_info.csv",
)
_RESERVE_COLS = ["store_id", "visit_datetime", "reserve_datetime", "reserve_visitors"]


class RecruitCsvSource(DataSource):
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _csv(self, name: str) -> Path:
        """Path to a table, preferring the plain CSV and falling back to a gzipped one.
        hpg_reserve.csv ships gzipped in-repo (it exceeds GitHub's 100MB limit uncompressed);
        pandas.read_csv decompresses .gz transparently by extension."""
        plain = self.root / name
        return plain if plain.exists() else self.root / f"{name}.gz"

    def load(self) -> RawData:
        missing = [f for f in _REQUIRED if not self._csv(f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing Recruit CSV(s) in {self.root}: {missing}. "
                "Run scripts/download_data.sh (Kaggle CLI + accepted competition rules)."
            )
        return RawData(
            visits=self._visits(),
            reservations=self._reservations(),
            stores=self._stores(),
            holidays=self._holidays(),
        )

    def _visits(self) -> pd.DataFrame:
        df = pd.read_csv(self._csv("air_visit_data.csv"), parse_dates=["visit_date"])
        return df.rename(columns={"air_store_id": "store_id", "visit_date": "date"})[
            ["store_id", "date", "visitors"]
        ]

    def _reservations(self) -> pd.DataFrame:
        dt = ["visit_datetime", "reserve_datetime"]
        air = pd.read_csv(self._csv("air_reserve.csv"), parse_dates=dt).rename(
            columns={"air_store_id": "store_id"}
        )
        relation = pd.read_csv(self._csv("store_id_relation.csv"))
        hpg = (
            pd.read_csv(self._csv("hpg_reserve.csv"), parse_dates=dt)
            .merge(relation, on="hpg_store_id", how="inner")  # map HPG id -> AIR id
            .rename(columns={"air_store_id": "store_id"})
        )
        res = pd.concat([air[_RESERVE_COLS], hpg[_RESERVE_COLS]], ignore_index=True)
        res["visit_date"] = res["visit_datetime"].dt.normalize()
        # Lead time: how far ahead the booking was made (a known-in-advance signal).
        res["lead_time_days"] = (
            res["visit_datetime"] - res["reserve_datetime"]
        ).dt.total_seconds() / 86400.0
        return res[["store_id", "visit_date", "reserve_visitors", "lead_time_days"]]

    def _stores(self) -> pd.DataFrame:
        df = pd.read_csv(self._csv("air_store_info.csv"))
        return df.rename(
            columns={
                "air_store_id": "store_id",
                "air_genre_name": "genre",
                "air_area_name": "area",
                "latitude": "lat",
                "longitude": "lon",
            }
        )[["store_id", "genre", "area", "lat", "lon"]]

    def _holidays(self) -> pd.DataFrame:
        df = pd.read_csv(self._csv("date_info.csv"), parse_dates=["calendar_date"])
        return df.rename(columns={"calendar_date": "date", "holiday_flg": "is_holiday"})[
            ["date", "is_holiday"]
        ]
