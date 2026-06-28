"""Port: where the demand panel + side tables come from."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RawData:
    """The tables a forecasting run needs, already loaded but not yet featurised."""

    visits: pd.DataFrame  # store_id, date, visitors (observed days only; absent = closed)
    reservations: pd.DataFrame  # store_id, visit_date, reserve_visitors, lead_time_days (row-level)
    stores: pd.DataFrame  # store_id, genre, area, lat, lon
    holidays: pd.DataFrame  # date, is_holiday


class DataSource(ABC):
    @abstractmethod
    def load(self) -> RawData: ...
