"""TimesFM 2.5 zero-shot baseline.

Pure foundation model: no training, no covariates. `fit` snapshots each store's
visitor history reindexed to a **daily grid ending at the train cutoff** (closed days
absent in the panel -> filled with 0, since a closed day means no visitors). `predict`
runs one batched TimesFM forecast over all stores and reads off the value at each
(store, date) by calendar-day offset from the cutoff. The honest "frontier model out of
the box" reference for the ablation.

The TimesFM forecaster is injected (a remote-endpoint adapter) so the CV needs no torch/timesfm
locally and tests can use a fake. See adapters/models/timesfm/remote.py.

ponytail: irregular gaps are 0-filled rather than interpolated — fine for a zero-shot
baseline; the hybrid (Phase 6) is where the signal gets engineered properly.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from ....domain.entities import DATE, STORE, TARGET
from ....domain.ports.model import Model
from ....domain.services.evaluation import clip_nonneg


class TimesFMZeroShot(Model):
    name = "timesfm_zeroshot"
    recursive = False  # native multi-step: forecasts the whole horizon in one call

    def __init__(self, forecaster, params: dict[str, object] | None = None) -> None:
        self.forecaster = forecaster
        self.params = params or {}
        self._series: dict[object, object] = {}
        self._cutoff: pd.Timestamp | None = None
        self._global_mean = 0.0

    def fit(self, train: pd.DataFrame) -> None:
        # No parameters to learn — snapshot each store's history; predict() forecasts from it.
        self._cutoff = train[DATE].max()
        self._global_mean = float(train[TARGET].mean())
        self._series = {}
        for store, g in train.groupby(STORE):
            s = g.set_index(DATE)[TARGET].sort_index()
            idx = pd.date_range(s.index.min(), self._cutoff, freq="D")
            self._series[store] = s.reindex(idx, fill_value=0.0).to_numpy()

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        horizon = int((features[DATE].max() - self._cutoff).days)
        stores = list(self._series)
        point, _ = self.forecaster.forecast([self._series[s] for s in stores], horizon)
        fc = {s: point[i] for i, s in enumerate(stores)}

        def lookup(row) -> float:
            arr = fc.get(row[STORE])
            off = int((row[DATE] - self._cutoff).days) - 1  # 1-based horizon step
            if arr is None or off < 0 or off >= len(arr):
                return self._global_mean
            return float(arr[off])

        y = features.apply(lookup, axis=1).to_numpy()
        return features.assign(y_pred=clip_nonneg(y))

    def save(self, path: Path) -> None:
        Path(path).write_bytes(pickle.dumps((self._series, self._cutoff, self._global_mean)))

    def load(self, path: Path) -> None:
        self._series, self._cutoff, self._global_mean = pickle.loads(Path(path).read_bytes())
