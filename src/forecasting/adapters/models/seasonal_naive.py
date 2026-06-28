"""Seasonal-naive baseline.

Predicts each (store, date) as the visitors on the same weekday a week earlier — which
is exactly the `lag_7` column `features.py` already computes. The honest anchor every
other model must beat: a model that can't clear seasonal-naive is not worth shipping.

No training. As a baseline it always sees the full feature frame (feature selection does
not apply to it), so `lag_7` is available; where it's NaN (a store's first week) we fall
back to the per-(store, dow) mean, then a global mean fixed at fit time.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from ...domain.ports.model import Model
from ...domain.services.evaluation import clip_nonneg


class SeasonalNaive(Model):
    name = "seasonal_naive"

    def __init__(self) -> None:
        self.global_mean = 0.0

    def fit(self, train: pd.DataFrame) -> None:
        # No parameters to learn — snapshot the global-mean fallback predict() needs.
        self.global_mean = float(train["visitors"].mean())

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        y = features.get("lag_7")
        if y is None:
            y = pd.Series(float("nan"), index=features.index)
        y = y.fillna(features.get("store_dow_mean", self.global_mean))
        y = y.fillna(self.global_mean)
        return features.assign(y_pred=clip_nonneg(y.to_numpy()))

    def save(self, path: Path) -> None:
        Path(path).write_bytes(pickle.dumps(self.global_mean))

    def load(self, path: Path) -> None:
        self.global_mean = pickle.loads(Path(path).read_bytes())
