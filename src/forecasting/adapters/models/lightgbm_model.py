"""LightGBM on engineered features.

Trained in log1p target space (RMSE on log1p ~= RMSLE); predictions are expm1'd and
clipped to >=0 before any metric. Category dtypes (genre/area) are handled natively.
`store_id`/`date` are dropped from the matrix — per-store level is already carried by the
target-encoded `store_mean`/`store_dow_mean` features.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from ...domain.entities import DATE, STORE, TARGET
from ...domain.ports.model import ModelWithFeatureImportance
from ...domain.services.evaluation import clip_nonneg

_DEFAULTS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def _feature_cols(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in (STORE, DATE, TARGET)]


class LightGBMModel(ModelWithFeatureImportance):
    name = "lightgbm"

    def __init__(self, params: dict[str, object] | None = None) -> None:
        self.params = {**_DEFAULTS, **(params or {})}
        self.model = LGBMRegressor(**self.params)
        self.features_: list[str] = []

    def fit(self, train: pd.DataFrame) -> None:
        self.features_ = _feature_cols(train)
        X = train[self.features_]
        y = np.log1p(train[TARGET].to_numpy())
        self.model.fit(X, y)  # category dtype -> native categorical handling

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X = features[self.features_]
        y = clip_nonneg(np.expm1(self.model.predict(X)))
        return features.assign(y_pred=y)

    def feature_importance(self) -> dict[str, float]:
        gain = self.model.booster_.feature_importance(importance_type="gain")
        names = self.model.booster_.feature_name()
        return dict(zip(names, gain.astype(float), strict=True))

    def save(self, path: Path) -> None:
        Path(path).write_bytes(pickle.dumps((self.model, self.features_)))

    def load(self, path: Path) -> None:
        self.model, self.features_ = pickle.loads(Path(path).read_bytes())
