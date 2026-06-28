"""XGBoost on engineered features.

Same log1p-space training as LightGBM (expm1 + clip on predict). Categoricals handled via
`enable_categorical` + the `hist` tree method. Doubles as the base learner of the hybrid
(timesfm_hybrid), so `xgboost` vs `timesfm_hybrid` is a clean ablation isolating TimesFM's
marginal value.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ...domain.entities import DATE, STORE, TARGET
from ...domain.ports.model import ModelWithFeatureImportance
from ...domain.services.evaluation import clip_nonneg

_DEFAULTS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "enable_categorical": True,
}


def _feature_cols(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in (STORE, DATE, TARGET)]


class XGBoostModel(ModelWithFeatureImportance):
    name = "xgboost"

    def __init__(self, params: dict[str, object] | None = None) -> None:
        self.params = {**_DEFAULTS, **(params or {})}
        self.model = XGBRegressor(**self.params)
        self.features_: list[str] = []

    def fit(self, train: pd.DataFrame) -> None:
        self.features_ = _feature_cols(train)
        X = train[self.features_]
        y = np.log1p(train[TARGET].to_numpy())
        self.model.fit(X, y)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X = features[self.features_]
        y = clip_nonneg(np.expm1(self.model.predict(X)))
        return features.assign(y_pred=y)

    def feature_importance(self) -> dict[str, float]:
        scores = self.model.get_booster().get_score(importance_type="gain")
        return {f: float(scores.get(f, 0.0)) for f in self.features_}

    def save(self, path: Path) -> None:
        Path(path).write_bytes(pickle.dumps((self.model, self.features_)))

    def load(self, path: Path) -> None:
        self.model, self.features_ = pickle.loads(Path(path).read_bytes())
