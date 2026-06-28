"""Hybrid: TimesFM window signal + engineered features -> XGBoost. The centerpiece.

TimesFM forecasts the **whole horizon once at the cutoff** (a block forecast beats one step
at a time) into a per-(store, step) signal (point + 10 quantiles); the tree fuses step k with
covariates TimesFM can't see (holidays, reservations, store stats) and the recursively-built
lags. Leakage-safe both ways: the lags read only the model's own fed-back predictions inside
the window, and the window forecast uses only history ≤ cutoff. See docs/05-timesfm-hybrid.md.

The `Predictor` calls `prepare_window(history, horizon)` once per origin (TimesFM out of the
loop), then `augment` indexes the precomputed window by days-ahead each step. Training uses the
generator's window signal over actual origins (no feedback at train time). `xgboost` vs
`timesfm_hybrid` is the clean ablation — same base learner, the only difference is the signal.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ....domain.entities import DATE, STORE, TARGET
from ....domain.ports.model import ModelWithFeatureImportance
from ....domain.services.evaluation import clip_nonneg
from ..xgboost_model import _DEFAULTS, _feature_cols


class TimesFMHybrid(ModelWithFeatureImportance):
    name = "timesfm_hybrid"
    recursive = True  # lags recurse; TimesFM is forecast once up front (prepare_window)

    def __init__(self, generator, params: dict[str, object] | None = None, *, horizon: int = 14):
        self.generator = generator
        self.horizon = horizon
        self.params = {**_DEFAULTS, **(params or {})}
        self.model = XGBRegressor(**self.params)
        self.features_: list[str] = []
        self._cutoff: pd.Timestamp | None = None
        self._window: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame) -> None:
        panel = train[[STORE, DATE, TARGET]].copy()
        panel[DATE] = pd.to_datetime(panel[DATE])
        origins = sorted(panel[DATE].unique())[-self.generator.max_train_cutoffs :]
        sig = self.generator.training_signal(panel, origins, self.horizon)
        rows = train.merge(sig, on=[STORE, DATE], how="inner")
        self.features_ = _feature_cols(rows)
        X = rows[self.features_]
        y = np.log1p(rows[TARGET].to_numpy())
        self.model.fit(X, y)

    def prepare_window(self, history: pd.DataFrame, horizon: int) -> None:
        """Forecast the whole horizon once at the cutoff (called by Predictor per origin)."""
        h = history.copy()
        h[DATE] = pd.to_datetime(h[DATE])
        self._cutoff = h[DATE].max()
        self._window = self.generator.window_signal(h, horizon)

    def augment(self, features: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        offset = int((pd.to_datetime(features[DATE].iloc[0]) - self._cutoff).days)
        w = self._window[self._window["step"] == offset].drop(columns="step")
        return features.merge(w, on=STORE, how="left")

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
