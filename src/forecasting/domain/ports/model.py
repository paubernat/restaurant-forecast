"""Port: a forecasting model.

Every model — seasonal-naive, LightGBM, XGBoost, TimesFM zero-shot, and the
TimesFM+tree hybrid — implements this same surface, so the use cases treat them
uniformly. `name` drives MLflow run naming and artifact paths.

Two tiers:
  - `Model` — the surface every model has: `fit` (prepare from training data — trees train a
    booster, baselines snapshot a mean/history), `predict`, `save`, `load`.
  - `ModelWithFeatureImportance` — adds `feature_importance`; only the booster-backed models
    (LightGBM, XGBoost, the hybrid's head) can answer it. Feature selection consumes it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class Model(ABC):
    name: str

    # Multi-step forecasting hooks read by `domain/services/Predictor`. Concrete defaults so
    # trees/naive ignore them; the recursive-vs-block models override.
    recursive: bool = True  # False => predict() forecasts the whole horizon block in one call

    def prepare_window(self, history: pd.DataFrame, horizon: int) -> None:  # noqa: B027
        """Called once before the recursive loop with the full history. Default: no-op.

        Optional hook — intentionally concrete (not abstract) so models that don't need it
        inherit the no-op. The hybrid overrides it to run its single cutoff-origin TimesFM
        window forecast up front (TimesFM out of the loop) so `augment` can index it per step.
        """

    def augment(self, features: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        """Attach extra columns for the current days-ahead step. Default: identity.

        The hybrid overrides it to attach the precomputed window signal (no TimesFM call); the
        engineered lags still recurse.
        """
        return features

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> None:
        """Prepare the model from training data (trees train; baselines snapshot state)."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return `features` with a `y_pred` column (plus optional quantile cols)."""

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...


class ModelWithFeatureImportance(Model, ABC):
    @abstractmethod
    def feature_importance(self) -> dict[str, float]:
        """Map feature name -> importance (tree gain). Drives greedy feature selection."""
