"""Selects which feature columns to keep, by cumulative tree importance (pure domain).

`FeatureSelector(model, features).select()` fits a tree model on the full feature matrix,
ranks the columns by how much they contributed, and keeps the smallest top set that still
accounts for `threshold` of the total contribution. Mirrors `FeatureBuilder`: construct
with inputs, call one verb.

The selector talks to a model only through `.fit()` / `.feature_importance()`, so it
imports no model library and knows nothing about the model registry.
"""

from __future__ import annotations

import pandas as pd

from ...entities import DATE, STORE, TARGET
from ...ports.model import ModelWithFeatureImportance


def subset_features(frame: pd.DataFrame, selected: list[str] | None) -> pd.DataFrame:
    """Keep only the `selected` feature columns (+ id/target); `None` => keep all.

    Applies the names `FeatureSelector.select()` returned to an actual feature frame, while
    always retaining the keys/target the model and scoring need.
    """
    if selected is None:
        return frame
    keep = set(selected) | {DATE, TARGET, STORE}
    return frame[[c for c in frame.columns if c in keep]]


class FeatureSelector:
    """Picks the feature columns worth keeping, by cumulative tree importance.

    Inputs
    ------
    model : a tree model exposing `.fit(features)` and `.feature_importance() -> {name: score}`.
        Passed in already configured (the caller chose the model and its params) so this
        class stays free of the model registry.
    features : the built feature frame to fit on (output of `FeatureBuilder.build()`).
    threshold : keep features in importance order until their cumulative share of the total
        importance reaches this fraction (e.g. 0.99 keeps everything but the long tail).

    Returns (from `.select()`)
    --------------------------
    The kept feature names, most-important first. Zero/negative-importance features are
    always dropped, so the result can be shorter than the threshold alone implies.
    """

    def __init__(
        self, model: ModelWithFeatureImportance, features: pd.DataFrame, *, threshold: float = 0.95
    ) -> None:
        self.model = model
        self.features = features
        self.threshold = threshold

    def select(self) -> list[str]:
        self.model.fit(self.features)
        importance = self.model.feature_importance()
        total = sum(importance.values())
        kept, cumulative = [], 0.0
        for name in sorted(importance, key=importance.get, reverse=True):
            if importance[name] <= 0:
                break
            kept.append(name)
            cumulative += importance[name]
            if total > 0 and cumulative / total >= self.threshold:
                break
        return kept
