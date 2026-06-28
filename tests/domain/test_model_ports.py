"""The model port ABCs.

Every adapter satisfies `Model`; only booster-backed models satisfy
`ModelWithFeatureImportance`. Guards the inheritance wiring (and the ABC enforcement) so a
mis-parented or incomplete model fails loudly instead of at some call site deep in the pipeline.
Uses the light baseline adapters + tiny stubs, so it runs without lightgbm/xgboost installed.
"""

from __future__ import annotations

import pytest

from forecasting.adapters.models.seasonal_naive import SeasonalNaive
from forecasting.adapters.models.timesfm.zeroshot import TimesFMZeroShot
from forecasting.domain.ports.model import Model, ModelWithFeatureImportance


class _Base(Model):
    name = "base"

    def fit(self, train): ...
    def predict(self, features):
        return features

    def save(self, path): ...
    def load(self, path): ...


class _WithImportance(_Base, ModelWithFeatureImportance):
    name = "fi"

    def feature_importance(self):
        return {}


def test_plain_model_is_not_a_feature_importance_model():
    m = _Base()
    assert isinstance(m, Model)
    assert not isinstance(m, ModelWithFeatureImportance)


def test_feature_importance_model_is_also_a_model():
    m = _WithImportance()
    assert isinstance(m, Model) and isinstance(m, ModelWithFeatureImportance)


def test_missing_feature_importance_cannot_instantiate():
    class _Broken(ModelWithFeatureImportance):
        name = "broken"

        def fit(self, train): ...
        def predict(self, features):
            return features

        def save(self, path): ...
        def load(self, path): ...
        # feature_importance deliberately missing

    with pytest.raises(TypeError):
        _Broken()


def test_baseline_adapters_are_plain_models():
    for m in (SeasonalNaive(), TimesFMZeroShot(forecaster=None)):
        assert isinstance(m, Model)
        assert not isinstance(m, ModelWithFeatureImportance)
