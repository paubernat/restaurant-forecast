"""Model adapters: fit -> predict (non-negative, finite) + save/load round-trip."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.adapters.models.lightgbm_model import LightGBMModel
from forecasting.adapters.models.seasonal_naive import SeasonalNaive
from forecasting.adapters.models.xgboost_model import XGBoostModel
from forecasting.domain.ports.model import ModelWithFeatureImportance

CUTOFF = pd.Timestamp("2016-05-01")


@pytest.mark.parametrize("Model", [LightGBMModel, XGBoostModel])
def test_tree_model_fit_predict_roundtrip(Model, feat, tmp_path):
    train = feat[feat["date"] < CUTOFF]
    valid = feat[feat["date"] >= CUTOFF]

    model = Model({"n_estimators": 20})
    model.fit(train)
    y_pred = model.predict(valid)["y_pred"].to_numpy()
    assert np.isfinite(y_pred).all() and (y_pred >= 0).all()

    path = tmp_path / f"{model.name}.pkl"
    model.save(path)
    reloaded = Model()
    reloaded.load(path)
    assert np.allclose(reloaded.predict(valid)["y_pred"].to_numpy(), y_pred)


@pytest.mark.parametrize("Model", [LightGBMModel, XGBoostModel])
def test_tree_model_is_feature_importance_model(Model):
    assert isinstance(Model(), ModelWithFeatureImportance)


def test_tree_model_feature_importance_nonempty(feat):
    model = LightGBMModel({"n_estimators": 20})
    model.fit(feat[feat["date"] < CUTOFF])
    imp = model.feature_importance()
    assert imp and all(v >= 0 for v in imp.values()) and sum(imp.values()) > 0


def test_seasonal_naive_predicts_lag7(feat):
    model = SeasonalNaive()
    model.fit(feat)
    out = model.predict(feat)
    assert (out["y_pred"] >= 0).all()
    # Where lag_7 exists, the naive prediction is exactly it (clipped >= 0).
    has_lag = feat["lag_7"].notna()
    assert np.allclose(
        out.loc[has_lag, "y_pred"].to_numpy(),
        feat.loc[has_lag, "lag_7"].clip(lower=0).to_numpy(),
    )
