"""predict use case: rebuild + load the deployed winner, forecast the next horizon, write CSV."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pandas as pd

from forecasting.adapters.local_artifacts import LocalArtifactStore
from forecasting.adapters.models.xgboost_model import XGBoostModel
from forecasting.application import find_best_model as fbm
from forecasting.application import predict, train


def _registry():
    return [
        fbm.ModelSpec(
            "xgboost", lambda p: XGBoostModel({**p, "n_estimators": 15}), is_tree=True
        )
    ]


def test_predict_forecasts_next_horizon_after_last_day(raw, tmp_path):
    store = LocalArtifactStore(tmp_path)
    store.save(
        "selection.json",
        json.dumps(
            {
                "model": "xgboost",
                "params": {"max_depth": 3},
                "selected_features": ["lag_7", "dow", "store_mean"],
                "horizon": 7,
                "final_horizon": 39,
                "metric": "rmsle",
                "cv_score": 0.0,
                "holdout_metrics": {},
            }
        ).encode(),
    )
    src = SimpleNamespace(load=lambda: raw)
    train.run(source=src, registry=_registry(), artifact_store=store, settings=SimpleNamespace())

    out = predict.run(
        source=src, registry=_registry(), artifact_store=store, settings=SimpleNamespace()
    )

    assert (tmp_path / "forecasts.csv").exists()
    written = pd.read_csv(io.BytesIO(store.load("forecasts.csv")), parse_dates=["date"])
    n_stores = raw.visits["store_id"].nunique()
    assert len(written) == n_stores * 7  # one row per (store, horizon-day)
    assert written["date"].min() > pd.to_datetime(raw.visits["date"]).max()  # strictly future
    assert (written["y_pred"] >= 0).all()
    assert list(out.columns) == ["store_id", "date", "y_pred"]
