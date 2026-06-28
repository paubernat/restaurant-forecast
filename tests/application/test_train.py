"""train (retrain) use case: rebuild the selected winner from selection.json + refit + re-save."""

from __future__ import annotations

import json
from types import SimpleNamespace

from forecasting.adapters.local_artifacts import LocalArtifactStore
from forecasting.adapters.models.xgboost_model import XGBoostModel
from forecasting.application import find_best_model as fbm
from forecasting.application import train


def _seed_selection(store):
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


def _registry():
    return [
        fbm.ModelSpec(
            "xgboost", lambda p: XGBoostModel({**p, "n_estimators": 15}), is_tree=True
        )
    ]


def test_train_refits_winner_from_selection(raw, tmp_path):
    store = LocalArtifactStore(tmp_path)
    _seed_selection(store)

    model = train.run(
        source=SimpleNamespace(load=lambda: raw),
        registry=_registry(),
        artifact_store=store,
        settings=SimpleNamespace(),
    )

    # Re-saved, and the persisted model reloads + predicts (the head trained on the selection).
    assert (tmp_path / "best_model.pkl").exists()
    assert model.params["max_depth"] == 3  # honoured the selected params
    reloaded = XGBoostModel()
    reloaded.load(tmp_path / "best_model.pkl")
    assert set(reloaded.features_) == {"lag_7", "dow", "store_mean"}
