"""find_best_model smoke: the 3-step pipeline returns a winner + selected features."""

from __future__ import annotations

import json
from types import SimpleNamespace

from forecasting.adapters.local_artifacts import LocalArtifactStore
from forecasting.adapters.models.lightgbm_model import LightGBMModel
from forecasting.adapters.models.seasonal_naive import SeasonalNaive
from forecasting.adapters.models.xgboost_model import XGBoostModel
from forecasting.application import find_best_model as fbm
from forecasting.domain.entities import ComparisonReport
from forecasting.entrypoints.cli import _grid, _registry


def _tiny_registry():
    return [
        fbm.ModelSpec("seasonal_naive", lambda p: SeasonalNaive()),
        fbm.ModelSpec(
            "lightgbm",
            lambda p: LightGBMModel({**p, "n_estimators": 15}),
            default={"learning_rate": 0.05},
            grid=[{"learning_rate": 0.05}, {"learning_rate": 0.1}],
            is_tree=True,
        ),
        fbm.ModelSpec(
            "xgboost",
            lambda p: XGBoostModel({**p, "n_estimators": 15}),
            default={"max_depth": 4},
            grid=[{"max_depth": 4}, {"max_depth": 6}],
            is_tree=True,
        ),
    ]


def test_pipeline_persists_winner_and_returns_report(raw, tmp_path):
    settings = SimpleNamespace(
        horizon_days=7,
        final_horizon_days=10,
        cv_window_days=365,
        cv_stride_days=5,
        selection_metric="rmsle",
        feature_select_threshold=0.95,
        under_weight=2.0,
        over_weight=1.0,
        artifacts_root=tmp_path,
    )
    winner, report = fbm.run(
        source=SimpleNamespace(load=lambda: raw),
        registry=_tiny_registry(),
        tracker=None,
        artifact_store=LocalArtifactStore(tmp_path),
        settings=settings,
        metric_name="weighted_mae",
        n_folds=2,
    )
    # Return shape: a trained, usable winner + the comparison report.
    assert hasattr(winner, "predict")
    assert isinstance(report, ComparisonReport)
    assert set(report.results) == {"seasonal_naive", "lightgbm", "xgboost"}

    # Persisted artifacts the future train-job reads.
    assert (tmp_path / "best_model.pkl").exists()
    assert (tmp_path / "selected_features.json").exists()
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["model"] in {"seasonal_naive", "lightgbm", "xgboost"}
    assert selection["horizon"] == 7 and selection["final_horizon"] == 10
    assert selection["selected_features"]  # non-empty
    assert (tmp_path / "report" / "pred_vs_real.png").exists()


def test_real_registry_grid_contains_step1_config():
    # The guarantee: every tree model's step-3 grid includes its step-1 default.
    for spec in _registry(horizon=7):
        if spec.is_tree:
            assert spec.default in spec.grid


def test_grid_builds_cartesian_product():
    grid = _grid({"a": [1, 2], "b": [3, 4]})
    assert {"a": 1, "b": 3} in grid and len(grid) == 4
