"""ModelSelector end-to-end smoke (fakes, no model libraries): the 4 steps wire together
and return the documented (result, report, winner) shape, with the step-2 artifact written."""

from __future__ import annotations

from types import SimpleNamespace

from forecasting.domain.entities import ComparisonReport
from forecasting.domain.services.model_selection import ModelSelector, ModelSpec


class _Naive:
    recursive = False  # one-shot constant keeps the smoke fast

    def fit(self, train):
        self.m = float(train["visitors"].mean())

    def predict(self, features):
        return features.assign(y_pred=self.m)


class _Tree(_Naive):
    def fit(self, train):
        super().fit(train)
        self.cols = [c for c in train.columns if c not in ("store_id", "date", "visitors")]

    def feature_importance(self):
        # decreasing importances so FeatureSelector keeps a non-trivial subset
        return {c: float(len(self.cols) - i) for i, c in enumerate(self.cols)}


def test_pipeline_runs_all_steps_and_returns_winner(raw, tmp_path):
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
    registry = [
        ModelSpec("naive", lambda p: _Naive(), is_tree=False),
        ModelSpec("tree", lambda p: _Tree(), default={"d": 1}, grid=[{"d": 1}, {"d": 2}],
                  is_tree=True),
    ]

    result, report, winner = ModelSelector(
        registry=registry, tracker=None, settings=settings
    ).run(raw, metric_name="weighted_mae", n_folds=2)

    # Result shape (steps 1-4 all contributed).
    assert result["step1_best"] in {"naive", "tree"}
    assert set(result["step1"]) == {"naive", "tree"}
    assert result["step3_winner"]["model"] in {"naive", "tree"}
    assert result["selected_features"]  # non-empty
    assert result["holdout_metrics"]    # winner scored on the holdout

    # Returned winner + report.
    assert hasattr(winner, "predict")
    assert isinstance(report, ComparisonReport)
    assert set(report.results) == {"naive", "tree"}

    # Step-2 artifact persisted.
    assert (tmp_path / "selected_features.json").exists()
