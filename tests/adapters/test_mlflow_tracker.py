"""MLflowTracker smoke: a logged run is recorded in the file store."""

from __future__ import annotations

import mlflow

from forecasting.adapters.mlflow_tracker import MLflowTracker


def test_tracker_logs_a_run(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tracker = MLflowTracker(experiment="test-exp", tracking_uri=uri)
    tracker.start_run("run-1")
    tracker.log_params({"model": "lightgbm"})
    tracker.log_metrics({"rmsle": 0.42})
    tracker.end_run()

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["test-exp"])
    assert len(runs) >= 1
    assert runs.iloc[0]["metrics.rmsle"] == 0.42
