"""ExperimentTracker backed by MLflow.

Logs params/metrics/artifacts per run against the tracking backend in
`config.Settings.mlflow_tracking_uri` (default `sqlite:///mlflow.db`, no server needed — the
file store is deprecated as of MLflow 3.x) so the model-selection comparison is reproducible
and browsable in the MLflow UI (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import mlflow

from ..domain.ports.tracker import ExperimentTracker


class MLflowTracker(ExperimentTracker):
    def __init__(
        self, experiment: str = "gstock-forecasting", tracking_uri: str | None = None
    ) -> None:
        self.experiment = experiment
        # One id per CV run (this tracker lives for a single `evaluate`/`find-best-model`
        # invocation). Tagged onto every run below so they group/filter together in the UI:
        # `mlflow runs ... --filter "tags.cv_run_id = '<id>'"`.
        self.cv_run_id = uuid4().hex
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    def start_run(self, name: str) -> None:
        mlflow.start_run(run_name=name)
        mlflow.set_tag("cv_run_id", self.cv_run_id)

    def log_params(self, params: dict[str, object]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    def log_artifact(self, path: Path) -> None:
        mlflow.log_artifact(str(path))

    def end_run(self) -> None:
        mlflow.end_run()
