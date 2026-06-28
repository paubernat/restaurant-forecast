"""Port: experiment tracking (implemented by the MLflow adapter)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ExperimentTracker(ABC):
    @abstractmethod
    def start_run(self, name: str) -> None: ...

    @abstractmethod
    def log_params(self, params: dict[str, object]) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float]) -> None: ...

    @abstractmethod
    def log_artifact(self, path: Path) -> None: ...

    @abstractmethod
    def end_run(self) -> None: ...
