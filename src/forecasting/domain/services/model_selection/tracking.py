"""`RunLogger` — a thin wrapper over the injected experiment tracker.

Each step logs through this instead of touching the tracker directly, so "no tracker
configured" is handled in exactly one place (the methods become no-ops).
"""

from __future__ import annotations

from pathlib import Path


class RunLogger:
    def __init__(self, tracker) -> None:
        self.tracker = tracker

    def log(self, name: str, params: dict, metrics: dict | None = None,
            artifact: Path | None = None) -> None:
        if self.tracker is None:
            return
        self.tracker.start_run(name)
        self.tracker.log_params({k: str(v) for k, v in params.items()})
        if metrics:
            self.tracker.log_metrics(metrics)
        if artifact is not None:
            self.tracker.log_artifact(artifact)
        self.tracker.end_run()
