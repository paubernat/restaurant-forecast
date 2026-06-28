"""Step 3 — re-CV every model on the SELECTED features, sweeping its small param grid.

Each model's grid contains its step-1 config, so that exact config is always re-tried on
the reduced feature set (its score can still shift — the feature set changed). We keep each
model's best config (with its fold predictions, for the report) and the overall winner.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .cross_validation import CrossValidator, FoldData
from .progress import log
from .spec import ModelSpec
from .tracking import RunLogger


@dataclass(frozen=True)
class ModelBest:
    """A model's best grid config and the fold predictions that produced its score."""
    score: float
    params: dict
    predictions: pd.DataFrame


@dataclass(frozen=True)
class Step3Result:
    winner_name: str
    winner_params: dict
    winner_score: float
    best_per_model: dict[str, ModelBest]  # name -> its best config (drives the step-4 report)


class Step3GridSearch:
    def __init__(self, specs: dict[str, ModelSpec], cross_validator: CrossValidator,
                 logger: RunLogger) -> None:
        self.specs = specs
        self.cv = cross_validator
        self.logger = logger

    def run(self, folds: list[FoldData], selected: list[str], *,
            metric_name: str, horizon: int) -> Step3Result:
        best_per_model = {
            name: self._best_config(spec, folds, selected, metric_name, horizon)
            for name, spec in self.specs.items()
        }
        winner_name, winner = min(best_per_model.items(), key=lambda kv: kv[1].score)
        return Step3Result(winner_name, winner.params, winner.score, best_per_model)

    def _best_config(self, spec: ModelSpec, folds: list[FoldData], selected: list[str],
                     metric_name: str, horizon: int) -> ModelBest:
        best: ModelBest | None = None
        log(f"step3: {spec.name} grid ({len(spec.grid)} configs × {len(folds)} folds)…")
        for i, params in enumerate(spec.grid, 1):
            result = self.cv.score(
                spec, params, folds, selected=selected, label=f"step3 {spec.name} cfg{i}"
            )
            log(f"step3: {spec.name} cfg {i}/{len(spec.grid)} "
                f"{metric_name}={result.score:.4f} {params}")
            self.logger.log(
                f"step3-{spec.name}",
                {**params, "step": 3, "features": "selected", "horizon": horizon},
                {metric_name: result.score, **result.suite},
            )
            if best is None or result.score < best.score:
                best = ModelBest(result.score, params, result.predictions)
        return best
