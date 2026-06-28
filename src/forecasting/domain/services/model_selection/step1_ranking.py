"""Step 1 — rank every model on the FULL feature set with its default params.

A first, cheap pass: CV each candidate as-is and keep the score. The winner here decides
which model drives feature selection in step 2; the scores are reported in the result.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cross_validation import CrossValidator, FoldData
from .progress import log
from .spec import ModelSpec
from .tracking import RunLogger


@dataclass(frozen=True)
class Step1Result:
    scores: dict[str, float]  # model name -> mean CV metric (lower is better)
    best_name: str            # the lowest-scoring model


class Step1ModelRanking:
    def __init__(self, specs: dict[str, ModelSpec], cross_validator: CrossValidator,
                 logger: RunLogger) -> None:
        self.specs = specs
        self.cv = cross_validator
        self.logger = logger

    def run(self, folds: list[FoldData], *, metric_name: str, horizon: int) -> Step1Result:
        scores = {name: self._score_model(spec, folds, metric_name, horizon)
                  for name, spec in self.specs.items()}
        return Step1Result(scores=scores, best_name=min(scores, key=scores.get))

    def _score_model(self, spec: ModelSpec, folds: list[FoldData],
                     metric_name: str, horizon: int) -> float:
        log(f"step1: scoring {spec.name} over {len(folds)} folds…")
        result = self.cv.score(spec, spec.default, folds, selected=None, label=f"step1 {spec.name}")
        log(f"step1: {spec.name} {metric_name}={result.score:.4f}")
        self.logger.log(
            f"step1-{spec.name}",
            {**spec.default, "step": 1, "features": "all", "horizon": horizon},
            {metric_name: result.score, **result.suite},
        )
        return result.score
