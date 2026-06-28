"""Step 2 — select the feature columns worth keeping.

Runs `FeatureSelector` on the step-1 winner (or the best *tree* if a naive model won, since
naive models have no importances). The selected list is persisted and reused by step 3 and
the deployable fit. Generous by design — step 3 re-CVs on the reduced set anyway.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ...entities import DATE, STORE, TARGET
from ..features import FeatureSelector
from .cross_validation import build_panel_features
from .progress import log
from .spec import ModelSpec
from .step1_ranking import Step1Result
from .tracking import RunLogger


class Step2FeatureSelection:
    def __init__(self, specs: dict[str, ModelSpec], raw, logger: RunLogger, *, settings) -> None:
        self.specs = specs
        self.raw = raw
        self.logger = logger
        self.settings = settings

    def run(self, cv_panel: pd.DataFrame, ranking: Step1Result) -> list[str]:
        base_name = self._feature_selection_model(ranking)
        features = build_panel_features(self.raw, cv_panel)
        if base_name is None:  # no tree to rank importances on -> keep every feature
            selected = [c for c in features.columns if c not in (DATE, STORE, TARGET)]
        else:
            spec = self.specs[base_name]
            selected = FeatureSelector(
                spec.factory(spec.default),
                features,
                threshold=self.settings.feature_select_threshold,
            ).select()
        log(f"step2: kept {len(selected)} features (base={base_name or '<all-features>'})")
        self._persist_and_log(base_name or "<all-features>", selected)
        return selected

    def _feature_selection_model(self, ranking: Step1Result) -> str | None:
        """The step-1 winner if it's a tree, else the best-scoring tree. None when no tree
        exists (seasonal_naive / timesfm_zeroshot are is_tree=False, so excluded)."""
        if self.specs[ranking.best_name].is_tree:
            return ranking.best_name
        trees = [name for name in ranking.scores if self.specs[name].is_tree]
        return min(trees, key=ranking.scores.get) if trees else None

    def _persist_and_log(self, base_name: str, selected: list[str]) -> None:
        artifact = Path(self.settings.artifacts_root) / "selected_features.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(selected, indent=2))
        self.logger.log(
            "step2-feature-selection",
            {
                "base_model": base_name,
                "n_selected": len(selected),
                "threshold": self.settings.feature_select_threshold,
            },
            {},
            artifact=artifact,
        )
