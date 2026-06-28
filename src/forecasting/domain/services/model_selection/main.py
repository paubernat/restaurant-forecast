"""`ModelSelector` — orchestrates the model + feature selection pipeline (pure domain).

It owns no step logic; it resolves the run config, carves the leakage-safe holdout + CV
folds, then wires the four steps together:

    step 1  rank all models on the full feature set        -> Step1ModelRanking
    step 2  select features on the step-1 winner            -> Step2FeatureSelection
    step 3  re-CV all models on selected features + grid    -> Step3GridSearch
    step 4  cross-model report + deployable winner          -> Step4Report

The model registry (factories + grids) and the tracker are injected, so this service
imports no adapter and no model library — ports only.
"""

from __future__ import annotations

from dataclasses import replace

from ...entities import DATE
from ...validation import final_holdout, rolling_origin_splits
from ..evaluation import selection_metric
from ..features import aggregate_reservations
from .cross_validation import CrossValidator
from .progress import log
from .spec import ModelSpec
from .step1_ranking import Step1ModelRanking, Step1Result
from .step2_feature_selection import Step2FeatureSelection
from .step3_grid_search import Step3GridSearch, Step3Result
from .step4_report import Step4Report
from .tracking import RunLogger


class ModelSelector:
    def __init__(self, *, registry: list[ModelSpec], tracker=None, settings) -> None:
        self.specs = {s.name: s for s in registry}
        self.tracker = tracker
        self.settings = settings

    def run(
        self,
        data,
        *,
        metric_name: str | None = None,
        horizon: int | None = None,
        n_folds: int | None = None,
    ) -> tuple[dict, object, object]:
        """Run the pipeline. Returns (result dict, ComparisonReport, deployable fitted winner)."""
        
        # Aggregate row-level bookings to (store, date) ONCE for the whole run — steps 2 & 4
        # build features straight off `data`, and the CV builds them thousands of times. The
        # per-build re-aggregation guard (FeatureBuilder/_add_reservations) then no-ops.
        if data.reservations is not None and "reserve_count" not in data.reservations.columns:
            data = replace(data, reservations=aggregate_reservations(data.reservations))

        s = self.settings
        metric_name = metric_name or s.selection_metric
        horizon = horizon or s.horizon_days
        n_folds = n_folds or (s.cv_window_days // s.cv_stride_days)  # ~last year of folds
        weights = {"under_weight": s.under_weight, "over_weight": s.over_weight}
        metric_fn = selection_metric(metric_name, **weights)

        panel = data.visits
        holdout = final_holdout(panel[DATE], horizon_days=s.final_horizon_days)
        cv_panel = panel[panel[DATE] <= holdout.train_end]
        splits = rolling_origin_splits(
            cv_panel[DATE], n_folds=n_folds, horizon_days=horizon, stride_days=s.cv_stride_days
        )

        logger = RunLogger(self.tracker)
        cv = CrossValidator(data, horizon=horizon, metric_fn=metric_fn, weights=weights)
        log(f"pipeline: metric={metric_name} horizon={horizon} folds={n_folds} "
            f"models={list(self.specs)} | featurising folds…")
        folds = cv.prepare_folds(cv_panel, splits)

        log(f"STEP 1/4 — ranking {len(self.specs)} models on all features")
        ranking = Step1ModelRanking(self.specs, cv, logger).run(
            folds, metric_name=metric_name, horizon=horizon
        )
        log(f"STEP 2/4 — feature selection (step-1 best={ranking.best_name})")
        selected_features = Step2FeatureSelection(self.specs, data, logger, settings=s).run(
            cv_panel, ranking
        )
        log(f"STEP 3/4 — grid search on {len(selected_features)} selected features")
        tuning = Step3GridSearch(self.specs, cv, logger).run(
            folds, selected_features, metric_name=metric_name, horizon=horizon
        )
        log(f"STEP 4/4 — holdout report + deployable winner (step-3 best={tuning.winner_name})")
        final = Step4Report(self.specs, data, settings=s, weights=weights, horizon=horizon).run(
            panel=panel, cv_panel=cv_panel, holdout=holdout,
            selected=selected_features, tuning=tuning,
        )
        log("DONE — assembling result + summary")

        result = self._assemble_result(
            metric_name, horizon, n_folds, ranking, selected_features, tuning, final
        )
        self._print_summary(result, ranking, tuning)
        return result, final.report, final.winner

    def _assemble_result(self, metric_name, horizon, n_folds, ranking: Step1Result,
                         selected_features, tuning: Step3Result, final) -> dict:
        return {
            "metric": metric_name,
            "horizon": horizon,
            "final_horizon": self.settings.final_horizon_days,
            "n_folds": n_folds,
            "step1": ranking.scores,
            "step1_best": ranking.best_name,
            "selected_features": selected_features,
            "step3_winner": {
                "model": tuning.winner_name,
                "params": dict(tuning.winner_params),
                "score": tuning.winner_score,
            },
            "holdout_metrics": final.holdout_metrics,
        }

    @staticmethod
    def _print_summary(result: dict, ranking: Step1Result, tuning: Step3Result) -> None:
        print(
            f"[find_best_model] metric={result['metric']} horizon={result['horizon']} "
            f"folds={result['n_folds']} | "
            f"step1 best={ranking.best_name} ({ranking.scores[ranking.best_name]:.4f}) | "
            f"selected {len(result['selected_features'])} features | "
            f"winner={tuning.winner_name} {dict(tuning.winner_params)} (cv {tuning.winner_score:.4f})"
        )
