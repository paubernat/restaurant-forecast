"""`CrossValidator` — honest n-day-ahead rolling-origin scoring, shared by steps 1 and 3.

The eval contract: each fold trains on `date <= train_end`, the `Predictor` rolls the
forecast forward `horizon` days (recursive multi-step), and we score against the observed
valid days. No fold ever sees the future — target-based store aggregates are rebuilt per
fold with `reference` = that fold's own train slice (see `build_panel_features`).

`prepare_folds` featurises each fold once up front; `score` then fits a (model, params)
across those prepared folds and returns the mean metric plus the concatenated fold
predictions (which the step-3.5 report reuses).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from ...entities import DATE, STORE, TARGET
from ..evaluation import get_metrics
from ..features import FeatureBuilder, aggregate_reservations, subset_features
from ..predictor import Predictor
from .progress import log
from .spec import ModelSpec


def build_panel_features(raw, panel: pd.DataFrame) -> pd.DataFrame:
    """Featurise a panel with the run's side tables, using the panel itself as `reference`.

    Shared by every featurisation in the pipeline (per-fold train, step-2 selection, the
    step-4 deployable fit) so the leakage-safe `reference=panel` wiring lives in one place.
    """
    return FeatureBuilder(
        panel,
        reservations=raw.reservations,
        stores=raw.stores,
        holidays=raw.holidays,
        reference=panel,
    ).build()


@dataclass(frozen=True)
class FoldData:
    """One prepared CV fold: train rows to fit on + history/cutoff the Predictor needs."""
    train_panel: pd.DataFrame   # raw train rows (history the Predictor rolls from)
    train_features: pd.DataFrame  # featurised train rows (what the model fits on)
    valid_obs: pd.DataFrame     # observed valid rows (what we score against)
    train_end: pd.Timestamp     # cutoff, for the days-ahead horizon_offset


@dataclass(frozen=True)
class CVScore:
    """The outcome of scoring one (model, params) across the folds."""
    score: float                # mean of the selection metric over folds
    suite: dict[str, float]     # mean of the full metric suite over folds
    predictions: pd.DataFrame   # concatenated fold preds [store, date, y_pred, y_true, horizon_offset]


class CrossValidator:
    def __init__(self, raw, *, horizon: int, metric_fn, weights: dict) -> None:
        # Aggregate the row-level bookings to (store, date) ONCE. Otherwise every feature build
        # re-aggregates the multi-million-row table — and the recursive inference loop rebuilds
        # features `horizon` times per fold per (model, params), so it'd run thousands of times.
        if (
            raw.reservations is not None
            and len(raw.reservations)
            and "reserve_count" not in raw.reservations.columns
        ):
            raw = replace(raw, reservations=aggregate_reservations(raw.reservations))
        self.raw = raw
        self.horizon = horizon
        self.metric_fn = metric_fn
        self.weights = weights

    def prepare_folds(self, cv_panel: pd.DataFrame, splits) -> list[FoldData]:
        splits = list(splits)
        folds = []
        for i, split in enumerate(splits, 1):
            folds.append(self._prepare_fold(cv_panel, split))
            log(f"  prep: fold {i}/{len(splits)} featurised (train<= {split.train_end.date()})")
        return folds

    def score(self, spec: ModelSpec, params: dict, folds: list[FoldData], *,
              selected: list[str] | None, label: str | None = None) -> CVScore:
        scores, suites, predictions = [], [], []
        for i, fold in enumerate(folds, 1):
            fold_pred = self._score_fold(spec, params, fold, selected=selected)
            if label:
                log(f"    {label}: fold {i}/{len(folds)} done")
            if fold_pred is None:
                continue
            y_true, y_pred = fold_pred["y_true"].to_numpy(), fold_pred["y_pred"].to_numpy()
            scores.append(self.metric_fn(y_true, y_pred))
            suites.append(get_metrics(y_true, y_pred, **self.weights))
            predictions.append(fold_pred)
        if not scores:
            return CVScore(float("inf"), {}, pd.DataFrame())
        return CVScore(
            score=sum(scores) / len(scores),
            suite={k: sum(s[k] for s in suites) / len(suites) for k in suites[0]},
            predictions=pd.concat(predictions, ignore_index=True),
        )

    def _prepare_fold(self, cv_panel: pd.DataFrame, split) -> FoldData:
        train_panel = cv_panel[cv_panel[DATE] <= split.train_end][[STORE, DATE, TARGET]].copy()
        valid_obs = cv_panel[
            (cv_panel[DATE] >= split.valid_start) & (cv_panel[DATE] <= split.valid_end)
        ][[STORE, DATE, TARGET]].copy()
        return FoldData(
            train_panel=train_panel,
            train_features=build_panel_features(self.raw, train_panel),
            valid_obs=valid_obs,
            train_end=split.train_end,
        )

    def _score_fold(self, spec: ModelSpec, params: dict, fold: FoldData, *,
                    selected: list[str] | None) -> pd.DataFrame | None:
        model = spec.factory(params)
        model.fit(subset_features(fold.train_features, selected if spec.is_tree else None))
        pred = Predictor(
            model,
            reservations=self.raw.reservations,
            stores=self.raw.stores,
            holidays=self.raw.holidays,
            reference=fold.train_panel,
        ).infer(fold.train_panel, self.horizon)
        merged = fold.valid_obs.merge(pred, on=[STORE, DATE], how="inner")
        if merged.empty:
            return None
        fold_pred = merged[[STORE, DATE, "y_pred", TARGET]].rename(columns={TARGET: "y_true"})
        fold_pred["horizon_offset"] = (merged[DATE] - fold.train_end).dt.days.to_numpy()
        return fold_pred
