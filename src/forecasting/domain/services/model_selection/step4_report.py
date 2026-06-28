"""Step 4 (the old "3.5") — cross-model report + the deployable winner.

Two views, no extra CV pass:
  - **Last-year CV** (reusing the retained step-3 fold predictions): per model an `EvalResult`
    with overall metrics, a per-**season** split (`season(date)`), and **error-by-horizon**
    (one row per days-ahead offset 1..horizon).
  - **Final holdout forecast**: each model fit on the whole CV panel then rolled forward
    `final_horizon` days (the untouched holdout), giving the pred-vs-real frame + tree
    importances.

Then refit the winning config on ALL data (holdout included) to get the model we ship.
`build_report` returns plain data (`ComparisonReport`); plots and MLflow logging live in the
adapters / use case, so the domain stays free of both.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ...entities import DATE, STORE, TARGET, ComparisonReport, EvalResult
from ..evaluation import get_metrics, season
from ..features import subset_features
from ..predictor import Predictor
from .cross_validation import build_panel_features
from .progress import log
from .spec import ModelSpec
from .step3_grid_search import Step3Result


def _suite(df: pd.DataFrame, weights: dict) -> dict[str, float]:
    return get_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy(), **weights)


def _store_regions(raw) -> dict[str, str]:
    """store_id -> prefecture, the first whitespace token of the address (mirrors the
    area_prefecture split in features/builder.py). Empty if no stores table."""
    stores = getattr(raw, "stores", None)
    if stores is None or "area" not in stores.columns:
        return {}
    pref = stores["area"].astype(str).str.split(n=1).str[0]
    return dict(zip(stores[STORE], pref, strict=False))


def _eval_result(
    name: str, preds: pd.DataFrame | None, weights: dict, regions: dict[str, str]
) -> EvalResult:
    if preds is None or preds.empty:
        return EvalResult(model_name=name, metrics={})
    by_segment = {
        s: _suite(g, weights) for s, g in preds.groupby(preds[DATE].map(season), sort=False)
    }
    pref = preds[STORE].map(regions).fillna("unknown")
    by_region = {r: _suite(g, weights) for r, g in preds.groupby(pref, sort=False)}
    by_horizon = pd.DataFrame(
        [
            {"horizon_offset": int(off), **_suite(g, weights)}
            for off, g in preds.groupby("horizon_offset")
        ]
    ).sort_values("horizon_offset", ignore_index=True)
    return EvalResult(name, _suite(preds, weights), by_segment, by_region, by_horizon)


def _holdout_forecast(spec, params, cv_feat, cv_panel, holdout_obs, raw, selected, final_horizon):
    """Fit the model on the whole CV panel, roll it forward `final_horizon` days, align to the
    holdout actuals. Returns (pred-vs-real frame, feature importances or None)."""
    model = spec.factory(params)
    model.fit(subset_features(cv_feat, selected if spec.is_tree else None))
    pred = Predictor(
        model,
        reservations=raw.reservations,
        stores=raw.stores,
        holidays=raw.holidays,
        reference=cv_panel,
    ).infer(cv_panel, final_horizon)
    merged = holdout_obs.merge(pred, on=[STORE, DATE], how="inner")
    frame = merged[[STORE, DATE, "y_pred", TARGET]].rename(columns={TARGET: "y_true"})
    imp = model.feature_importance() if spec.is_tree else None
    return frame, imp


def build_report(
    *,
    specs: dict,
    best_params: dict[str, dict],
    selected: list[str],
    cv_preds: dict[str, pd.DataFrame],
    cv_panel: pd.DataFrame,
    holdout_obs: pd.DataFrame,
    raw,
    horizon: int,
    final_horizon: int,
    weights: dict,
) -> ComparisonReport:
    """Per-model CV breakdown (from retained step-3 preds) + a final holdout forecast each."""
    cv_feat = build_panel_features(raw, cv_panel)
    regions = _store_regions(raw)
    results: dict[str, EvalResult] = {}
    holdout_preds: dict[str, pd.DataFrame] = {}
    importances: dict[str, dict[str, float]] = {}
    for name, spec in specs.items():
        log(f"  step4: holdout forecast {name} (+{final_horizon}d)…")
        results[name] = _eval_result(name, cv_preds.get(name), weights, regions)
        frame, imp = _holdout_forecast(
            spec, best_params[name], cv_feat, cv_panel, holdout_obs, raw, selected, final_horizon
        )
        holdout_preds[name] = frame
        if imp is not None:
            importances[name] = imp
    return ComparisonReport(
        results=results,
        holdout_preds=holdout_preds,
        importances=importances,
        horizon=horizon,
        final_horizon=final_horizon,
    )


@dataclass(frozen=True)
class Step4Result:
    report: object                      # ComparisonReport
    holdout_metrics: dict[str, float]   # the winner's metrics on the final holdout
    winner: object                      # the deployable model, fitted on all data


class Step4Report:
    def __init__(self, specs: dict[str, ModelSpec], raw, *, settings, weights: dict,
                 horizon: int) -> None:
        self.specs = specs
        self.raw = raw
        self.settings = settings
        self.weights = weights
        self.horizon = horizon

    def run(self, *, panel: pd.DataFrame, cv_panel: pd.DataFrame, holdout,
            selected: list[str], tuning: Step3Result) -> Step4Result:
        report = self._build_report(panel, cv_panel, holdout, selected, tuning)
        return Step4Result(
            report=report,
            holdout_metrics=self._winner_holdout_metrics(report, tuning.winner_name),
            winner=self._fit_deployable_winner(panel, selected, tuning),
        )

    def _build_report(self, panel, cv_panel, holdout, selected, tuning: Step3Result):
        holdout_obs = panel[
            (panel[DATE] >= holdout.valid_start) & (panel[DATE] <= holdout.valid_end)
        ][[STORE, DATE, TARGET]].copy()
        holdout_obs[DATE] = pd.to_datetime(holdout_obs[DATE])
        return build_report(
            specs=self.specs,
            best_params={n: b.params for n, b in tuning.best_per_model.items()},
            selected=selected,
            cv_preds={n: b.predictions for n, b in tuning.best_per_model.items()},
            cv_panel=cv_panel,
            holdout_obs=holdout_obs,
            raw=self.raw,
            horizon=self.horizon,
            final_horizon=self.settings.final_horizon_days,
            weights=self.weights,
        )

    def _winner_holdout_metrics(self, report, winner_name: str) -> dict[str, float]:
        preds = report.holdout_preds.get(winner_name)
        if preds is None or preds.empty:
            return {}
        return get_metrics(preds["y_true"].to_numpy(), preds["y_pred"].to_numpy(), **self.weights)

    def _fit_deployable_winner(self, panel, selected: list[str], tuning: Step3Result):
        log(f"  step4: fitting deployable winner {tuning.winner_name} on all data…")
        spec = self.specs[tuning.winner_name]
        features = build_panel_features(self.raw, panel)
        winner = spec.factory(dict(tuning.winner_params))
        winner.fit(subset_features(features, selected if spec.is_tree else None))
        return winner
