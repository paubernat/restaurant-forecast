"""build_report: per-model season + horizon breakdowns from retained CV preds, plus a final
holdout forecast per model — with fakes (no model libraries)."""

from __future__ import annotations

import pandas as pd

from forecasting.domain.entities import ComparisonReport
from forecasting.domain.services.evaluation import get_metrics
from forecasting.domain.services.model_selection import ModelSpec
from forecasting.domain.services.model_selection.step4_report import (
    _eval_result,
    _store_regions,
    build_report,
)


class _Flat:
    """One-shot constant model (recursive=False keeps the holdout forecast cheap)."""

    name = "flat"
    recursive = False

    def fit(self, train):
        self.m = float(train["visitors"].mean())

    def predict(self, features):
        return features.assign(y_pred=self.m)


class _Tree(_Flat):
    name = "tree"

    def fit(self, train):
        super().fit(train)
        self.cols = [c for c in train.columns if c not in ("store_id", "date", "visitors")]

    def feature_importance(self):
        return {c: 1.0 for c in self.cols}


def _cv_preds():
    """Four seasons × offsets 1..3, so by_segment has 4 keys and by_horizon has 3 rows."""
    rows = []
    for month in ("2016-01", "2016-04", "2016-07", "2016-10"):  # winter/spring/summer/autumn
        for off in (1, 2, 3):
            rows.append(("s", pd.Timestamp(f"{month}-1{off}"), 5.0, 6.0, off))
    return pd.DataFrame(rows, columns=["store_id", "date", "y_pred", "y_true", "horizon_offset"])


def test_build_report_breaks_down_by_season_and_horizon(raw):
    cutoff = pd.Timestamp("2016-05-21")
    cv_panel = raw.visits[raw.visits["date"] <= cutoff]
    holdout_obs = raw.visits[raw.visits["date"] > cutoff][["store_id", "date", "visitors"]].copy()
    specs = {
        "flat": ModelSpec("flat", lambda p: _Flat(), is_tree=False),
        "tree": ModelSpec("tree", lambda p: _Tree(), is_tree=True),
    }
    preds = _cv_preds()

    report = build_report(
        specs=specs,
        best_params={"flat": {}, "tree": {}},
        selected=["lag_7", "dow", "store_mean"],
        cv_preds={"flat": preds, "tree": preds},
        cv_panel=cv_panel,
        holdout_obs=holdout_obs,
        raw=raw,
        horizon=3,
        final_horizon=10,
        weights={"under_weight": 1.0, "over_weight": 1.0},
    )

    assert isinstance(report, ComparisonReport)
    res = report.results["tree"]
    assert set(res.by_segment) == {"winter", "spring", "summer", "autumn"}
    assert list(res.by_horizon["horizon_offset"]) == [1, 2, 3]
    assert res.metrics["rmsle"] >= 0.0

    # Final holdout forecast is aligned to the held-out actuals.
    hp = report.holdout_preds["tree"]
    assert not hp.empty and set(hp.columns) >= {"store_id", "date", "y_pred", "y_true"}
    assert hp["date"].min() > cutoff

    # Importances only for the tree spec.
    assert "tree" in report.importances and "flat" not in report.importances
    assert set(report.importances["tree"]) == {"lag_7", "dow", "store_mean"}


def test_eval_result_breaks_down_by_prefecture(raw):
    """by_region groups CV preds by store prefecture (area's first token) and the per-region
    metrics match the suite computed on that region's rows alone."""
    regions = _store_regions(raw)
    assert regions["air_a"] == "Tokyo" and regions["air_c"] == "Osaka"  # from the fixture areas

    preds = pd.DataFrame({
        "store_id": ["air_a", "air_a", "air_c", "air_c"],
        "date": pd.to_datetime(["2016-04-10", "2016-04-11", "2016-04-10", "2016-04-11"]),
        "y_pred": [5.0, 7.0, 9.0, 4.0],
        "y_true": [6.0, 6.0, 8.0, 5.0],
        "horizon_offset": [1, 2, 1, 2],
    })
    weights = {"under_weight": 1.0, "over_weight": 1.0}
    res = _eval_result("m", preds, weights, regions)

    assert set(res.by_region) == {"Tokyo", "Osaka"}
    osaka = preds[preds["store_id"] == "air_c"]
    expected = get_metrics(osaka["y_true"].to_numpy(), osaka["y_pred"].to_numpy(), **weights)
    assert res.by_region["Osaka"] == expected
