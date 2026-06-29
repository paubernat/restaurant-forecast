"""render_report writes the expected PNG set (Agg backend, no display)."""

from __future__ import annotations

import pandas as pd

from forecasting.adapters.plotting import render_report
from forecasting.domain.entities import ComparisonReport, EvalResult


def _holdout(name):
    dates = pd.date_range("2017-04-01", periods=6, freq="D")
    rows = [(s, d, 5.0, 6.0) for s in ("a", "b") for d in dates]
    return pd.DataFrame(rows, columns=["store_id", "date", "y_pred", "y_true"])


def _report():
    by_horizon = pd.DataFrame({"horizon_offset": [1, 2, 3], "rmsle": [0.1, 0.2, 0.3],
                               "mae": [1.0, 1.2, 1.4], "weighted_mae": [1.5, 1.7, 1.9]})
    suite = {"rmsle": 0.2, "mae": 1.0, "weighted_mae": 1.5}
    res = EvalResult(
        "m1",
        suite,
        by_segment={"spring": suite, "winter": suite},
        by_region={"Tokyo": suite, "Osaka": suite},
        by_horizon=by_horizon,
    )
    suite = {"rmsle": 0.2, "mae": 1.0, "weighted_mae": 1.5}
    return ComparisonReport(
        results={"m1": res, "m2": res},
        holdout_preds={"m1": _holdout("m1"), "m2": _holdout("m2")},
        importances={"m1": {"f1": 1.0, "f2": 2.0}},
        horizon=3,
        final_horizon=6,
        holdout_metrics={"m1": suite, "m2": suite},
    )


def test_render_report_writes_all_pngs(tmp_path):
    paths = render_report(_report(), tmp_path)
    names = {p.name for p in paths}
    assert {"pred_vs_real.png", "error_by_horizon.png", "seasonal.png", "residuals.png",
            "error_by_prefecture.png", "holdout_scores.png", "index.html"} <= names
    assert "feature_importance_m1.png" in names
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
