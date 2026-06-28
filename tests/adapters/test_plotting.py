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
    by_horizon = pd.DataFrame({"horizon_offset": [1, 2, 3], "rmsle": [0.1, 0.2, 0.3]})
    res = EvalResult(
        "m1",
        {"rmsle": 0.2},
        {"spring": {"rmsle": 0.2}, "winter": {"rmsle": 0.3}},
        by_horizon,
    )
    return ComparisonReport(
        results={"m1": res, "m2": res},
        holdout_preds={"m1": _holdout("m1"), "m2": _holdout("m2")},
        importances={"m1": {"f1": 1.0, "f2": 2.0}},
        horizon=3,
        final_horizon=6,
    )


def test_render_report_writes_all_pngs(tmp_path):
    paths = render_report(_report(), tmp_path)
    names = {p.name for p in paths}
    assert {"pred_vs_real.png", "error_by_horizon.png", "seasonal.png", "residuals.png"} <= names
    assert "feature_importance_m1.png" in names
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
