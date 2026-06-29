"""`find_best_model` use case — select the winner, build the report, persist & log it.

The CV + report maths live in `domain/services` (pure); this layer wires in the adapters the
domain stays free of: it renders the report to PNGs (`adapters.plotting`), persists the
deployable winner + `selection.json` via the injected `ArtifactStore`, and logs the
`report-<model>` + `report-summary` MLflow runs via the injected tracker. Returns the trained,
ready-to-use winner together with the `ComparisonReport`.

The registry (model factories + grids) and all ports are injected from the composition root
(`entrypoints/cli.py`). `ModelSpec` is re-exported so callers keep one import site.
"""

from __future__ import annotations

import json
import pickle

import pandas as pd

from ..adapters.plotting import render_report
from ..domain.services.model_selection import ModelSelector, ModelSpec

__all__ = ["ModelSpec", "run"]


def run(
    *,
    source,
    registry,
    tracker=None,
    artifact_store=None,
    settings,
    metric_name=None,
    horizon=None,
    n_folds=None,
):
    data = source.load()
    result, report, winner = ModelSelector(
        registry=registry, tracker=tracker, settings=settings
    ).run(data, metric_name=metric_name, horizon=horizon, n_folds=n_folds)

    plots = _persist(result, report, winner, artifact_store) if artifact_store is not None else []
    if tracker is not None:
        _log_report(result, report, tracker, artifact_store, plots)
    return winner, report


def _persist(result, report, winner, store):
    """Save the deployable winner + selection.json, render the report PNGs. Returns the paths."""
    winner.save(store.path_for("best_model.pkl"))
    w = result["step3_winner"]
    selection = {
        "model": w["model"],
        "params": w["params"],
        "selected_features": result["selected_features"],
        "horizon": result["horizon"],
        "final_horizon": result["final_horizon"],
        "metric": result["metric"],
        "cv_score": w["score"],
        "holdout_metrics": result["holdout_metrics"],
    }
    store.save("selection.json", json.dumps(selection, indent=2).encode())
    # Persist the report (pure data) so charts can be re-rendered with `render-report` — no
    # model run, no endpoint — after any plotting tweak.
    store.save("report.pkl", pickle.dumps(report))
    return render_report(report, store.path_for("report"))


def _flatten(res) -> dict[str, float]:
    """EvalResult -> flat MLflow metrics: overall + per-season suite + RMSLE per horizon."""
    out = dict(res.metrics)
    for s, m in res.by_segment.items():
        out.update({f"{k}_{s}": v for k, v in m.items()})
    if res.by_horizon is not None and not res.by_horizon.empty:
        for _, row in res.by_horizon.iterrows():
            out[f"rmsle_h{int(row['horizon_offset'])}"] = float(row["rmsle"])
    return out


def _write_csvs(name, res, store) -> list:
    paths = []
    if res.by_horizon is not None and not res.by_horizon.empty:
        p = store.path_for(f"report/by_horizon_{name}.csv")
        p.parent.mkdir(parents=True, exist_ok=True)
        res.by_horizon.to_csv(p, index=False)
        paths.append(p)
    if res.by_segment:
        p = store.path_for(f"report/by_season_{name}.csv")
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(res.by_segment).T.to_csv(p)  # season rows, metric columns
        paths.append(p)
    return paths


def _log_report(result, report, tracker, store, plots) -> None:
    prefix = "feature_importance_"
    fi = {p.stem.removeprefix(prefix): p for p in plots if prefix in p.stem}
    cross = [p for p in plots if prefix not in p.stem]

    for name, res in report.results.items():
        tracker.start_run(f"report-{name}")
        metrics = _flatten(res)
        if metrics:
            tracker.log_metrics(metrics)
        if store is not None:
            for csv in _write_csvs(name, res, store):
                tracker.log_artifact(csv)
        if name in fi:
            tracker.log_artifact(fi[name])
        tracker.end_run()

    tracker.start_run("report-summary")
    w = result["step3_winner"]
    tracker.log_params(
        {
            "winner": w["model"],
            "metric": result["metric"],
            "horizon": result["horizon"],
            "folds": result["n_folds"],
            "final_horizon": result["final_horizon"],
            **{f"param_{k}": str(v) for k, v in w["params"].items()},
        }
    )
    if result["holdout_metrics"]:
        tracker.log_metrics({f"holdout_{k}": v for k, v in result["holdout_metrics"].items()})
    for p in cross:
        tracker.log_artifact(p)
    if store is not None:
        tracker.log_artifact(store.path_for("selection.json"))
        tracker.log_artifact(store.path_for("best_model.pkl"))
    tracker.end_run()
