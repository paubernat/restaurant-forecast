"""CLI / composition root: `python -m forecasting {train,predict,evaluate,find-best-model}`.

This is the only place adapters get wired to use cases; everything below stays decoupled
behind ports.
"""

from __future__ import annotations

import argparse
import json
from itertools import product

from ..adapters.data.recruit_csv import RecruitCsvSource
from ..adapters.local_artifacts import LocalArtifactStore
from ..adapters.mlflow_tracker import MLflowTracker
from ..adapters.models.lightgbm_model import LightGBMModel
from ..adapters.models.seasonal_naive import SeasonalNaive
from ..adapters.models.xgboost_model import XGBoostModel
from ..application import find_best_model, predict, train
from ..application.find_best_model import ModelSpec
from ..config import settings


def _grid(options: dict[str, list]) -> list[dict]:
    """Cartesian product of param options -> list of full param dicts."""
    keys = list(options)
    return [dict(zip(keys, combo, strict=True)) for combo in product(*options.values())]


def _make_forecaster(horizon: int):
    """The TimesFM forecaster for this run: the remote GPU endpoint if configured, else None
    (the CV then runs the tree/baseline models only). TimesFM is served only over HTTP from a
    Hugging Face Space (see `space/app.py`) — never downloaded or run locally. Set
    `FORECAST_TIMESFM_ENDPOINT` to include it.

    Configured with `max_horizon = max(horizon, final_horizon_days)` so the same forecaster
    serves both the CV horizon and the longer final pred-vs-real forecast (the report's 39-day
    holdout); the Space itself is compiled with a `max_horizon` that covers both."""
    if not settings.timesfm_endpoint:
        return None
    from ..adapters.models.timesfm.remote import RemoteTimesFMForecaster

    return RemoteTimesFMForecaster(max_horizon=max(horizon, settings.final_horizon_days))


def _timesfm_specs(horizon: int) -> list[ModelSpec]:
    """TimesFM 2.5 the two ways — only when a remote forecaster endpoint is configured.

    Without `FORECAST_TIMESFM_ENDPOINT`, the CV runs the three tree/baseline models; TimesFM is
    served only over HTTP from a GPU Hugging Face Space (never run locally), so the CV machine
    needs no torch/timesfm. A single forecaster is shared by both specs.
    """
    forecaster = _make_forecaster(horizon)
    if forecaster is None:
        return []
    from ..adapters.models.timesfm.features import TimesFMFeatureGenerator
    from ..adapters.models.timesfm.hybrid import TimesFMHybrid
    from ..adapters.models.timesfm.zeroshot import TimesFMZeroShot

    generator = TimesFMFeatureGenerator(
        forecaster, max_train_cutoffs=settings.timesfm_train_cutoffs
    )
    hybrid_default = {"learning_rate": 0.05, "max_depth": 6}
    return [
        ModelSpec("timesfm_zeroshot", lambda p: TimesFMZeroShot(forecaster)),
        # Grid kept to just the step-1 config: each grid combo refits the head (and reruns the
        # recursive inference loop), so we bound the cost (still satisfies grid-contains-default).
        ModelSpec(
            "timesfm_hybrid",
            lambda p: TimesFMHybrid(generator, p, horizon=horizon),
            default=hybrid_default,
            grid=[hybrid_default],
            is_tree=True,
        ),
    ]


def _registry(horizon: int, *, offline: bool = False) -> list[ModelSpec]:
    base = [
        ModelSpec("seasonal_naive", lambda p: SeasonalNaive()),
        ModelSpec(
            "lightgbm",
            lambda p: LightGBMModel(p),
            default={"learning_rate": 0.05, "num_leaves": 31},
            grid=_grid({"learning_rate": [0.05, 0.1], "num_leaves": [31, 63]}),
            is_tree=True,
        ),
        ModelSpec(
            "xgboost",
            lambda p: XGBoostModel(p),
            default={"learning_rate": 0.05, "max_depth": 6},
            grid=_grid({"learning_rate": [0.05, 0.1], "max_depth": [6, 8]}),
            is_tree=True,
        ),
    ]
    # Offline: tree/baseline models only — never builds the TimesFM forecaster, so it needs no
    # endpoint and makes zero network calls (the CV runs fully local).
    if offline:
        return base
    return [*base, *_timesfm_specs(horizon)]


def _find_best_model(
    metric: str | None, horizon: int | None, folds: int | None, *, offline: bool = False
) -> None:
    """`find-best-model` (and its `evaluate` alias): full CV selection + comparison report."""
    horizon = horizon or settings.horizon_days
    find_best_model.run(
        source=RecruitCsvSource(settings.data_root),
        registry=_registry(horizon, offline=offline),
        tracker=MLflowTracker(tracking_uri=settings.mlflow_tracking_uri),
        artifact_store=LocalArtifactStore(settings.artifacts_root),
        settings=settings,
        metric_name=metric,
        horizon=horizon,
        n_folds=folds,
    )


def _selection_horizon(store: LocalArtifactStore) -> int:
    """Horizon the winner was selected at (find-best-model wrote it to selection.json); it
    drives the registry that train/predict rebuild the architecture from."""
    path = store.path_for("selection.json")
    if not path.exists():
        raise SystemExit("No artifacts/selection.json — run `find-best-model` first.")
    return int(json.loads(path.read_text())["horizon"])


def _train() -> None:
    """`train`: retrain the selected winner on all current data, re-save best_model.pkl."""
    store = LocalArtifactStore(settings.artifacts_root)
    train.run(
        source=RecruitCsvSource(settings.data_root),
        registry=_registry(_selection_horizon(store)),
        artifact_store=store,
        tracker=MLflowTracker(tracking_uri=settings.mlflow_tracking_uri),
        settings=settings,
    )


def _predict(horizon: int | None) -> None:
    """`predict`: batch-forecast the next horizon from the deployed model -> forecasts.csv."""
    store = LocalArtifactStore(settings.artifacts_root)
    predict.run(
        source=RecruitCsvSource(settings.data_root),
        registry=_registry(_selection_horizon(store)),
        artifact_store=store,
        settings=settings,
        horizon=horizon,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="forecasting")
    parser.add_argument("command", choices=["train", "predict", "evaluate", "find-best-model"])
    parser.add_argument(
        "--metric", default=None, help="CV ranking metric (e.g. rmsle, weighted_mae)"
    )
    parser.add_argument(
        "--horizon", type=int, default=None, help="forecast horizon in days (default: 14)"
    )
    parser.add_argument(
        "--folds", type=int, default=None,
        help="CV folds (default: ~cv_window_days // cv_stride_days)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="tree/baseline models only (naive, lightgbm, xgboost) — skip TimesFM, no endpoint/network",
    )
    args = parser.parse_args()
    if args.command in ("find-best-model", "evaluate"):
        _find_best_model(args.metric, args.horizon, args.folds, offline=args.offline)
    elif args.command == "train":
        _train()
    elif args.command == "predict":
        _predict(args.horizon)


if __name__ == "__main__":
    main()
