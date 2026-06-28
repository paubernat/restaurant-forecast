"""Runtime configuration (env-overridable via pydantic-settings)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Auto-loads .env (gitignored). Precedence: real env vars > .env > defaults. So a token
    # exported in your shell wins, and `python -m forecasting ...` / docker / make all pick it up.
    model_config = SettingsConfigDict(env_prefix="FORECAST_", env_file=".env", extra="ignore")

    data_root: Path = Path("data/raw")
    artifacts_root: Path = Path("artifacts")
    # SQLite backend (the file store is deprecated/maintenance-mode as of MLflow 3.x);
    # browse with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    # Validation
    # Forecast horizon: n days ahead from one cutoff (recursive multi-step). 14 = the
    # operational ordering window; override per run with `--horizon`.
    horizon_days: int = 14
    # The separate, longer window for the headline pred-vs-real holdout (carved off first,
    # never touched by CV). 39 = the Recruit holdout (2017-04-23 → 2017-05-31), so the demo
    # forecast mirrors the dataset's official test window.
    final_horizon_days: int = 39
    # CV reads (roughly) the last year; the use case derives n_folds = cv_window_days //
    # cv_stride_days. Override per run with `--folds`.
    cv_window_days: int = 365
    # Days between consecutive fold origins. Coprime with 7 (9) so the forecast origin rotates
    # through every weekday instead of locking to one — at stride < horizon the windows overlap
    # (correlated folds, more compute). 9 → ~40 folds over the year (vs 73 at stride 5). Set =
    # horizon_days for the classic non-overlapping CV.
    cv_stride_days: int = 9

    # Model selection (find_best_model)
    selection_metric: str = "rmsle"  # metric the CV ranks by in steps 1 & 3
    feature_select_threshold: float = 0.95  # keep features up to this cumulative gain
    # weighted_mae: under-prediction (stockout) penalised harder than over (waste)
    under_weight: float = 2.0
    over_weight: float = 1.0

    # TimesFM
    timesfm_max_context: int = 512  # history length the remote forecaster trims each series to
    # TimesFM is served only over HTTP from a GPU Hugging Face Space (never run locally). When
    # set, the CV POSTs forecasts there via RemoteTimesFMForecaster (needs only `requests`).
    # Empty = TimesFM is skipped and the CV runs the tree/baseline models. Token = bearer header.
    timesfm_endpoint: str = ""
    timesfm_endpoint_token: str = ""
    # Hybrid: how many recent window-forecast origins to compute the TimesFM signal for
    # (bounds the precompute cost; the head trains on those windows). Subsampled = the most
    # recent. One TimesFM call per origin; lower this if the run is too long.
    timesfm_train_cutoffs: int = 60
    # A TimesFM forecast is a pure function of (history <= cutoff, horizon), so cache it on disk
    # and reuse across folds, steps, both TimesFM models, AND reruns — each unique forecast hits
    # the GPU once, ever. Path is gitignored; empty = caching disabled (always call the endpoint).
    timesfm_cache_dir: str = ".cache/timesfm"


settings = Settings()
