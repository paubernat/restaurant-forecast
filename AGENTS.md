# AGENTS.md — working conventions for this repo

Guidance for any agent (human or AI) contributing here. Keep it boring and consistent.
**Read this before adding a file** — it exists so the layout never drifts again.

## What this project is

Demand forecasting for hospitality on the Recruit Restaurant dataset, comparing tree
models against a TimesFM hybrid. Read [`docs/00-overview.md`](./docs/00-overview.md) first,
then the relevant `docs/` page for the area you're touching.

## Architecture: hexagonal, domain-centric

```
src/forecasting/
  domain/         PURE. No IO, no third-party model libs (no pandas readers, lightgbm,
                  xgboost, timesfm, mlflow, matplotlib). Contains:
      entities.py, validation.py   value objects + temporal splits
      ports/      abc.ABC interfaces — the dependency boundary. The domain OWNS its ports;
                  adapters implement them, application depends on them. (model, data_source,
                  artifact_store, tracker.)
      services/   pure orchestration. A service takes ports + data in and returns results;
                  it never imports an adapter. Subpackages:
                    evaluation/        metrics (RMSLE/MAE/weighted_mae, season)
                    features/          FeatureBuilder + FeatureSelector
                    model_selection/   the 4-step pipeline (ModelSelector + one file per step)
                    predictor.py       recursive multi-step Predictor (the eval contract)
  adapters/       The ONLY layer allowed to do IO / import third-party libs. Each adapter
                  implements a domain port:
                    data/recruit_csv         the Recruit CSVs -> canonical RawData
                    models/                  seasonal_naive, lightgbm_model, xgboost_model, and
                                             timesfm/ (remote HTTP client, zeroshot, hybrid,
                                             features) — TimesFM is NEVER imported here, only
                                             reached over HTTP (see the TimesFM gotcha below)
                    mlflow_tracker, local_artifacts, plotting
  application/    Use cases. THIN — see the pattern below.
  entrypoints/    Composition roots (cli). The ONLY place adapters are wired to use cases.
  config.py       Cross-cutting settings (pydantic-settings). Stays at the package root —
                  adapters import it, so it cannot live under entrypoints.
```

The standalone TimesFM server lives OUTSIDE the package in `space/` (`app.py` + `Dockerfile`) —
a self-contained Hugging Face Space; the package only talks to it over HTTP.

**The rule that matters:** dependencies point inward. `domain` imports nothing from
`adapters` or `entrypoints`. Tempted to import an adapter into the domain? Add a port instead.

**Where things go (decide once, don't re-litigate):**
- An interface several things implement → `domain/ports/`.
- Pure logic that coordinates ports (no IO) → `domain/services/`.
- Anything that reads/writes/renders or imports a model/plot lib → `adapters/`. Plotting is an
  adapter (matplotlib renders = IO); the *numbers* it plots come from `domain/services/evaluation/`.
- Wiring concrete adapters together → `entrypoints/cli.py`. Nowhere else.

Single-implementation adapters are flat files (`mlflow_tracker.py`, `plotting.py`); families
with several (`data/`, `models/`) are folders.

## The use-case pattern (keep application thin)

A use case loads via an adapter and delegates to a domain service. No business logic in
`application/`:

```python
def run(*, source, registry, tracker, settings, ...):
    data = source.load()                                  # adapter (a port)
    return ModelSelector(registry, tracker, settings).run(data, ...)   # domain service
```

If a use case grows loops, feature-building, or scoring logic, that logic belongs in a
`domain/services/` class, not the use case.

## Phasing

Work is phased (see [`roadmap/`](./roadmap)). Stubs carry a `Phase N` marker and raise
`NotImplementedError`. When you implement one:

1. Implement the module.
2. Add/extend its test under `tests/` (mirror the `src/forecasting/` path).
3. Write or update its `docs/` page.
4. Update the phase **status** + a one-line log entry in `roadmap/README.md`.

## Conventions

- **Python ≥ 3.11**, `from __future__ import annotations`, full type hints.
- **Format/lint:** `make fmt` (black + ruff --fix), `make lint`. Line length 100.
- **Tests:** `make test` (pytest). Domain logic must have a unit test. Non-trivial numeric
  code (metrics, features, splits, the recursive predictor) gets a real assertion-based test.
- **Canonical schema:** tidy frames keyed by `store_id`, `date`, target `visitors`
  (see `domain/entities.py`). Adapters normalise raw sources onto this.

## Forecasting-specific gotchas (do not regress these)

- **Recursive multi-step is the eval contract.** A forecast is n days ahead from one cutoff.
  NEVER featurise the whole horizon at once — lags/rolling (and the hybrid's TimesFM input)
  for day +k must come from the model's own predictions of +1..+k-1, never the intra-window
  actuals. The `domain/services/Predictor` owns this loop. Horizon `n` is configurable
  (`config.horizon_days`, `--horizon`).
- **Log-domain safety:** train trees in `log1p` space, invert with `expm1`, and `clip_nonneg`
  predictions **before any metric**. RMSLE on a negative or `log(0)` crashes.
- **No leakage:** every lag/rolling feature is shifted so a row never sees its own target or
  the future; target-based store aggregates use `reference` = the train slice. Validation is by
  date only — never random K-fold on a time series; the final holdout is carved off first.
- **Golden Week:** at least one training fold must contain a prior holiday period; evaluation
  reports normal-days vs holiday-window error separately (reuses the `golden_week` feature).
- **TimesFM, two ways:** (1) `timesfm_zeroshot` — TimesFM standalone, native n-day forecast,
  `recursive = False`. (2) `timesfm_hybrid` — TimesFM **whole-horizon window** signal +
  engineered features → XGBoost. TimesFM forecasts the whole horizon **once at the cutoff**
  (over history ≤ cutoff) via the `Predictor.prepare_window` hook, so it runs **out of the
  recursive loop entirely**; only the engineered lags recurse (each step's prediction feeds
  back into the lags, never into TimesFM). **Training** the hybrid uses the window signal over
  *actual* origins (`training_signal`, memoised in-memory per origin — a Parquet/feature-store
  cache is the scale-up path, not what runs today). Never run TimesFM inside a training/CV fit.
- **TimesFM is remote-only.** The package never imports `timesfm` or `torch` and never
  downloads the checkpoint — it POSTs to a GPU Hugging Face Space (`space/app.py`) through
  `RemoteTimesFMForecaster`, injected from the CLI. Set `FORECAST_TIMESFM_ENDPOINT` to include
  it; unset, the CV runs the tree/baseline models only.

## Commands

`make help` lists everything. Common: `make test`, `make fmt`,
`python -m forecasting find-best-model --metric rmsle --horizon 7`, `docker compose up`.
