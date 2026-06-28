# Gstock — Demand Forecasting for Hospitality

Daily-demand forecasting per restaurant (comensales por centro) on the **Recruit
Restaurant Visitor Forecasting** dataset. The project compares classic feature-engineered
tree models against a **TimesFM foundation-model hybrid**, with correct temporal
validation, experiment tracking, plots, and a one-command Docker run.

> Built as a technical test. The goal is to show *how the problem is framed, validated,
> packaged and communicated* — not to chase a leaderboard.

> **Status:** the end-to-end pipeline is implemented and tested — all five models behind one
> port, the 4-step rolling-origin CV + selection, MLflow tracking, comparison plots, the Docker
> one-command run, and the K8s manifests. The TimesFM models only join the run when a Hugging
> Face Space endpoint is configured (`FORECAST_TIMESFM_ENDPOINT`); without it the CV runs the
> tree/baseline models. Still open: the standalone experiment notebook and the embedding/LoRA
> stretch — see [`roadmap/`](./roadmap).

## The idea in one paragraph

Demand per center is a multi-series daily forecasting problem. We train a seasonal-naive
baseline, **LightGBM** and **XGBoost** on engineered features (calendar, holidays, lags,
rolling stats, reservations), and run **TimesFM 2.5** zero-shot. The centerpiece is a
**hybrid**: TimesFM digests trend/seasonality into a per-series signal, and an XGBoost head
fuses that signal with business covariates it can't see. The `xgboost` vs `timesfm_hybrid`
comparison is a clean **ablation that measures the foundation model's marginal value**.

## Meeting the brief

How each requirement is addressed. Status: ✅ implemented & tested · 🟡 partial / scaffolded ·
📋 planned (see [`roadmap/`](./roadmap)).

**Mandatory**

| Requirement | | Where |
|---|:--:|---|
| Python | ✅ | `src/forecasting/` — Python ≥ 3.11, full type hints |
| One-command Docker run, no manual steps | ✅ | [`Dockerfile`](./Dockerfile) + [`docker-compose.yml`](./docker-compose.yml); data bundled, CPU-only image. TimesFM is opt-in via a Hugging Face Space (`FORECAST_TIMESFM_ENDPOINT`) → [Quickstart](#quickstart) |
| Clear README | ✅ | this file + [`docs/`](./docs) |
| ≥ 2 distinct models | ✅ | **5** behind one `Model` port: seasonal-naive, LightGBM, XGBoost, TimesFM zero-shot, TimesFM hybrid → [`docs/03-models.md`](./docs/03-models.md) |
| Temporal validation + appropriate metrics | ✅ | RMSLE (+MAE/weighted_mae); rolling-origin CV + forward holdout, recursive multi-step scoring → [`docs/04-evaluation.md`](./docs/04-evaluation.md) |

**Valued bonuses**

| Bonus | | Where |
|---|:--:|---|
| Justified metric choice | ✅ | RMSLE rationale (count data, log-domain safety) → [`docs/04-evaluation.md`](./docs/04-evaluation.md) |
| Code quality (lint/format/types/tests) | ✅ | `ruff` + `black` + type hints; leakage & metric unit tests in [`tests/`](./tests) (46 green); conventions in [`AGENTS.md`](./AGENTS.md) |
| Multi-center scaling reflection | ✅ | [`docs/06-scaling.md`](./docs/06-scaling.md) |
| K8s manifests | ✅ | [`k8s/train-job.yaml`](./k8s/train-job.yaml) (one-off select) + [`k8s/retrain-cronjob.yaml`](./k8s/retrain-cronjob.yaml) (weekly retrain) + [`k8s/predict-cronjob.yaml`](./k8s/predict-cronjob.yaml) (nightly batch forecast) |
| Experiment tracking (MLflow) | ✅ | `adapters/mlflow_tracker.py` + compose `mlflow` service |
| Comparison plots | ✅ | `adapters/plotting.py` — pred-vs-real, residuals, error-by-horizon, seasonal, importances |

The **headline** beyond the checklist: the `xgboost` vs `timesfm_hybrid` ablation, which
*measures a foundation model's marginal value* on this data — see
[`docs/05-timesfm-hybrid.md`](./docs/05-timesfm-hybrid.md).

## Architecture (hexagonal)

```
domain/         pure logic — no IO, no model libs
  entities, validation           value objects + temporal splits
  ports/                         ABC interfaces (DataSource, Model, ExperimentTracker, ArtifactStore)
  services/                      evaluation (metrics), features (build/select),
                                 model_selection (4-step pipeline), predictor (recursive multi-step)
adapters/       implementations — recruit_csv, lightgbm/xgboost models, timesfm/ (remote HTTP
                client + zeroshot + hybrid), mlflow_tracker, local_artifacts, plotting
application/    thin use cases — find_best_model / evaluate / train / predict
entrypoints/    cli.py composition root: python -m forecasting {find-best-model,evaluate,train,predict}
space/          standalone TimesFM Hugging Face Space (outside the package, called over HTTP)
```

The domain imports no adapters; the CLI is the only place they're wired together. See
[`AGENTS.md`](./AGENTS.md) and [`docs/`](./docs).

## Quickstart

**Prerequisites:** Python ≥ 3.11 (Docker optional). The Recruit CSVs are **already bundled** in
`data/raw/` (committed), so nothing below needs a download.

### 1 · Install

```bash
git clone <repo-url> && cd gstock
python -m venv .venv && source .venv/bin/activate   # recommended
make install                                        # pip install -e ".[dev]"
make test                                           # 46 tests (~15s) — confirms the install
```

### 2 · TimesFM endpoint (optional)

TimesFM is served over HTTP from a Hugging Face Space — never run locally. To include the two
TimesFM models, point the app at the Space:

```bash
cp .env.example .env        # set FORECAST_TIMESFM_ENDPOINT (+ _TOKEN if the Space is private)
```

`.env` is gitignored and auto-loaded by every command (and by `docker compose`). **Without it**,
runs use the tree/baseline models only — or pass `--offline` to skip TimesFM explicitly. Probe
the Space with `make check-endpoint`.

### 3 · Run

```bash
# Smoke test — 1 CV fold, tree/baseline only, no endpoint (~4 min, full 4-step pipeline):
python -m forecasting evaluate --folds 1 --offline

make evaluate     # full temporal CV + comparison report + plots  (the headline)
make train        # refit the selected winner on all data -> artifacts/best_model.pkl
make predict      # batch-forecast the next horizon  -> artifacts/forecasts.csv
```

Every subcommand accepts `--horizon` (days ahead, default 14); `evaluate` also takes `--folds`
and `--metric`. See `python -m forecasting --help`.

### Docker (one command, no setup)

```bash
docker compose up          # temporal CV across models, writes plots + MLflow runs
# MLflow UI -> http://localhost:5000 ; artifacts -> ./artifacts
```

CPU-only image, data baked in, no torch or Kaggle creds at runtime. Reads the same `.env` for
the optional TimesFM endpoint.

## Reproducing the results

```bash
make install
make evaluate              # data is bundled; export FORECAST_TIMESFM_ENDPOINT to include TimesFM
```

Outputs land in `artifacts/`: **`report/index.html`** — a single HTML report bundling every
chart (forecast vs actual split by model family, error by season / prefecture / horizon for all
three metrics, residuals, feature importances) — alongside the individual PNGs, `selection.json`
(chosen winner + horizon + holdout metrics), and `best_model.pkl`. Open `report/index.html`
first. Browse the tracked runs with `mlflow ui --backend-store-uri sqlite:///mlflow.db` (or
`docker compose up` → http://localhost:5000).

## Kubernetes (demo)

```bash
kubectl apply -f k8s/train-job.yaml         # one-off Job: full CV + selection (writes selection.json + best_model.pkl)
kubectl apply -f k8s/retrain-cronjob.yaml   # weekly: refit the selected winner on fresh data
kubectl apply -f k8s/predict-cronjob.yaml   # nightly batch forecast ("the system that just runs it")
```

## Docs

| Doc | What |
|-----|------|
| [docs/00-overview.md](./docs/00-overview.md) | Problem framing & narrative |
| [docs/01-data.md](./docs/01-data.md) | Dataset choice, files, the missing-day policy |
| [docs/02-features.md](./docs/02-features.md) | Feature engineering & leakage rules |
| [docs/03-models.md](./docs/03-models.md) | The 5-model lineup & rationale |
| [docs/04-evaluation.md](./docs/04-evaluation.md) | Metrics, temporal validation, Golden Week |
| [docs/05-timesfm-hybrid.md](./docs/05-timesfm-hybrid.md) | The TimesFM hybrid, remote-serving & precompute design |
| [docs/06-scaling.md](./docs/06-scaling.md) | Scaling to many centers (multi-series, hierarchies) |
| [docs/07-presentation.md](./docs/07-presentation.md) | Working outline for the live technical-test talk |
| [docs/adr/](./docs/adr) | Architecture decision records (hexagonal, dataset, TimesFM, validation) |
| [roadmap/](./roadmap) | Phased plan with up-front time estimates and live status |
