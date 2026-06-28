# Roadmap

This folder is the **planning artifact**, kept separate from `docs/` (which documents the
*solution*). It records how the work was broken down and sequenced: a per-phase **time
estimate committed up front**, plus a live **status** and notes as each phase lands.

The estimates are frozen — they're the plan as it stood before building. The point is a
visible, honest plan you can hold the delivery against, not a stopwatch.

## Status legend

`todo` · `wip` · `done`

## Phases

| # | Phase | Est (h) | Status | Notes |
|---|-------|:------:|:------:|-------|
| 0 | Scaffold & tooling (structure, pyproject, ruff/black, AGENTS.md, roadmap, docs skeleton, Docker skeleton, git init) | 2 | done | metrics + log-domain safety implemented early |
| 1 | Data ingestion + EDA notebook (bundle Recruit, `recruit_csv`, confirm fit, `01-data.md`) | 3 | done | EDA + `RecruitCsvSource` adapter (AIR/HPG normalised onto canonical schema), 4 tests |
| 2 | Feature engineering (`domain/features.py` + tests, `02-features.md`) | 4 | done | v1 set built; leakage tests green; verified on real data in notebook |
| 3 | Temporal validation + metrics (`validation.py`, `evaluation.py` + tests, `04-evaluation.md`) | 3 | done | rolling-origin CV + carve-first final holdout; metrics already done in Phase 0; `holiday_segments` dropped (reuses `golden_week` feature) |
| 4 | Baseline + tree models (naive, LightGBM, XGBoost adapters) | 3 | done | + `find_best_model` 3-step selection pipeline, `weighted_mae` asymmetric metric, MLflow tracker — pulled fwd from phases 7 & 9 |
| 5 | TimesFM integration (loader, zero-shot adapter, checkpoint baking, `timesfm_features`) | 4 | done | **superseded design** (see ADR 0003 + Phase-10 log): TimesFM moved to a remote GPU HF Space — the local loader and checkpoint-bake were removed; the package is a pure HTTP client (`RemoteTimesFMForecaster`). zero-shot adapter + `timesfm/features` kept |
| 6 | Hybrid model (TimesFM forecast/quantile features → XGBoost; precompute-to-Parquet pipeline) | 5 | done | hybrid in the CV; TimesFM **whole-horizon window** forecast once at the cutoff (out of the recursive loop), training signal memoised **in-memory** per origin (subsampled cutoffs); both TimesFM uses ranked together |
| 7 | Experiment notebook (param selection + 5-model ablation, `03-models.md`, `05-timesfm-hybrid.md`) | 4 | todo | feature selection + hyperparameter grid already done in Phase 4's `find_best_model`; standalone notebook still empty |
| 8 | Use cases + CLI (`train`/`predict`/`evaluate`) + batch predict runner | 3 | done | `find-best-model`/`evaluate`/`train`/`predict` wired in `entrypoints/cli.py`; `--offline` flag skips TimesFM |
| 9 | MLflow tracking adapter | 2 | done | implemented early for `find_best_model` (Phase 4); SQLite backend (file store deprecated in MLflow 3.x) |
| 10 | Comparison plots | 3 | done | step-4 `ComparisonReport` + `adapters/plotting.render_report` (pred-vs-real, error-by-horizon, seasonal, residuals, importances); glued onto `find-best-model` |
| 11 | Dockerization (bake checkpoint, compose end-to-end, no manual steps) | 3 | done | CPU-only image, no checkpoint bake (TimesFM is remote); data bundled; `docker compose up` runs evaluate + MLflow UI |
| 12 | K8s manifests (predict Job + retrain CronJob, kind demo) | 2 | done | train Job (`find-best-model`) + weekly retrain CronJob + nightly predict CronJob |
| 13 | README + docs polish + scaling reflection (`06-scaling.md`) | 3 | wip | docs + comments fact-checked against the code (this pass) |
| 14 | **Stretch:** TimesFM embedding extraction + LoRA fine-tune | 4 | todo | only if time allows; off the critical path |
| 15 | Presentation prep | 2 | todo | |

**Estimated total:** 44 h core (phases 0–13, 15) · 50 h with the stretch (14).

## Log

- _2026-06-28_ — Phases 8 / 11 / 12 closed out + a docs/comments fact-check pass. **TimesFM is
  now remote-only**: the local loader and the Dockerfile checkpoint-bake (`scripts/bake_checkpoint.py`,
  which never existed) were removed — the package POSTs to a GPU HF Space via
  `RemoteTimesFMForecaster`, so the image is CPU-only with no torch/checkpoint (ADR 0003).
  CLI exposes `find-best-model`/`evaluate`/`train`/`predict` (+ `--offline`); K8s has train +
  retrain + predict manifests; `docker compose up` runs end-to-end. Swept the docs and code
  comments for stale facts: `final_horizon_days` 38→39, CV stride 5→9 (~40 folds),
  `comparison.build_report`→`model_selection/step4_report`, "1-day signal"→whole-window,
  "Parquet cache"→in-memory memoise, `timesfm_loader`/`timesfm_server` paths. Fixed the one
  broken test (`test_timesfm_remote` imported a removed module) — full suite 46 green.
- _2026-06-28_ — Phase 10 done + the final-CV redesign. **Hybrid now forecasts the TimesFM
  window once at the cutoff** (over actuals only) and the tree fuses each window step with
  recursively-built lags — TimesFM left the per-step loop (`prepare_window` hook on the
  `Predictor`), so inference is 1 TimesFM call/origin (was ~horizon) and `training_signal` emits
  the same offset-k feature the head sees at inference (memoised per origin). CV reworked for the
  **operational horizon 14** reading the **last year** (`n_folds = cv_window_days // horizon`,
  `--folds` override) with a separate **38-day final holdout** for the headline pred-vs-real.
  Added **step 3.5**: `domain/services/comparison.build_report` → `ComparisonReport` (overall /
  by-season / by-horizon from retained CV preds + a 38-day forecast per model + importances),
  rendered by `adapters/plotting.render_report` and logged as `report-<model>` + `report-summary`
  MLflow runs; `find-best-model` now **persists the winner** (re-fit on all data) as
  `best_model.pkl` + `selection.json` and returns `(winner, report)`. `season()` helper added.
  Tests: window offset + training signal, `prepare_window` once, `season`, comparison, plotting,
  persistence/return shape (37 green).
- _2026-06-26_ — Phase 0 started. Scaffolded the hexagonal repo; implemented the
  evaluation metrics + log-domain safety ahead of schedule (cheap, and it pins the
  RMSLE-on-zeros decision early).
- _2026-06-27_ — Phase 1 done: downloaded the real Recruit data, ran a full EDA
  (`notebooks/01_eda.ipynb`) confirming fit (caught+fixed a rolling-feature index bug), and
  implemented `RecruitCsvSource` — normalises the AIR/HPG CSVs onto the canonical
  `store_id/date/visitors` schema (HPG mapped via `store_id_relation`, reservation lead time
  computed). 4 adapter tests; verified end-to-end through `build_features` on real data
  (296,279 × 38). The notebook now loads via the port, not inline glue.
- _2026-06-28_ — Phase 6 + a big refactor done. **Recursive n-day forecasting is now the eval
  contract**: a new `domain/services/Predictor` forecasts `--horizon` days from one cutoff,
  feeding each prediction back into the next day's lags/rolling (and the hybrid's TimesFM
  input) — fixing the old leaky one-shot window. `find_best_model`'s 3-step logic moved into
  `domain/services/ModelSelector` (the use case is now thin); scoring is against observed valid
  days. Implemented `timesfm_hybrid` (TimesFM 1-day signal + engineered features → XGBoost,
  fully recursive at inference; training signal precomputed+memoised on actual history,
  subsampled cutoffs) and added it + zero-shot to the CV (5 models). `--horizon` flag,
  `horizon_days` default 7. **Architecture reorg** (locked in AGENTS.md): `ports/`→
  `domain/ports/`, new `domain/services/`, `plots/`→`adapters/plotting.py`, `cli`→
  `entrypoints/`. 31 tests green (predictor recursion, native path, hybrid fit/predict).
- _2026-06-28_ — Phase 5 done: TimesFM 2.5 integration. `adapters/timesfm_loader.py`
  loads+compiles the `google/timesfm-2.5-200m-pytorch` checkpoint once (`lru_cache`, lazy
  import, CPU) and exposes a batched `forecast(series, horizon) -> (point, quantiles)`;
  `timesfm_zeroshot` snapshots each store's daily-reindexed history at `fit` and reads the
  forecast by calendar-day offset at `predict`. The forecaster is injected (port-style) so
  the CV runs without TimesFM installed and the adapter unit-tests with a fake (2 tests). The
  long-standing pin blocker is resolved by moving TimesFM to a git-only `[timesfm]` optional
  extra (PyPI caps at 2.0.1; repo needs an LFS-smudge-free clone). `find_best_model` now picks
  up `timesfm_zeroshot` automatically. The `timesfm_features` precompute-to-Parquet stays with
  the hybrid (Phase 6) — the loader is the shared piece. Ran the full rolling-origin CV on real
  data ranked by RMSLE (the Recruit/Kaggle metric).
- _2026-06-27_ — Phase 4 done (reshaped): the 3 model adapters (seasonal-naive on `lag_7`,
  LightGBM + XGBoost in log1p space with native categoricals) **plus** a `find_best_model`
  3-step selection pipeline — (1) CV all models/all features, (2) generous cumulative-gain
  feature selection on the winner, (3) re-CV on selected features over a small grid where each
  model's grid contains its step-1 config. Added a selectable `weighted_mae` asymmetric metric
  (stockout vs waste) and the MLflow tracker (SQLite backend; file store is deprecated in
  MLflow 3.x). Pulled MLflow (9) and feature-selection/grid (part of 7) forward. 12 new tests;
  full suite 26 green. `timesfm[torch]>=2.5` pin is unresolvable on PyPI — deferred to Phase 5.
- _2026-06-27_ — Phase 3 done: `domain/validation.py` temporal splits — `final_holdout`
  (carve the last 39 days off first) + `rolling_origin_splits` (expanding-window, earliest-
  first, date-boundary splits so they're leakage-safe by construction). 3 unit tests (no
  train/valid overlap, expanding train, 2016 Golden Week lands in train). Dropped the
  `holiday_segments` stub — the normal-vs-holiday report reuses the `golden_week` feature.
- _2026-06-27_ — Phase 2 done: `domain/features.py` with the v1 feature set
  (calendar/holiday/closed-day/lag/rolling-mean/reservation/store), leakage-safe shifts +
  train-only aggregates, 5 unit tests green, verified on real data (296,279 × 38;
  `store_dow_mean` corr 0.74). Scope cut vs the research menu: dropped KMeans/weather/
  std-min-max/outlier-capping to "possible improvements" in `02-features.md`.
