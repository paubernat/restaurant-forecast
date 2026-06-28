# 00 — Overview

## The problem

Gstock trains per-center demand models (daily sales, comensales, units per product) that
feed automatic supplier orders. This project is a small-scale stand-in for that: **forecast
daily demand per center** and show the full path — data → features → models → temporal
evaluation → packaging → communication.

We deliberately optimise for *reasoning and rigour*, not leaderboard score.

## Use case & dataset

**Recruit Restaurant Visitor Forecasting** — daily visitor counts per restaurant in Japan.
It maps almost 1:1 onto "comensales por centro", is natively **multi-series** (hundreds of
stores), and carries the signals that make feature engineering interesting (reservations,
holidays, genre/area). It also enables the **scaling-to-many-centers** reflection the brief
asks for. Full rationale and the rejected alternatives are in
[`01-data.md`](./01-data.md).

## Approach

A ladder of models, from honest baseline to foundation-model hybrid:

1. **seasonal-naive** — the anchor everything must beat.
2. **LightGBM** — engineered features, trained.
3. **XGBoost** — engineered features, trained; also the hybrid's base learner.
4. **TimesFM 2.5 zero-shot** — a pretrained time-series foundation model, no training.
5. **TimesFM hybrid** — TimesFM's per-series signal + business covariates → XGBoost.

The story is **trained-with-domain-features vs zero-shot foundation model**, resolved by a
hybrid that takes the best of both. The `xgboost` vs `timesfm_hybrid` comparison is a clean
ablation that **measures TimesFM's marginal value**. Details in
[`03-models.md`](./03-models.md) and [`05-timesfm-hybrid.md`](./05-timesfm-hybrid.md).

## Evaluation

Primary metric **RMSLE** (matches the dataset's benchmark and count data), with MAE (readable)
and **weighted_mae** (asymmetric stockout-vs-waste cost) alongside, reported overall, **by
horizon**, and **stratified normal vs holiday windows**. Validation is **rolling-origin temporal CV** plus a forward holdout
mirroring the competition's last ~39 days. See [`04-evaluation.md`](./04-evaluation.md).

## Engineering

- **Hexagonal** architecture (see [`../AGENTS.md`](../AGENTS.md)).
- **Docker / compose** one-command run; TimesFM served by a Hugging Face Space, called over HTTP.
- **MLflow** experiment tracking; **matplotlib** comparison plots.
- **K8s** Job (train) + CronJob (nightly batch forecast — "the system that just runs it").
- **ruff + black + type hints + pytest**.

## Planning

The work is broken into phases with a time **estimate committed up front** and a live status
as each lands — see [`../roadmap/`](../roadmap). For how each brief requirement is covered, see
[Meeting the brief](../README.md#meeting-the-brief) in the README.
