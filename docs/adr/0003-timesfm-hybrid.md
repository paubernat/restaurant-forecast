# ADR 0003 — TimesFM hybrid as the centerpiece

**Status:** accepted · **Date:** 2026-06-26

## Context

We want a model comparison that's more informative than "RF vs XGBoost". TimesFM 2.5 is a
pretrained time-series foundation model that pairs naturally with the multi-series Recruit
data. Full design in [`../05-timesfm-hybrid.md`](../05-timesfm-hybrid.md).

## Decision

Build a **hybrid**: TimesFM produces a per-series signal (v1 = `forecast()` point + quantiles)
that an **XGBoost** head fuses with business covariates. Ship it alongside a TimesFM
zero-shot reference and the standalone tree models, so `xgboost` vs `timesfm_hybrid` is a clean
**ablation measuring TimesFM's marginal value**.

## Key choices

- **v1 uses the supported forecast/quantile output**, not raw embeddings. Embedding extraction
  (forward hooks) and LoRA fine-tuning are stretch goals — undocumented/fragile, off the
  critical path.
- **Forecast the TimesFM window once per origin and cache it**: in-memory per origin within a
  run, plus a durable **content-addressed per-series disk cache** in the client
  (`.cache/timesfm`) reused across folds/steps/models/reruns (a shared feature store is the
  scale-up path). Never run the Transformer inside the tree/CV loop or per horizon step
  (otherwise CPU inference cost explodes across cutoffs × stores).
- **Serve TimesFM from a dedicated GPU Hugging Face Space** (`space/`); the pipeline calls it
  over HTTP and never bundles torch or the checkpoint. *(Superseded the original choice to bake
  the checkpoint into the image: TimesFM 2.5 + torch ballooned the image to ~2–3 GB, and CPU
  inference was ~6 min/call.)*

## Consequences

- A modern, defensible narrative for the presentation.
- The application image stays lean — no torch, no checkpoint; TimesFM compute lives in the
  Space. Trade-off: runs now need the TimesFM endpoint reachable, not a self-contained
  offline image.
- Rolling-inference throughput is the main risk; the once-per-origin window forecast + memoise
  (and the GPU Space) is the mitigation.
