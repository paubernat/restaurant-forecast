# 06 — Scaling to many centers

How this goes from ~829 restaurants in a test to Gstock's real fleet of centers, each with
many products. (The brief explicitly asks for this reflection.)

## 1. Global model, not one-per-series

We already train a **single global model** with `store_id`/`genre`/`area` as features rather
than one model per restaurant. This is the scalable choice:

- **Cross-learning:** a new or low-history center borrows patterns from similar ones (genre,
  area) instead of starting cold.
- **Ops:** one model to train, deploy, monitor — not N.
- **TimesFM amplifies this:** zero-shot it forecasts a brand-new series with no training at
  all, which is exactly the cold-start case (new center, new product) that hurts most.

## 2. The hierarchy: center × product

Gstock forecasts units **per product per center**, so the real series count is
`centers × SKUs` — easily millions. Two structural tools:

- **Hierarchical / grouped series:** total → center → category → SKU. Forecast at levels and
  **reconcile** (e.g. MinT) so child forecasts sum to parents — orders must be coherent.
- **Sparse/intermittent demand:** many SKUs sell 0 most days. Tree models with the right
  features and `log1p` handle this; specialised methods (Croston, zero-inflated) for the
  sparsest tails. The missing-day/closed-day handling from [`01-data.md`](./01-data.md)
  generalises directly.

## 3. Feature & compute scaling

- The **TimesFM signal cache** ([`05`](./05-timesfm-hybrid.md)) is the scaling linchpin. Today
  forecasts are already content-addressed and cached **per series on disk** (`.cache/timesfm`),
  reused across folds, steps, both TimesFM models and reruns; at scale that disk cache graduates
  to a shared **feature store** keyed by `(series, cutoff, horizon)` so a precompute batch job's
  signals are reused fleet-wide. Inference parallelises trivially across series.
- Feature pipelines move to a columnar store / feature store keyed by `(center, sku, date)`.
- Heavy backfills become Spark/Ray jobs; the hexagonal `DataSource` port means swapping the
  CSV adapter for a warehouse adapter touches one file.

## 4. Productionisation

- **Batch is the right shape** (already built): a nightly CronJob loads the latest model +
  recent data and writes per-center/-product forecasts that feed auto-ordering. No live API —
  ordering is a scheduled decision, not a request/response one.
- **Per-center/per-segment models** can coexist with the global model where a large center
  earns its own; the `Model` port makes this a routing concern, not a rewrite.
- **Monitoring:** track RMSLE drift per center, and alert on the holiday-window segment
  specifically — that's where bad forecasts cost the most in over/under-ordering.

## 5. What I'd add next

Probabilistic outputs (TimesFM already gives quantiles) → order to a service level, not a
point estimate; promotion/price covariates; weather; and hierarchical reconciliation across
the center→SKU tree.
