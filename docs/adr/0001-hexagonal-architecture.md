# ADR 0001 — Hexagonal architecture

**Status:** accepted · **Date:** 2026-06-26

## Context

An ML forecasting project that must stay testable, swap data sources/models cleanly, and read
as a code-quality showcase. The risk is the usual ML-project sprawl: notebooks and IO tangled
with modelling logic.

## Decision

Use **hexagonal (ports & adapters)**. Pure `domain` (features, metrics, validation) depends
only on `ports` (Protocols); `adapters` implement them and own all IO + third-party libs;
`application` wires use cases; `cli.py` is the composition root.

## Consequences

- Domain logic is unit-testable without pandas readers / lightgbm / timesfm / mlflow.
- Swapping the CSV source for a warehouse, or adding a model, is one adapter — no domain edits.
- Cost: more files and indirection than a flat script. Mitigated by flattening
  single-implementation adapters and keeping the domain small. We do **not** add a port until
  there's a real boundary (no speculative interfaces).
