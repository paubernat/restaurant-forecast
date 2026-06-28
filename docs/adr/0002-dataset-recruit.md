# ADR 0002 — Dataset: Recruit Restaurant Visitor Forecasting

**Status:** accepted · **Date:** 2026-06-26

## Context

The brief allows any hospitality dataset. We weighed Hotel Booking Demand, Recruit Restaurant,
and Restaurant Revenue (full comparison in [`../01-data.md`](../01-data.md)).

## Decision

Use **Recruit Restaurant Visitor Forecasting**.

## Rationale

- Native **daily multi-series** — maps to "comensales por centro" and enables the
  scaling-to-many-centers reflection the brief rewards.
- Rich signals for feature engineering: reservations (leading indicator), holidays, genre/area.
- Built-in **RMSLE** benchmark and a defined holdout window.
- Best fit for the **TimesFM** angle (a foundation model needs many real seasonal series).

## Rejected

- **Hotel Booking:** really cancellation *classification*; ~2 daily series after aggregation.
- **Restaurant Revenue:** not a time series (one revenue value per restaurant, 137 rows,
  anonymized features) — temporal validation is impossible.

## Consequences

- No per-product/SKU dimension; we address "units per product" conceptually in
  [`../06-scaling.md`](../06-scaling.md) rather than empirically.
- Two booking systems (air/hpg) require a small join.
