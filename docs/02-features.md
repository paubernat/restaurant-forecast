# 02 — Feature engineering

Implemented in `domain/services/features/builder.py` (pure; operates on the canonical tidy panel
`store_id, date, visitors`). Built once and shared by the tree models and the hybrid's head.

## v1 feature families (what we build first)

| Family | Features | Why |
|---|---|---|
| **Calendar** | day-of-week, is_weekend, month, **day-of-month** (payday proxy), **day-of-year sin/cos** | Restaurant demand is strongly weekly; weekends differ; payday lifts spend; sin/cos give a smooth annual cycle |
| **Holiday** | is_holiday, day-before / day-after a holiday, **Golden Week window flag** | Holidays multiply demand; proximity matters (eve of a holiday ≠ a random Tuesday) |
| **Closed-day** | reindex each store to a daily calendar, `visitors`→0, **`is_closed` flag** | Keeps rolling windows calendar-aligned; the flag lets the model tell a real 0 from a closed day (see [`01-data.md`](./01-data.md)) |
| **Lag** | visitors lag 1, 7, 14, 21, 28, 35 | Autoregressive signal; multiples of 7 capture the weekly cycle |
| **Rolling** | **mean** (and median) over 7, 14, 28, 35-day windows | Local level. Top solutions leaned on **means** (and exponentially-weighted means), *not* std/min/max — those are in possible improvements |
| **Reservation** | reserved-visitors sum, reservation count, mean lead time (air + hpg) | **Leading indicator** — known-in-advance demand from `air_reserve`/`hpg_reserve` |
| **Store** | genre, area split into **prefecture / ward**, per-store & per-(store, dow) mean/median | Cross-sectional level differences in a single global model; cheap priors that also help cold-start stores |

Empirical note on rolling stats: in [dkivaranovic's complete solution](https://github.com/dkivaranovic/kaggledays-recruit)
the visitor features are **lags + window means only** (no std/min/max/median); Max Halford's
8th place adds exponentially-weighted means. So v1 keeps `mean` as the workhorse (+ optional
`median`, robust to the Golden-Week spike) and leaves the rest as opt-in.

## Possible improvements (not in v1)

Deliberately deferred — each is an ablation we can add and measure, kept out of v1 to keep the
first end-to-end CV simple:

- **Location clustering:** KMeans (k≈30) on lat/lon → cluster id, distance-to-center, restaurant
  density. Dropped from v1 (extra moving parts; `store_id` + area already encode location).
- **Weather:** nearest-station temperature + precipitation via the published
  [rrv-weather-data](https://www.kaggle.com/datasets/huntermcgushion/rrv-weather-data) (nearest
  station precomputed). Behind a `WeatherSource` port/adapter; passed into `build_features` as a
  frame. Deferred: it's a **second external download** (breaks single-dataset `docker run`
  simplicity) for marginal ROI — weather mostly moves extreme days.
- **Extra rolling stats:** std / min / max and quantiles, plus exponentially-weighted means.
- **Outlier capping:** clip values beyond ~2.4σ to the max non-outlier before computing rolling
  stats, so one freak day doesn't poison the windows.

## The one rule: no leakage

Every windowed feature is **shifted** so a row never sees its own target or anything in the
future. A lag/rolling computed without a shift silently leaks the label and inflates CV
scores — the most common way to fool yourself on time series. Concretely:

- lags use only strictly past observations;
- rolling stats are computed on the shifted series (`shift(1)` before `.rolling(...)`);
- per-store aggregates used as features are computed on the **training** portion of each
  fold only, never the full series.

## Target transform

The target is modelled in **`log1p` space** (counts, RMSLE objective). Predictions are
`expm1`'d and clipped to `>= 0` before metrics — see
[`04-evaluation.md`](./04-evaluation.md).

## Categorical handling

`store_id`, `genre`, `area` are high-ish cardinality categoricals. LightGBM/XGBoost handle
them natively (LightGBM) or via category codes; we lean on a **single global model** with the
store as a feature rather than one model per store — the scalable pattern discussed in
[`06-scaling.md`](./06-scaling.md).
