# 01 — Data

## Why Recruit (and not the others)

The brief suggested several datasets. We compared the three restaurant/hotel ones:

| Dataset | For | Against | Verdict |
|---|---|---|---|
| **Hotel Booking Demand** | On-theme (hotelería); rich booking attributes; cancellation angle | Natively a **classification** problem (cancellation); only ~2 daily series after aggregation; weak multi-center story; no per-product | Viable but thin |
| **Recruit Restaurant** | Native **daily multi-series**; reservations + holidays + genre/area; built-in RMSLE benchmark; *the* dataset for the scaling reflection; maps to "comensales por centro" | Visitors only (no SKU/units); Japanese context; two booking systems to join | **Chosen** |
| **Restaurant Revenue** | Quick, tiny | **Not a time series** — one revenue number per restaurant; 137 rows; anonymized features; temporal validation impossible | Ruled out |

Recruit also pairs best with the **TimesFM** angle: a foundation model for time series needs
many real seasonal series, which Recruit has and the others don't (see
[`05-timesfm-hybrid.md`](./05-timesfm-hybrid.md)).

## Structure of the dataset

Recruit Holdings ran **two** systems, so the original Kaggle dataset ships as **8 CSVs**, not
one. That split is the only reason for the file count:

- **AIR** = *AirREGI*, the point-of-sale / reservation app. This is where the **actual
  visitor counts** come from — our ground truth.
- **HPG** = *HotPepper Gourmet*, the online reservation site. Extra reservation signal for
  stores that also list there.

The two systems use different store IDs, hence a dedicated **relation** table to join them,
plus a shared **calendar**. The 8 files group into 5 logical tables:

| Original file(s) | What it holds | → canonical table |
|---|---|---|
| `air_visit_data.csv` | `air_store_id`, `visit_date`, **`visitors`** | **`visits`** — the target series |
| `air_reserve.csv`, `hpg_reserve.csv` | reservation rows with `reserve_datetime` + `visit_datetime` + party size | `reservations` — leading indicator |
| `air_store_info.csv` | genre, area name, lat/lon | `stores` — static features |
| `store_id_relation.csv` | the AIR↔HPG id map | used to remap HPG reservations onto AIR store ids |
| `date_info.csv` | `calendar_date`, `day_of_week`, **`holiday_flg`** | `holidays` — calendar features |
| `hpg_store_info.csv` | HPG-side store metadata | present in the dataset, **not read** by the adapter (we forecast AIR stores) |
| `sample_submission.csv` | the (store, date) rows to predict | defines the official holdout; not read as input |

The `recruit_csv` adapter reads the six files it needs (`_REQUIRED` in
`adapters/data/recruit_csv.py`), does the joins once, and normalises everything onto the
canonical schema in `domain/entities.py` (`store_id`, `date`, `visitors`), so the rest of the
codebase never sees the AIR/HPG split.

**Shape:** ~829 AIR stores, daily from **Jan 2016 to Apr 2017**. The official holdout is the
~39 days **2017-04-23 → 2017-05-31**, which contains Japan's **Golden Week** — the single
biggest demand spike of the year, and why evaluation is stratified around it (see
[`04-evaluation.md`](./04-evaluation.md)).

## Why this structure fits the task

Gstock forecasts **demand per center** to drive automatic supplier orders. Mapping that onto
Recruit:

- **Visitors per store per day = comensales per center per day.** Same shape, same daily
  cadence, same business question ("how many people will we serve, per site, so we can
  order").
- **Multi-series, not one series.** ~829 stores means a real *global* forecasting problem —
  the scaling-to-many-centers reflection the brief rewards ([`06-scaling.md`](./06-scaling.md))
  is grounded in real data, not hand-waved.
- **The covariates are the ones that actually move restaurant demand:** reservations (a true
  leading indicator), holidays, day-of-week, and store genre/area. That's enough to build a
  credible feature pipeline ([`02-features.md`](./02-features.md)) rather than toy lags.
- **A defined metric and holdout** (RMSLE, last 39 days) give an honest, externally-comparable
  yardstick instead of a number we invented.
- **Many real seasonal series** is exactly what the **TimesFM** foundation-model angle needs
  ([`05-timesfm-hybrid.md`](./05-timesfm-hybrid.md)).

The one thing Recruit lacks is a **per-product/SKU** dimension (it's visitors, not units per
dish). We address that as a scaling argument in [`06-scaling.md`](./06-scaling.md) rather than
pretending the data has it.

## The missing-day policy (important)

In `air_visit_data`, **closed days are absent rows, not zeros.** A restaurant shut on a
Monday simply has no row for that Monday. This is a real modelling decision, not a detail:

- **Leave the gap:** lags/rolling windows operate on observed days only. Simpler, but
  "7 days ago" may not be a calendar week ago.
- **Impute 0 for closed days:** reindex to a full daily calendar and fill closed days with
  0. Makes windows calendar-aligned, but injects zeros that the RMSLE/log1p path must handle
  (it does — see the log-domain safety note) and risks teaching the model spurious zeros.

**Confirmed in EDA** (`notebooks/01_eda.ipynb`): per-store coverage averages **0.85** (median
0.86), i.e. **44,171** closed day-rows are absent across stores' active spans — a real gap, not
a rounding detail. **Decision:** reindex each store to its active date range on a full daily
calendar, mark imputed days with an `is_closed` flag, and keep the target as observed (no fake
zeros) so lags/rolling are calendar-aligned without inventing demand. The flag becomes a
feature. (Top Kaggle solutions instead fill `0` for closed days; we keep that as a Phase-7
A/B since it changes what the rolling windows mean.)

## Bundling for Docker

The brief requires `docker run` with no manual steps, so the full CSVs are **bundled into
the image** at build time (`COPY data ./data` in the Dockerfile, reading the local
`data/raw/`) — no Kaggle credentials needed at runtime. The CSVs sit in `data/raw/` on disk;
`.gitignore` excludes `data/raw/*.csv` (the raw set is large — `hpg_reserve.csv` alone is
~120 MB), so they're packaged via the Docker build context rather than committed to git. Run
`make data` (Kaggle CLI) to repopulate `data/raw/` on a fresh checkout.
