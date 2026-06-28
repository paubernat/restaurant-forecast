"""Builds the v1 feature matrix for the demand panel (pure domain — no IO).

`FeatureBuilder(panel, ...).build()` takes a tidy frame (`store_id`, `date`, `visitors`)
plus the optional side tables and returns it enriched with the feature families locked in
docs/02-features.md. Column selection lives next door in `selector.py`.

Leakage rule (the one that matters): every windowed feature is shifted so a row never
sees its own target or the future, and the **target-based store aggregates are computed
from `reference`** (the training panel in CV) — never the row's own future.

Deferred to "possible improvements" (see docs/02-features.md): KMeans location clustering,
weather, rolling std/min/max/quantiles + ewm, outlier capping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...entities import DATE, STORE, TARGET

LAGS = (1, 7, 14, 21, 28, 35)
ROLL_WINDOWS = (7, 14, 28, 35)


def aggregate_reservations(reservations: pd.DataFrame) -> pd.DataFrame:
    """Collapse row-level bookings to one row per (store, date).

    This is the costly part of the reservation feature (a groupby over the *full* row-level
    bookings table — air + hpg, millions of rows). It's identical for every feature build in a
    run, so it MUST be done once up front, not inside the recursive inference loop. The result
    (one row per store/date, carrying `reserve_count`) is what `_add_reservations` merges; it
    detects the pre-aggregated shape and skips re-aggregating.
    """
    r = reservations.copy()
    date_col = DATE if DATE in r.columns else "visit_date"
    r = r.rename(columns={date_col: DATE})
    r[DATE] = pd.to_datetime(r[DATE]).dt.normalize()
    res = (
        r.groupby([STORE, DATE])
        .agg(reserve_visitors=("reserve_visitors", "sum"), reserve_count=("reserve_visitors", "size"))
        .reset_index()
    )
    if "lead_time_days" in r.columns:
        lt = (
            r.groupby([STORE, DATE])["lead_time_days"]
            .mean()
            .reset_index(name="reserve_lead_time_mean")
        )
        res = res.merge(lt, on=[STORE, DATE], how="left")
    return res


class FeatureBuilder:
    """Builds the v1 feature set for the demand panel.

    Construct with the panel and side tables, then call :meth:`build`. The build runs the
    feature families in order — each step appends columns to the working frame:

        reindex -> calendar -> holiday -> lags/rolling -> reservations -> store

    Inputs
    ------
    panel : (store_id, date, visitors)
        The tidy demand panel to enrich. Only these three columns are read.
    reservations : (store_id, date|visit_date, reserve_visitors[, lead_time_days]) | None
        Pre-joined reservation rows, one per booking. Aggregated to (store, date) here.
    stores : (store_id, genre, area) | None
        Static store metadata. `area` is split into prefecture/ward.
    holidays : (date|calendar_date, is_holiday|holiday_flg) | None
        National holiday calendar.
    reference : (store_id, date, visitors[, dow]) | None
        Panel the target-based store aggregates are computed from. Pass the **training**
        rows during CV so the store/(store,dow) means never leak the validation target.
        Defaults to the (already reindexed) panel itself — fine for a single fit on a
        closed train set.

    Final dataframe
    ---------------
    One row per (store_id, date) over each store's full daily calendar. Each row says, for
    one store on one day: how many people came (the thing we predict) plus a set of clues
    the model uses to predict it. The columns, grouped by family:

    Keys & target
        store_id          : which store the row is about.
        date              : which calendar day the row is about.
        visitors          : how many people visited that day — the value we want to predict.
                            On days the store was closed this is set to 0.

    Closed flag
        is_closed         : 1 if the store had no record that day (we treat it as closed and
                            filled visitors with 0), 0 if it was a normal open day. Lets the
                            model tell "real zero demand" apart from "shut, so of course zero".

    Calendar — clues you can read straight off a calendar, no past data needed
        dow               : day of the week as a number, 0 = Monday ... 6 = Sunday.
        is_weekend        : 1 on Saturday/Sunday, 0 otherwise. Restaurants are busier on
                            weekends, so this single flag carries a lot of signal.
        month             : month of the year, 1 = January ... 12 = December (captures
                            seasons — e.g. December parties vs a quiet February).
        day_of_month      : the date number, 1...31 (e.g. pay-day or month-end effects).
        doy_sin, doy_cos  : the position in the year (Jan 1 ... Dec 31) turned into two
                            numbers that trace a circle. We do this so the model sees that
                            Dec 31 and Jan 1 are *next to each other*; a plain "day 365 vs
                            day 1" would look maximally far apart and hide that wrap-around.

    Holiday — clues about public holidays
        is_holiday        : 1 if that day is a national holiday.
        day_before_holiday: 1 if the *next* day is a holiday (the eve — people often go out).
        day_after_holiday : 1 if the *previous* day was a holiday.
        golden_week       : 1 during Japan's late-April-to-early-May holiday cluster, a
                            notably busy and volatile stretch we flag on its own.

    Lag — "what happened N days ago at this same store"
        A lag just copies the visitor count from an earlier day onto today's row, so the
        model can look back. lag_1 = yesterday's visitors, lag_7 = same weekday last week,
        and lag_14 / lag_21 / lag_28 / lag_35 = the same weekday 2, 3, 4, 5 weeks ago.
        Demand is strongly habitual, so "what happened recently / this time last week" is
        usually the single best predictor of today.

    Rolling — "the recent average / typical level at this same store"
        A rolling feature summarises a sliding window of recent days. roll_mean_7 is the
        average visitors over the last 7 days; roll_median_7 is the middle value over those
        days (less swayed by one freak day). We also do 14-, 28- and 35-day windows. The
        window always ends *yesterday* (never includes today), so the row can't peek at the
        answer it's trying to predict. Means smooth out the trend; medians ignore outliers.
            roll_mean_{7,14,28,35}, roll_median_{7,14,28,35}

    Reserve — advance-booking clues for that day (only when a bookings table is supplied)
        reserve_visitors       : total number of guests booked in advance for that day — a
                                 direct hint at demand before the day even starts.
        reserve_count          : how many separate reservations were made for that day.
        reserve_lead_time_mean : on average, how many days ahead those bookings were made
                                 (only present if the bookings data carries lead time).

    Store — what's typical for this particular store
        store_mean / store_median : the store's usual visitor level (average / middle value),
                                    so the model knows a big venue from a small one.
        store_dow_mean / store_dow_median : the store's usual level broken down by weekday
                                    (a store's typical Saturday differs from its Tuesday).
        genre                     : the store's cuisine/type (e.g. izakaya, cafe).
        area_prefecture, area_ward: where the store is, split out of its address.
        (These last three are categorical and only appear when a stores table is supplied.)

    Notes
        - Lag/rolling columns are blank (NaN) for a store's first days because there isn't
          enough history yet to fill the window; callers drop or impute those warm-up rows.
        - Holiday and reserve columns default to 0 when their side table isn't supplied.
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        *,
        reservations: pd.DataFrame | None = None,
        stores: pd.DataFrame | None = None,
        holidays: pd.DataFrame | None = None,
        reference: pd.DataFrame | None = None,
    ) -> None:
        self.panel = panel
        self.reservations = reservations
        self.stores = stores
        self.holidays = holidays
        self.reference = reference

    def build(self) -> pd.DataFrame:
        """Run every feature family and return the enriched frame."""
        df = self._reindex_daily()
        df = self._add_calendar(df)
        df = self._add_holiday(df)
        df = self._add_lags_rolling(df)
        df = self._add_reservations(df)
        ref = df if self.reference is None else self.reference
        df = self._add_store(df, ref)
        return df

    def _reindex_daily(self) -> pd.DataFrame:
        """Reindex each store to a full daily calendar over its active span.

        The raw panel only has rows for days a store served visitors, so a closed day is
        simply a missing row. Lag/rolling windows count *calendar* days, so those gaps
        would silently shorten the windows. We fill each store's date range densely:
        absent days become `visitors=0` with `is_closed=1` (a feature in its own right —
        the model learns closures suppress demand). See docs/01-data.md for the policy.
        """
        panel = self.panel[[STORE, DATE, TARGET]].copy()
        panel[DATE] = pd.to_datetime(panel[DATE])
        # One vectorised pass instead of a Python per-store loop: resample each store to daily
        # freq over its own span. Absent days come back as NaN -> is_closed=1, visitors=0.
        dense = (
            panel.set_index(DATE).groupby(STORE)[TARGET].resample("D").asfreq().reset_index()
        )
        dense["is_closed"] = dense[TARGET].isna().astype("int8")
        dense[TARGET] = dense[TARGET].fillna(0.0)
        return dense[[STORE, DATE, TARGET, "is_closed"]]

    def _add_calendar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pure date-derived seasonality — no target, so no leakage risk.

          dow / is_weekend / month / day_of_month : weekly and monthly seasonality.
          doy_sin, doy_cos : day-of-year encoded on the unit circle so Dec 31 sits next to
              Jan 1 (a raw 1..365 integer would put them maximally far apart).
        """
        d = df[DATE].dt
        df["dow"] = d.dayofweek.astype("int8")
        df["is_weekend"] = (d.dayofweek >= 5).astype("int8")
        df["month"] = d.month.astype("int8")
        df["day_of_month"] = d.day.astype("int8")
        angle = 2.0 * np.pi * d.dayofyear / 365.25  # cyclical: Dec 31 sits next to Jan 1
        df["doy_sin"] = np.sin(angle)
        df["doy_cos"] = np.cos(angle)
        return df

    def _add_holiday(self, df: pd.DataFrame) -> pd.DataFrame:
        """National-holiday context.

          is_holiday : 1 if the date is a public holiday.
          day_before_holiday / day_after_holiday : neighbour-day flags computed on the
              holiday calendar itself (people go out the eve of / day after a holiday).
          golden_week : Japan's late-Apr -> early-May holiday cluster — flagged explicitly
              because it's the holdout's high-volatility window and worth its own signal.

        When no holiday table is passed, the three calendar-driven flags default to 0;
        golden_week is always derivable from month/day so it's computed unconditionally.
        """
        holidays = self.holidays
        if holidays is not None:
            h = holidays.rename(columns={"calendar_date": DATE, "holiday_flg": "is_holiday"})
            h = h[[DATE, "is_holiday"]].copy()
            h[DATE] = pd.to_datetime(h[DATE])
            h = h.sort_values(DATE)
            # "tomorrow is a holiday" / "yesterday was a holiday" on the calendar itself.
            h["day_before_holiday"] = h["is_holiday"].shift(-1).fillna(0)
            h["day_after_holiday"] = h["is_holiday"].shift(1).fillna(0)
            df = df.merge(h, on=DATE, how="left")
        for col in ("is_holiday", "day_before_holiday", "day_after_holiday"):
            df[col] = df.get(col, 0)
            df[col] = df[col].fillna(0).astype("int8")
        # Japan's Golden Week: late Apr -> early May (the holdout's high-volatility window).
        df["golden_week"] = (
            ((df["month"] == 4) & (df["day_of_month"] >= 29))
            | ((df["month"] == 5) & (df["day_of_month"] <= 5))
        ).astype("int8")
        return df

    def _add_lags_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Autoregressive history — the strongest signals for daily demand.

          lag_k : visitors k days ago (same store). LAGS = 1,7,14,21,28,35 capture
              yesterday plus the same weekday 1–5 weeks back.
          roll_mean_w / roll_median_w : rolling stats over the last w days. The series is
              `shift(1)` BEFORE rolling so window w ends *yesterday* — a row never sees its
              own target. Stays within each store.
        """
        df = df.sort_values([STORE, DATE]).reset_index(drop=True)
        grp = df.groupby(STORE)[TARGET]
        for k in LAGS:
            df[f"lag_{k}"] = grp.shift(k)
        # shift(1) so the window ends yesterday, then the C-path groupby rolling (Cython) —
        # far faster than transform(lambda …) which runs a Python call per store per window.
        roll = grp.shift(1).groupby(df[STORE], sort=False)
        for w in ROLL_WINDOWS:
            df[f"roll_mean_{w}"] = roll.rolling(w).mean().reset_index(level=0, drop=True)
            df[f"roll_median_{w}"] = roll.rolling(w).median().reset_index(level=0, drop=True)
        return df

    def _add_reservations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Same-day booking signal, aggregated from per-booking rows to (store, date).

          reserve_visitors : total booked covers for the day.
          reserve_count    : number of reservations for the day.
          reserve_lead_time_mean : mean days between booking and visit (only if the input
              carries `lead_time_days`).

        These are observed at the start of the target day (bookings precede the visit), so
        no shift is needed. Missing (store, date) pairs fill with 0 — no bookings on record.
        """
        reservations = self.reservations
        if reservations is not None and len(reservations):
            # Pre-aggregated (one row per store/date, has reserve_count) -> merge as-is; the
            # expensive groupby was hoisted out of the hot loop. Row-level -> aggregate here.
            res = (
                reservations if "reserve_count" in reservations.columns
                else aggregate_reservations(reservations)
            )
            df = df.merge(res, on=[STORE, DATE], how="left")
        df["reserve_visitors"] = df.get("reserve_visitors", 0.0)
        df["reserve_visitors"] = df["reserve_visitors"].fillna(0.0)
        df["reserve_count"] = df.get("reserve_count", 0)
        df["reserve_count"] = df["reserve_count"].fillna(0).astype(int)
        if "reserve_lead_time_mean" in df.columns:
            df["reserve_lead_time_mean"] = df["reserve_lead_time_mean"].fillna(0.0)
        return df

    def _add_store(self, df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
        """Per-store level and static metadata.

          store_mean / store_median : the store's typical visitor level.
          store_dow_mean / store_dow_median : its typical level per day-of-week (a store's
              Saturday baseline differs from its Tuesday baseline).
          genre / area_prefecture / area_ward : static categoricals from the stores table
              (area is split on whitespace into prefecture + ward).

        The target-based aggregates are computed from `reference`, NOT `df` — pass the
        training panel during CV so these means never average in the validation target.
        """
        ref = reference.copy()
        if "dow" not in ref.columns:
            ref["dow"] = pd.to_datetime(ref[DATE]).dt.dayofweek.astype("int8")
        store_stats = (
            ref.groupby(STORE)[TARGET].agg(store_mean="mean", store_median="median").reset_index()
        )
        df = df.merge(store_stats, on=STORE, how="left")
        sdow = (
            ref.groupby([STORE, "dow"])[TARGET]
            .agg(store_dow_mean="mean", store_dow_median="median")
            .reset_index()
        )
        df = df.merge(sdow, on=[STORE, "dow"], how="left")

        stores = self.stores
        if stores is not None:
            s = stores.copy()
            if "area" in s.columns:
                parts = s["area"].astype(str).str.split(" ", n=2, expand=True)
                s["area_prefecture"] = parts[0]
                s["area_ward"] = parts[1] if parts.shape[1] > 1 else None
            keep = [STORE] + [c for c in ("genre", "area_prefecture", "area_ward") if c in s.columns]
            df = df.merge(s[keep].drop_duplicates(STORE), on=STORE, how="left")
            for c in ("genre", "area_prefecture", "area_ward"):
                if c in df.columns:
                    df[c] = df[c].astype("category")
        return df
