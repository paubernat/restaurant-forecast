"""`Predictor` — recursive multi-step forecasting (pure domain service).

The eval contract: a forecast is **n days ahead from one cutoff**. We never featurise the
whole horizon at once (that leaks the intra-window actuals into lags/rolling). Instead, for
day +k we build features over actual history **plus the model's own predictions** for
+1..+k-1, predict, and feed that prediction back before moving to +k+1. So `lag_*`, rolling
means — and, for the hybrid, TimesFM's input — only ever see the past or the model's prior
guesses. See AGENTS.md ("recursive multi-step is the eval contract").

Three optional, duck-typed hooks on a model (defaults keep trees/naive working unchanged):
  - `recursive: bool` (default True). A model that natively forecasts the whole block in one
    call (TimesFM zero-shot) sets `recursive = False` — we then build the n target rows and
    call `predict` once.
  - `prepare_window(history, horizon)` (default: no-op). Called once before the recursive loop
    with the *full* history. The hybrid uses it to run its single cutoff-origin TimesFM window
    forecast up front (TimesFM out of the loop), so `augment` can just index it per step.
  - `augment(features, history) -> features` (default: identity). The hybrid uses it to attach
    the window signal for the current days-ahead step (no TimesFM call). The lags still recurse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..entities import DATE, STORE, TARGET
from ..ports.model import Model
from .features import LAGS, ROLL_WINDOWS, FeatureBuilder

# Enough trailing *actual* history for every lag/rolling window of any horizon step: deeper
# steps lean on the predicted days (always kept), so this bound is on the actuals side only.
_KEEP_DAYS = max(LAGS) + max(ROLL_WINDOWS) + 10


class Predictor:
    def __init__(
        self,
        model: Model,
        *,
        reservations: pd.DataFrame | None = None,
        stores: pd.DataFrame | None = None,
        holidays: pd.DataFrame | None = None,
        reference: pd.DataFrame | None = None,
    ) -> None:
        self.model = model
        self.reservations = reservations
        self.stores = stores
        self.holidays = holidays
        self.reference = reference

    def infer(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Forecast `horizon` consecutive days after `history`'s last date.

        `history` is a tidy panel (store_id, date, visitors) of observed days up to the
        cutoff. Returns a frame [store_id, date, y_pred] for cutoff+1..cutoff+horizon.
        """
        history = history[[STORE, DATE, TARGET]].copy()
        history[DATE] = pd.to_datetime(history[DATE])
        cutoff = history[DATE].max()
        stores = history[STORE].unique()
        target_dates = [cutoff + pd.Timedelta(days=k) for k in range(1, horizon + 1)]

        if not getattr(self.model, "recursive", True):
            rows = pd.DataFrame(
                [(s, d) for d in target_dates for s in stores], columns=[STORE, DATE]
            )
            pred = self.model.predict(rows)
            return pred[[STORE, DATE, "y_pred"]].reset_index(drop=True)

        # Recursive path. The naive rewrite rebuilt the whole feature matrix once per horizon
        # step (then threw away every row but the target day's). Here we featurise the
        # *prediction-independent* columns (calendar/holiday/reservations/store) ONCE over
        # [trailing actuals + all H target placeholders], and inside the loop recompute only the
        # autoregressive lags/rolling — the only features that consume the fed-back predictions.
        # Leakage-safe: a target row's lags/rolling read only dates <= its own day - 1, so having
        # the later placeholders present (still 0 until their step) never touches an earlier row.
        prepare = getattr(self.model, "prepare_window", None)
        if prepare is not None:
            prepare(history, horizon)
        work = history[history[DATE] >= cutoff - pd.Timedelta(days=_KEEP_DAYS)].copy()
        # Placeholders (visitors=0.0, not NaN) keep is_closed=0 on the days we forecast.
        placeholders = pd.DataFrame(
            [(s, d) for d in target_dates for s in stores], columns=[STORE, DATE]
        )
        placeholders[TARGET] = 0.0
        fb = FeatureBuilder(
            pd.concat([work, placeholders], ignore_index=True),
            reservations=self.reservations,
            stores=self.stores,
            holidays=self.holidays,
            reference=self.reference,
        )
        dense = fb._add_holiday(fb._add_calendar(fb._reindex_daily()))
        static = fb._add_reservations(dense.copy())
        static = fb._add_store(static, static if self.reference is None else self.reference)

        # Lags/rolling for the target rows ONLY: pivot the dense target to a (day x store) matrix
        # and slice the needed offsets per step. Each store's dense series is daily-contiguous, so
        # row p-k is its lag_k and rows [p-w, p-1] are its w-day window; a NaN in that window (a
        # store with too little history) propagates to NaN, matching rolling(min_periods=w). The
        # only per-step cost is a few numpy slices + the model's predict — no full re-featurise.
        piv = dense.pivot(index=DATE, columns=STORE, values=TARGET).sort_index()
        piv = piv.reindex(pd.date_range(piv.index.min(), piv.index.max(), freq="D"))
        mat = piv.to_numpy(dtype=float)
        store_axis = piv.columns.to_numpy()
        row_of = {d: i for i, d in enumerate(piv.index)}

        augment = getattr(self.model, "augment", None)
        out = []
        for target_date in target_dates:
            p = row_of[target_date]
            cols = {f"lag_{k}": mat[p - k] for k in LAGS}
            for w in ROLL_WINDOWS:
                win = mat[p - w : p]  # rows [p-w, p-1] = the w days ending yesterday
                cols[f"roll_mean_{w}"] = win.mean(axis=0)
                cols[f"roll_median_{w}"] = np.median(win, axis=0)
            lr = pd.DataFrame({STORE: store_axis, DATE: target_date, **cols})
            target = static[static[DATE] == target_date].merge(lr, on=[STORE, DATE], how="left")
            if augment is not None:
                target = augment(target, work)  # hybrid indexes its own window; ignores arg
            pred = self.model.predict(target)
            step = pred[[STORE, DATE]].copy()
            step["y_pred"] = pred["y_pred"].to_numpy()
            out.append(step)
            # Feed the prediction back so the next step's lags/rolling see it as observed.
            mat[p] = pd.Series(store_axis).map(step.set_index(STORE)["y_pred"]).to_numpy()
        return pd.concat(out, ignore_index=True)
