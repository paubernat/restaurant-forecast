"""TimesFM feature generator — the window signal that feeds the hybrid's tree.

TimesFM forecasts a *block* better than one step at a time, so the hybrid forecasts the
**whole horizon once at the cutoff** (actuals only) and the tree fuses each window step with
its recursively-built lag/rolling features (see docs/05-timesfm-hybrid.md):

  - `window_signal(history, horizon)` — ONE forecast over the whole horizon from `history`'s
    last date, as a long frame `[store_id, step, tfm_point, tfm_q0..tfm_q9]` (step = 1..horizon).
    `TimesFMHybrid.prepare_window` calls it once per forecast origin; `augment` then just indexes
    step = days-ahead. No TimesFM call inside the recursive loop.
  - `training_signal(panel, origins, horizon)` — the training cache: for each origin cutoff `c`
    forecast the window once and emit rows at `date = c + step`, so the head trains on the *same*
    offset-k window feature it sees at inference. Memoised by origin — the window forecast for `c`
    depends only on history ≤ c, so it's identical across CV folds/configs. `origins` is
    subsampled (recent cutoffs) to bound cost: one TimesFM call per origin.

The signal columns are the point forecast plus the 10 quantile-head outputs (`tfm_point`,
`tfm_q0..tfm_q9`). Series are reindexed to a daily grid (closed days -> 0) before inference,
matching the zero-shot adapter. The Transformer never runs inside a tree fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ....domain.entities import DATE, STORE, TARGET

CHECKPOINT = "google/timesfm-2.5-200m-pytorch"


class TimesFMFeatureGenerator:
    def __init__(self, forecaster, *, max_train_cutoffs: int = 60) -> None:
        self.forecaster = forecaster
        self.max_train_cutoffs = max_train_cutoffs
        self._memo: dict[pd.Timestamp, pd.DataFrame] = {}

    def _long_frame(self, stores: list, point: np.ndarray, quant: np.ndarray) -> pd.DataFrame:
        """(point: (n, H), quant: (n, H, Q)) -> long [store_id, step, tfm_point, tfm_q0..q9]."""
        point = np.asarray(point)
        quant = np.asarray(quant)
        frames = []
        for k in range(point.shape[1]):  # k = 0..H-1 -> step k+1
            f = pd.DataFrame({STORE: stores, "step": k + 1, "tfm_point": point[:, k]})
            for qi in range(quant.shape[2]):
                f[f"tfm_q{qi}"] = quant[:, k, qi]
            frames.append(f)
        return pd.concat(frames, ignore_index=True)

    def window_signal(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """One per-store forecast over the whole horizon from `history`'s last date.

        History is reindexed to a daily grid up to the cutoff (closed days = 0); the forecast
        uses only that history (no leakage). Returns [store_id, step, tfm_point, tfm_q0..q9].
        """
        h = history.copy()
        h[DATE] = pd.to_datetime(h[DATE])
        end = h[DATE].max()
        stores, series = [], []
        for store, g in h.groupby(STORE, sort=False):
            g = g[g[DATE] <= end]
            stores.append(store)
            if g.empty:
                series.append(np.zeros(1, dtype=float))
                continue
            idx = pd.date_range(g[DATE].min(), end, freq="D")
            vals = g.set_index(DATE)[TARGET].reindex(idx, fill_value=0.0).to_numpy(dtype=float)
            series.append(vals)
        point, quant = self.forecaster.forecast(series, horizon=horizon)
        return self._long_frame(stores, point, quant)

    def training_signal(self, panel: pd.DataFrame, origins, horizon: int) -> pd.DataFrame:
        """Window signal per origin, emitted at the forecast dates `c + step`.

        Returns long rows [store_id, date, tfm_point, tfm_q0..q9]; a date may appear under
        several origins (different days-ahead) — that's the offset-k training the head needs.
        Memoised by origin (history ≤ c is fold/config-independent).
        """
        panel = panel.copy()
        panel[DATE] = pd.to_datetime(panel[DATE])
        frames = []
        for c in origins:
            c = pd.Timestamp(c)
            if c not in self._memo:
                w = self.window_signal(panel[panel[DATE] <= c], horizon)
                w[DATE] = c + pd.to_timedelta(w["step"], unit="D")
                self._memo[c] = w.drop(columns="step")
            frames.append(self._memo[c])
        return pd.concat(frames, ignore_index=True)
