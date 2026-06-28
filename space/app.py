"""Standalone TimesFM forecast server for a Hugging Face Docker Space (GPU).

Self-contained on purpose: it does NOT import the gstock `forecasting` package, so the Space
only needs this one file + the Dockerfile. The compile flags (`ForecastConfig` below) live
ONLY here — the gstock package never loads TimesFM, it just POSTs to this server via
`RemoteTimesFMForecaster` (see docs/05-timesfm-hybrid.md).

    POST /forecast  {"horizon": int, "series": [[float, ...], ...]}
      -> {"point": [[...]], "quantiles": [[[...]]]}      # shapes (n, h) and (n, h, 10)
    GET  /health    -> {"status": "ok", "cuda": <bool>}  # confirms the GPU is seen
"""

from __future__ import annotations

import os
from collections import defaultdict
from functools import lru_cache

import numpy as np
import timesfm
import torch
from fastapi import FastAPI
from pydantic import BaseModel

CHECKPOINT = os.environ.get("TIMESFM_CHECKPOINT", "google/timesfm-2.5-200m-pytorch")
# >= largest run horizon (CV 14, holdout 39)
MAX_HORIZON = int(os.environ.get("TIMESFM_MAX_HORIZON", "64"))
# Context buckets (multiples of the 32-pt patch). A series is fed to the LARGEST bucket that
# fits its real history, most-recent points only — no padding. TimesFM NaNs when an input is
# shorter than its compiled context (short input -> internal pad+mask -> NaN, issue #321), so we
# size the model to the data, not the data to the model. Granularity is dense in the HIGH range
# so a ~1-year history isn't truncated below the annual season (384/448 = ~12.5/14.7 months);
# it's coarse in the low range (short stores carry no seasonality to keep). Buckets compile
# LAZILY, so a bucket no series reaches (e.g. 512 — this dataset tops out ~478 days) never
# compiles and costs nothing. Override w/ env.
BUCKETS = tuple(
    sorted(int(x) for x in os.environ.get("TIMESFM_BUCKETS", "64,128,256,384,448,512").split(","))
)


@lru_cache(maxsize=len(BUCKETS))
def _model(max_context: int):
    """One TimesFM compiled for `max_context`. Cached per bucket (each compiles on first use)."""
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(CHECKPOINT)
    model.compile(
        timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=MAX_HORIZON,
            normalize_inputs=False,  # the model's internal per-series revin still runs
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    return model


class ForecastRequest(BaseModel):
    horizon: int
    series: list[list[float]]


app = FastAPI(title="TimesFM forecast")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "cuda": torch.cuda.is_available()}


def _bucket_for(length: int) -> int:
    """Largest bucket that fits `length` real points; the smallest bucket if shorter than all."""
    fits = [b for b in BUCKETS if b <= length]
    return max(fits) if fits else BUCKETS[0]


def _prep(s: list[float], bucket: int) -> np.ndarray:
    """The most-recent `bucket` real points. Pads (with the edge value, never 0 — a 0 means a
    closed day here and would bias the forecast down) only when the series is shorter than the
    smallest bucket: the rare thin-history store, which has little signal either way."""
    a = np.asarray(s, dtype=np.float32)[-bucket:]
    if a.size == 0:
        a = np.zeros(1, dtype=np.float32)
    if a.size < bucket:
        a = np.concatenate([np.full(bucket - a.size, a[0], dtype=np.float32), a])
    return a


@app.post("/forecast")
def forecast(req: ForecastRequest) -> dict:
    # Route each series to the largest bucket that fits its real history (real points only, no
    # padding), then run one batched forecast per bucket (each bucket = its own compiled model).
    point: list = [None] * len(req.series)
    quant: list = [None] * len(req.series)
    groups: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(req.series):
        groups[_bucket_for(len(s))].append(i)
    for bucket, idxs in groups.items():
        p, q = _model(bucket).forecast(
            horizon=req.horizon, inputs=[_prep(req.series[i], bucket) for i in idxs]
        )
        p, q = np.asarray(p), np.asarray(q)
        for j, i in enumerate(idxs):
            point[i], quant[i] = p[j].tolist(), q[j].tolist()
    return {"point": point, "quantiles": quant}
