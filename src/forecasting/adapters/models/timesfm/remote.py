"""Remote TimesFM forecaster — proxy the forecast over HTTP to a GPU-backed endpoint.

Implements the `forecast(series, horizon) -> (point, quantiles)` forecaster port, so the CV
consumes TimesFM like any other forecaster (see `entrypoints/cli._timesfm_specs`). TimesFM runs
only on a GPU Hugging Face Space (see `space/app.py`) — never locally — collapsing the
~6-min/call CPU cost to seconds.

Because the forecaster is injected, the machine running the CV needs **neither torch nor
timesfm installed** — it only POSTs JSON.

Caching: a TimesFM forecast is a pure function of `(trimmed history, horizon)`, so we cache
**per series** (not per batch) on disk, keyed by content. The same (store, cutoff) series is
forecast across many folds, across steps 1 & 3, by both the zero-shot and hybrid models, and
across reruns — with the cache each unique forecast hits the GPU exactly once, ever. Only the
cache-misses in a batch are POSTed; if the whole batch is cached, no HTTP call happens at all.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import requests

from ....config import settings

_CACHE_VERSION = "v1"  # bump to invalidate every cached forecast (format/model change)


class RemoteTimesFMForecaster:
    def __init__(
        self,
        *,
        endpoint: str = settings.timesfm_endpoint,
        token: str = settings.timesfm_endpoint_token,
        max_context: int = settings.timesfm_max_context,
        max_horizon: int = settings.horizon_days,
        timeout: float = 600.0,
        cache_dir: str = settings.timesfm_cache_dir,
    ) -> None:
        if not endpoint:
            raise ValueError("set FORECAST_TIMESFM_ENDPOINT to use RemoteTimesFMForecaster")
        self.endpoint = endpoint
        self.token = token
        self.max_context = max_context
        self.max_horizon = max_horizon
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def forecast(self, series: list[np.ndarray], horizon: int) -> tuple[np.ndarray, np.ndarray]:
        horizon = int(horizon)
        trimmed = [np.asarray(s, dtype=float)[-self.max_context :] for s in series]
        keys = [self._key(t, horizon) for t in trimmed]
        results: list[tuple[np.ndarray, np.ndarray] | None] = [self._load(k) for k in keys]

        miss = [i for i, r in enumerate(results) if r is None]
        if miss:  # only the uncached series go to the GPU; whole-batch-cached => no HTTP at all
            point, quant = self._remote([trimmed[i] for i in miss], horizon)
            for j, i in enumerate(miss):
                val = (np.asarray(point[j], dtype=float), np.asarray(quant[j], dtype=float))
                self._store(keys[i], val)
                results[i] = val

        return (
            np.stack([r[0] for r in results]),  # type: ignore[index]
            np.stack([r[1] for r in results]),  # type: ignore[index]
        )

    def _key(self, trimmed: np.ndarray, horizon: int) -> str:
        h = hashlib.sha256(f"{_CACHE_VERSION}|{horizon}|{self.max_context}|".encode())
        h.update(np.ascontiguousarray(trimmed, dtype=np.float64).tobytes())
        return h.hexdigest()

    def _load(self, key: str) -> tuple[np.ndarray, np.ndarray] | None:
        if key in self._mem:
            return self._mem[key]
        if self.cache_dir is not None:
            p = self.cache_dir / f"{key}.npz"
            if p.exists():
                with np.load(p) as z:
                    val = (z["point"], z["quant"])
                self._mem[key] = val
                return val
        return None

    def _store(self, key: str, val: tuple[np.ndarray, np.ndarray]) -> None:
        self._mem[key] = val
        if self.cache_dir is not None:
            p = self.cache_dir / f"{key}.npz"
            tmp = self.cache_dir / f"{key}.tmp.npz"  # write-then-rename = atomic, no partial files
            np.savez(tmp, point=val[0], quant=val[1])
            tmp.replace(p)

    # Retry transient endpoint errors (a Space restart/cold-start blips for a minute or two)
    # so one hiccup never tears down an hours-long run. A *paused* Space won't recover on its
    # own, so the bounded backoff (~3.5 min total) eventually gives up rather than hanging.
    _RETRY_WAITS = (10, 30, 60, 120)

    def _remote(self, trimmed: list[np.ndarray], horizon: int) -> tuple[np.ndarray, np.ndarray]:
        payload = {"horizon": horizon, "series": [t.tolist() for t in trimmed]}
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        last_err: Exception | None = None
        for attempt, wait in enumerate((0, *self._RETRY_WAITS)):
            if wait:
                print(f"[timesfm] endpoint error, retry {attempt}/{len(self._RETRY_WAITS)} "
                      f"in {wait}s: {last_err}", file=sys.stderr, flush=True)
                time.sleep(wait)
            try:
                r = requests.post(self.endpoint, json=payload, headers=headers,
                                  timeout=self.timeout)
                r.raise_for_status()
                d = r.json()
                return np.asarray(d["point"], dtype=float), np.asarray(d["quantiles"], dtype=float)
            except requests.exceptions.RequestException as e:
                last_err = e
        raise last_err  # type: ignore[misc]


if __name__ == "__main__":  # self-check: mem cache, disk persistence, partial-batch miss
    import tempfile

    calls = {"n": 0}
    cdir = tempfile.mkdtemp()

    def _make():
        f = RemoteTimesFMForecaster(endpoint="http://stub", token="", cache_dir=cdir)

        def fake(trimmed, horizon):
            calls["n"] += 1
            n = len(trimmed)
            return np.ones((n, horizon)), np.ones((n, horizon, 2))

        f._remote = fake  # type: ignore[method-assign]
        return f

    s = [np.arange(10.0), np.arange(5.0)]
    f1 = _make()
    p1, _ = f1.forecast(s, 3)
    f1.forecast(s, 3)  # in-memory hit
    assert calls["n"] == 1, calls
    f2 = _make()  # fresh instance, same dir
    p3, _ = f2.forecast(s, 3)  # disk hit -> no remote
    assert calls["n"] == 1, calls
    assert np.array_equal(p1, p3)
    f2.forecast([*s, np.arange(7.0)], 3)  # one new series -> one remote call for it only
    assert calls["n"] == 2, calls
    print("timesfm cache self-check OK")
