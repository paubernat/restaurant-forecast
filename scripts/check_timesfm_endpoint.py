"""Dumb end-to-end check for the TimesFM Space.

    python scripts/check_timesfm_endpoint.py https://<user>-<space>.hf.space [HF_TOKEN]

Reads the URL/token from argv, else from FORECAST_TIMESFM_ENDPOINT[_TOKEN]. Hits /health,
POSTs a raw /forecast, then runs the real gstock client (RemoteTimesFMForecaster) so you test
the exact path the CV uses. Prints shapes + timing (the first call is slow: checkpoint download).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import requests


def main() -> None:
    args = sys.argv[1:]
    base = args[0] if args else os.environ.get("FORECAST_TIMESFM_ENDPOINT", "")
    token = args[1] if len(args) > 1 else os.environ.get("FORECAST_TIMESFM_ENDPOINT_TOKEN", "")
    if not base:
        sys.exit("usage: check_timesfm_endpoint.py <space-url> [token]")
    base = base.rstrip("/").removesuffix("/forecast")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    # 1) health — confirms the server is up and whether it sees a GPU
    h = requests.get(f"{base}/health", headers=headers, timeout=60)
    print(f"[health] {h.status_code} {h.text}")
    h.raise_for_status()
    if not h.json().get("cuda"):
        print("  !! cuda=false -> the Space is on CPU hardware, this will be slow")

    # 2) raw POST /forecast — three fake series, a week ahead
    series = [list(np.arange(30.0)), list(np.arange(50.0)), list(np.sin(np.arange(40.0)) * 5 + 20)]
    t = time.time()
    r = requests.post(
        f"{base}/forecast", json={"horizon": 7, "series": series}, headers=headers, timeout=600
    )
    print(f"[forecast] {r.status_code} in {time.time() - t:.1f}s")
    r.raise_for_status()
    d = r.json()
    point = np.asarray(d["point"], dtype=float)
    quant = np.asarray(d["quantiles"], dtype=float)
    print(f"  point {point.shape} (expect (3, 7)), quantiles {quant.shape} (expect (3, 7, 10))")
    print(f"  finite point values: {int(np.isfinite(point).sum())}/{point.size}")
    print(f"  series[0] week-ahead point: {np.round(point[0], 2).tolist()}")
    assert point.shape == (3, 7) and quant.shape == (3, 7, 10), "unexpected output shape"
    assert np.isfinite(point).all(), (
        f"endpoint returned non-finite (NaN/null) forecasts: {point[0].tolist()} — the model on "
        "the Space is broken (check its logs / timesfm+torch versions), not the transport."
    )

    # 3) the real gstock client (the exact path the CV uses)
    sys.path.insert(0, "src")
    from forecasting.adapters.models.timesfm.remote import RemoteTimesFMForecaster

    fc = RemoteTimesFMForecaster(endpoint=f"{base}/forecast", token=token, max_horizon=7)
    p2, q2 = fc.forecast([np.arange(30.0), np.arange(50.0)], horizon=7)
    print(f"[client] RemoteTimesFMForecaster -> point {p2.shape}, quantiles {q2.shape}")
    assert p2.shape == (2, 7) and q2.shape == (2, 7, 10), "client output shape wrong"

    print("\nOK — endpoint works. Set FORECAST_TIMESFM_ENDPOINT and run find-best-model.")


if __name__ == "__main__":
    main()
