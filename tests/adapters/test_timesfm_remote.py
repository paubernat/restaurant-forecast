"""Wire contract for the remote TimesFM forecaster (client side).

TimesFM itself is served by a standalone Hugging Face Space (`space/app.py`, not importable
here — it pulls torch + timesfm), so this test exercises the part that lives in the package:
`RemoteTimesFMForecaster`. It monkeypatches `requests.post` with a fake that mimics the Space's
JSON response, and proves the client trims each series to `max_context` and parses the
`point` / `quantiles` shapes the CV relies on.
"""

import numpy as np

from forecasting.adapters.models.timesfm import remote as timesfm_remote
from forecasting.adapters.models.timesfm.remote import RemoteTimesFMForecaster


def test_remote_roundtrip_and_context_trim(monkeypatch):
    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            # Echo a 1..horizon ramp per series as the point forecast; zeros for 10 quantiles —
            # the same {"point", "quantiles"} schema space/app.py returns.
            n = len(captured["json"]["series"])
            horizon = captured["json"]["horizon"]
            point = np.tile(np.arange(1, horizon + 1, dtype=float), (n, 1))
            quant = np.zeros((n, horizon, 10))
            return {"point": point.tolist(), "quantiles": quant.tolist()}

    def fake_post(url, json, headers, timeout):
        captured["json"] = json
        return _FakeResponse()

    monkeypatch.setattr(timesfm_remote.requests, "post", fake_post)

    fc = RemoteTimesFMForecaster(endpoint="http://test/forecast", max_context=2, max_horizon=7)
    point, quant = fc.forecast([np.arange(5.0), np.array([4.0, 5.0])], horizon=3)

    assert point.shape == (2, 3)
    assert quant.shape == (2, 3, 10)
    assert point[0].tolist() == [1.0, 2.0, 3.0]
    # client trimmed the 5-long series to max_context=2 before sending
    assert captured["json"]["series"][0] == [3.0, 4.0]
