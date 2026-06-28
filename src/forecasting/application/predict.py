"""Predict use case — "the system that just runs it".

Batch-forecast the next `horizon` days for every store from the deployed model. Reads
`selection.json` (model name + params + horizon) to rebuild the architecture via the **same
registry factory** as selection/training, restores the trained weights from `best_model.pkl`,
then rolls the recursive `Predictor` forward from the last observed day. Writes the per-store
forecasts to the artifact store (`forecasts.csv`). This is what the nightly CronJob invokes to
feed auto-ordering — batch, not a live API (see docs/06-scaling.md).
"""

from __future__ import annotations

import json

from ..domain.entities import DATE, STORE
from ..domain.services.predictor import Predictor

__all__ = ["run"]


def run(*, source, registry, artifact_store, settings, horizon=None) -> object:
    """Forecast the next `horizon` days after the last observed date. Returns the forecast frame."""
    selection = json.loads(artifact_store.load("selection.json"))
    name, params = selection["model"], selection["params"]
    horizon = horizon or selection["horizon"]

    spec = {s.name: s for s in registry}[name]
    model = spec.factory(params)
    model.load(artifact_store.path_for("best_model.pkl"))

    data = source.load()
    pred = Predictor(
        model,
        reservations=data.reservations,
        stores=data.stores,
        holidays=data.holidays,
        reference=data.visits,
    ).infer(data.visits, horizon)

    out = pred[[STORE, DATE, "y_pred"]].sort_values([STORE, DATE])
    artifact_store.save("forecasts.csv", out.to_csv(index=False).encode())

    print(f"[predict] {name}: {len(out)} forecasts ({horizon} days) -> forecasts.csv")
    return out
