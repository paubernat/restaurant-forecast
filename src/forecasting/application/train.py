"""Train (retrain) use case — refit the *already-selected* winner on fresh data.

`find-best-model` does the expensive part (CV selection) and writes `selection.json`
(model name + params + selected features + horizon) plus `best_model.pkl`. This use case is
the cheap, repeatable counterpart: read that selection, rebuild the same architecture via the
**same registry factory** the selector uses (so the hybrid's TimesFM generator is wired
automatically), refit it on **all** currently-available data, and re-save `best_model.pkl`.

That's its whole value over `find-best-model`: skip the CV, just keep the deployed model
current as new days arrive. It's what the retrain CronJob runs.
"""

from __future__ import annotations

import json

from ..domain.services.features import FeatureBuilder
from ..domain.services.features import subset_features

__all__ = ["run"]


def run(*, source, registry, artifact_store, tracker=None, settings) -> object:
    """Rebuild the selected winner, refit on all data, persist. Returns the fitted model."""
    selection = json.loads(artifact_store.load("selection.json"))
    name, params = selection["model"], selection["params"]
    selected = selection["selected_features"]

    data = source.load()
    feat = FeatureBuilder(
        data.visits,
        reservations=data.reservations,
        stores=data.stores,
        holidays=data.holidays,
        reference=data.visits,
    ).build()

    spec = {s.name: s for s in registry}[name]
    model = spec.factory(params)
    model.fit(subset_features(feat, selected if spec.is_tree else None))
    model.save(artifact_store.path_for("best_model.pkl"))

    if tracker is not None:
        tracker.start_run(f"train-{name}")
        tracker.log_params({"model": name, "retrained_on_rows": str(len(data.visits)), **{
            f"param_{k}": str(v) for k, v in params.items()
        }})
        tracker.end_run()

    print(f"[train] retrained {name} on {len(data.visits)} rows -> best_model.pkl")
    return model
