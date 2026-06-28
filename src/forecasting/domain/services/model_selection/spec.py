"""`ModelSpec` — the registry entry for one candidate model (pure data, no behaviour).

The CLI builds a list of these (factory + default config + grid) and hands it to
`ModelSelector`; nothing here imports a model library, so the domain stays adapter-free.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

Model = object  # structural: anything with fit/predict (+ feature_importance for trees)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[dict], Model]  # params -> fresh model
    default: dict = field(default_factory=dict)  # step-1 config
    grid: list[dict] = field(default_factory=lambda: [{}])  # step-3 grid (contains default)
    is_tree: bool = False  # participates in feature selection / grid
