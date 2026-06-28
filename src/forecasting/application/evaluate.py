"""Evaluate use case — the headline comparison.

The evaluation *is* the model-selection comparison: rolling-origin temporal CV across all five
models, the per-season / by-horizon breakdowns, the xgboost↔timesfm_hybrid ablation, and the
plots. That lives in `find_best_model` (the `ModelSelector` pipeline; the cross-model report
is built in its step-4 `build_report`), so
`evaluate` is a thin alias — the default Docker / `make evaluate` entrypoint. See
docs/04-evaluation.md.
"""

from __future__ import annotations

from . import find_best_model

__all__ = ["run"]


def run(**kwargs):
    return find_best_model.run(**kwargs)
