"""Tiny flushed-to-stderr progress logger for the selection pipeline.

The MLflow runs only land at each step's *end*, so a long CV run otherwise looks frozen for
minutes. This prints one timestamped, flushed line per unit of work (fold prep, each model
scored, each grid config, each holdout fit) — a live, greppable heartbeat: `tail -f` the run.
"""

from __future__ import annotations

import sys
import time

_START = time.perf_counter()


def log(msg: str) -> None:
    el = time.perf_counter() - _START
    print(f"[{int(el) // 60:02d}:{int(el) % 60:02d}] {msg}", file=sys.stderr, flush=True)
