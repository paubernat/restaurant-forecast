"""Tiny flushed-to-stderr progress logger for the selection pipeline.

The MLflow runs only land at each step's *end*, so a long CV run otherwise looks frozen for
minutes. This prints one timestamped, flushed line per unit of work (fold prep, each model
scored, each grid config, each holdout fit) — a live, greppable heartbeat: `tail -f` the run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_START = time.perf_counter()
_SINK = None  # optional run log file (set per CV run via set_log_file)


def set_log_file(path) -> None:
    """Also tee progress lines to `path` (creating parent dirs). One file per CV run."""
    global _SINK
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _SINK = open(path, "a", buffering=1)  # line-buffered so `tail -f` is live


def log(msg: str) -> None:
    el = time.perf_counter() - _START
    line = f"[{int(el) // 60:02d}:{int(el) % 60:02d}] {msg}"
    print(line, file=sys.stderr, flush=True)
    if _SINK is not None:
        print(line, file=_SINK, flush=True)
