"""ArtifactStore on the local filesystem.

Holds the deployable model (`best_model.pkl`), the selection record (`selection.json`),
batch forecasts (`forecasts.csv`) and the report (plots + CSVs) under an artifacts/ root
(mounted as a volume in Docker/K8s).
"""

from __future__ import annotations

from pathlib import Path

from ..domain.ports.artifact_store import ArtifactStore


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str) -> Path:
        return self.root / name

    def save(self, name: str, payload: bytes) -> Path:
        path = self.path_for(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def load(self, name: str) -> bytes:
        return self.path_for(name).read_bytes()
