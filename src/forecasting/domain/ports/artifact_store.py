"""Port: persisting trained models / forecasts / the TimesFM feature cache."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ArtifactStore(ABC):
    @abstractmethod
    def save(self, name: str, payload: bytes) -> Path: ...

    @abstractmethod
    def load(self, name: str) -> bytes: ...

    @abstractmethod
    def path_for(self, name: str) -> Path: ...
