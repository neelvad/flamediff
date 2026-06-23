"""Adapter contract + a tiny registry. Adapters parse a system-specific checkpoint
format into the normalized ``Checkpoint`` representation."""
from __future__ import annotations

from typing import Protocol

from flamediff.types import Checkpoint


class CheckpointAdapter(Protocol):
    name: str

    def can_load(self, path: str) -> bool:
        """Cheap structural sniff: does this path look like our format?"""
        ...

    def load(self, path: str) -> Checkpoint:
        ...


_REGISTRY: list[CheckpointAdapter] = []


def register(adapter: CheckpointAdapter) -> CheckpointAdapter:
    _REGISTRY.append(adapter)
    return adapter


def get_adapter(path: str) -> CheckpointAdapter:
    for adapter in _REGISTRY:
        if adapter.can_load(path):
            return adapter
    raise ValueError(f"no registered adapter can load {path!r}")
