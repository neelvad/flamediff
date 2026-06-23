"""flamediff: structural drift diffing for model checkpoints."""
from __future__ import annotations

from flamediff.adapters import torchrec_mch  # noqa: F401  (registers the adapter)
from flamediff.adapters.base import get_adapter
from flamediff.diff import diff_checkpoints, diff_dense, diff_table
from flamediff.types import (
    Checkpoint,
    CheckpointDiff,
    DenseTensorDiff,
    EmbeddingTableDiff,
)


def load_checkpoint(path: str) -> Checkpoint:
    """Load a checkpoint, selecting an adapter from the registry by sniffing the path."""
    return get_adapter(path).load(path)


__all__ = [
    "load_checkpoint",
    "diff_checkpoints",
    "diff_table",
    "diff_dense",
    "Checkpoint",
    "CheckpointDiff",
    "EmbeddingTableDiff",
    "DenseTensorDiff",
]
