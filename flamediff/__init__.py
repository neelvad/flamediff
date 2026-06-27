"""flamediff: structural drift diffing for model checkpoints."""
from __future__ import annotations

from flamediff.adapters import torchrec_mch, torchrec_mch_sharded  # noqa: F401  (register adapters)
from flamediff.adapters.base import get_adapter
from flamediff.detect import DetectionResult, Event, detect_trajectory
from flamediff.diff import diff_checkpoints, diff_dense, diff_table
from flamediff.mutate import Mutation, mutate_checkpoint, mutate_table
from flamediff.trajectory import MetricSeries, TrajectoryDiff, diff_trajectory
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
    "mutate_table",
    "mutate_checkpoint",
    "Mutation",
    "diff_trajectory",
    "detect_trajectory",
    "TrajectoryDiff",
    "MetricSeries",
    "DetectionResult",
    "Event",
    "Checkpoint",
    "CheckpointDiff",
    "EmbeddingTableDiff",
    "DenseTensorDiff",
]
