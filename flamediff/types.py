"""Normalized representation (the adapter<->diff contract) and diff-result types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch


class EmbeddingTable(Protocol):
    """One managed-collision embedding table at one checkpoint.

    All access is vectorized (id arrays in, arrays/tensors out) so it scales, and weight
    access goes through ``gather`` so a future mmap/sharded backend is a drop-in.
    """

    name: str
    num_slots: int
    dim: int

    def ids(self) -> np.ndarray:
        """Resident raw ids, sorted ascending (int64)."""
        ...

    def slot_of(self, ids: np.ndarray) -> np.ndarray:
        """Map raw ids -> slot index; -1 for non-resident ids."""
        ...

    def counts(self, ids: np.ndarray) -> np.ndarray | None:
        """LFU access count per id, or None if the format carries none."""
        ...

    def gather(self, ids: np.ndarray) -> torch.Tensor:
        """Embeddings ``[len(ids), dim]`` for the given (resident) ids."""
        ...


class InMemoryTable:
    """``EmbeddingTable`` backed by the same sorted parallel-array layout as the on-disk MCH
    format: ``_sorted_ids`` / ``_slots`` / ``_counts`` (per resident id) plus a (possibly
    mmap-backed) ``[num_slots, dim]`` weight tensor."""

    def __init__(
        self,
        name: str,
        num_slots: int,
        sorted_ids: np.ndarray,
        slots: np.ndarray,
        weights: torch.Tensor,
        counts: np.ndarray | None = None,
    ) -> None:
        self.name = name
        self.num_slots = int(num_slots)
        self._sorted_ids = np.ascontiguousarray(sorted_ids, dtype=np.int64)
        self._slots = np.ascontiguousarray(slots, dtype=np.int64)
        self._counts = None if counts is None else np.ascontiguousarray(counts, dtype=np.int64)
        self._weights = weights
        self.dim = int(weights.shape[1])
        if self._slots.shape != self._sorted_ids.shape:
            raise ValueError("sorted_ids and slots must have the same shape")

    def ids(self) -> np.ndarray:
        return self._sorted_ids

    def _positions(self, ids: np.ndarray) -> np.ndarray:
        """Index into the sorted arrays for each id, or -1 if not present."""
        ids = np.asarray(ids, dtype=np.int64)
        pos = np.searchsorted(self._sorted_ids, ids)
        in_range = pos < self._sorted_ids.shape[0]
        found = np.zeros(ids.shape, dtype=bool)
        found[in_range] = self._sorted_ids[pos[in_range]] == ids[in_range]
        return np.where(found, pos, -1)

    def slot_of(self, ids: np.ndarray) -> np.ndarray:
        pos = self._positions(ids)
        out = np.full(pos.shape, -1, dtype=np.int64)
        out[pos >= 0] = self._slots[pos[pos >= 0]]
        return out

    def counts(self, ids: np.ndarray) -> np.ndarray | None:
        if self._counts is None:
            return None
        pos = self._positions(ids)
        out = np.full(pos.shape, -1, dtype=np.int64)
        out[pos >= 0] = self._counts[pos[pos >= 0]]
        return out

    def gather(self, ids: np.ndarray) -> torch.Tensor:
        slots = self.slot_of(ids)
        if (slots < 0).any():
            raise KeyError("gather() called with non-resident ids")
        return self._weights[torch.from_numpy(slots)]

    def copy_weights(self) -> torch.Tensor:
        return self._weights.clone()

    def with_weights(self, weights: torch.Tensor) -> InMemoryTable:
        """A new table sharing this one's id->slot map and counts but with replaced weights."""
        return InMemoryTable(
            self.name, self.num_slots, self._sorted_ids, self._slots, weights, self._counts
        )


class DenseTensor:
    """A non-embedding weight tensor (MLP layers, interaction net, ...)."""

    def __init__(self, name: str, tensor: torch.Tensor) -> None:
        self.name = name
        self._tensor = tensor

    def values(self) -> torch.Tensor:
        return self._tensor

    @property
    def shape(self) -> tuple:
        return tuple(self._tensor.shape)


@dataclass
class Checkpoint:
    path: str
    step: int | None = None
    embedding_tables: dict[str, EmbeddingTable] = field(default_factory=dict)
    dense_tensors: dict[str, DenseTensor] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


@dataclass
class GeomStats:
    """Geometry of a table's resident embedding cloud (from the row covariance)."""

    n: int
    mean_row_norm: float
    effective_rank: float
    anisotropy: float


@dataclass
class EmbeddingTableDiff:
    name: str
    n_prev: int
    n_cur: int
    n_survivors: int
    n_inserted: int
    n_evicted: int
    n_slot_stable: int
    n_slot_moved: int
    # per-id arrays over the CLEAN set (slot-stable survivors), all aligned with surv_ids
    surv_ids: np.ndarray
    delta_norm: np.ndarray
    cosine: np.ndarray
    dcount: np.ndarray
    freq_resid: np.ndarray
    frozen_score: np.ndarray
    geom_prev: GeomStats
    geom_cur: GeomStats
    inserted_ids: np.ndarray | None = None
    evicted_ids: np.ndarray | None = None
    slot_moved_ids: np.ndarray | None = None

    def churn_summary(self) -> dict:
        return {
            "prev": self.n_prev,
            "cur": self.n_cur,
            "survivors": self.n_survivors,
            "inserted": self.n_inserted,
            "evicted": self.n_evicted,
            "slot_stable": self.n_slot_stable,
            "slot_moved": self.n_slot_moved,
        }

    def top_movers(self, k: int = 10, by: str = "freq_resid") -> list[tuple]:
        """Top-k clean survivors by descending metric (most positive first); for freq_resid
        that is "moved more than its training predicts". Returns (id, score, delta_norm, dcount).
        """
        arr = getattr(self, by)
        if arr.size == 0:
            return []
        order = np.argsort(-arr)[:k]
        return [
            (int(self.surv_ids[i]), float(arr[i]), float(self.delta_norm[i]), int(self.dcount[i]))
            for i in order
        ]

    def top_frozen(self, k: int = 10) -> list[tuple]:
        """Top-k clean survivors most "trained but didn't move" (saturated/frozen).

        Returns (id, frozen_score, delta_norm, dcount).
        """
        if self.frozen_score.size == 0:
            return []
        order = np.argsort(-self.frozen_score)[:k]
        return [
            (int(self.surv_ids[i]), float(self.frozen_score[i]),
             float(self.delta_norm[i]), int(self.dcount[i]))
            for i in order
        ]


@dataclass
class DenseTensorDiff:
    name: str
    delta_norm: float
    rel_delta_norm: float
    cosine: float
    eff_rank_prev: float
    eff_rank_cur: float
    spectral_norm_prev: float
    spectral_norm_cur: float


@dataclass
class CheckpointDiff:
    step_prev: int | None
    step_cur: int | None
    embedding_diffs: dict[str, EmbeddingTableDiff] = field(default_factory=dict)
    dense_diffs: dict[str, DenseTensorDiff] = field(default_factory=dict)
