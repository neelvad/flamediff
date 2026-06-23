"""Inject *known* corruptions into a checkpoint -- the ground truth for detection-power
(mutation) tests: confirm a metric catches a planted change above the noise floor.

Operates on the in-memory normalized representation (no Modal, no re-serialization), so it is
fully offline and format-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from flamediff.types import Checkpoint, InMemoryTable

KINDS = ("scramble", "zero", "noise")


@dataclass
class Mutation:
    """Record of an injected change (the ground truth a detection-power test asserts against)."""

    table: str
    kind: str
    ids: np.ndarray
    magnitude: float


def mutate_table(
    table: InMemoryTable, *, kind: str, ids, magnitude: float = 1.0, seed: int = 0
) -> tuple[InMemoryTable, Mutation]:
    """Return a copy of ``table`` with the given resident ids corrupted, plus the ground truth.

    kinds: "scramble" (replace each row with a random vector of norm magnitude*typical),
    "zero" (set rows to 0), "noise" (add gaussian noise of scale magnitude*typical).
    """
    ids = np.asarray(ids, dtype=np.int64)
    slots = table.slot_of(ids)
    if (slots < 0).any():
        raise ValueError("mutation ids must be resident in the table")
    W = table.copy_weights()
    idx = torch.from_numpy(slots)
    gen = torch.Generator().manual_seed(seed)
    typical = float(W.norm(dim=1).median()) or 1.0
    k = len(slots)
    if kind == "zero":
        W[idx] = 0.0
    elif kind == "scramble":
        r = torch.randn(k, table.dim, generator=gen)
        W[idx] = r / (r.norm(dim=1, keepdim=True) + 1e-12) * (magnitude * typical)
    elif kind == "noise":
        W[idx] = W[idx] + torch.randn(k, table.dim, generator=gen) * (magnitude * typical)
    else:
        raise ValueError(f"unknown mutation kind {kind!r} (expected one of {KINDS})")
    return table.with_weights(W), Mutation(table.name, kind, ids, float(magnitude))


def mutate_checkpoint(
    ckpt: Checkpoint, table_name: str, **kwargs
) -> tuple[Checkpoint, Mutation]:
    """Return a copy of ``ckpt`` with one embedding table mutated (see ``mutate_table``)."""
    mutated_table, mutation = mutate_table(ckpt.embedding_tables[table_name], **kwargs)
    tables = dict(ckpt.embedding_tables)
    tables[table_name] = mutated_table
    new = Checkpoint(
        path=f"{ckpt.path}[mutated:{table_name}]",
        step=ckpt.step,
        embedding_tables=tables,
        dense_tensors=ckpt.dense_tensors,
        meta=ckpt.meta,
    )
    return new, mutation
