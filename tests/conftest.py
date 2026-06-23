"""Test fixtures: an in-code synthetic-MCH factory (the workhorse) and a locator for the
large generated trajectory (for the gated integration test)."""
from __future__ import annotations

import glob
import json

import numpy as np
import pytest
import torch

from flamediff.types import InMemoryTable

_DELIM = np.iinfo(np.int64).max


@pytest.fixture
def make_table():
    """Build an InMemoryTable directly with hand-authored id->slot->weight (for diff tests)."""

    def _make(name="t", *, ids, slots=None, counts=None, weights=None, num_slots=None,
              dim=4, seed=0):
        ids = np.asarray(ids, dtype=np.int64)
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        slots = (np.arange(ids.size, dtype=np.int64) if slots is None
                 else np.asarray(slots, dtype=np.int64)[order])
        counts = None if counts is None else np.asarray(counts, dtype=np.int64)[order]
        if num_slots is None:
            num_slots = int(slots.max()) + 1 if slots.size else 1
        if weights is None:
            g = torch.Generator().manual_seed(seed)
            weights = torch.randn(num_slots, dim, generator=g)
        return InMemoryTable(name, num_slots, ids, slots, weights, counts)

    return _make


@pytest.fixture
def write_ckpt(tmp_path):
    """Write a checkpoint dir whose state_dict mimics the TorchRec MCEC serialization."""

    def _write(name, tables, step=0):
        sd = {}
        for tbl, t in tables.items():
            ids = np.asarray(t["ids"], dtype=np.int64)
            slots = np.asarray(t["slots"], dtype=np.int64)
            counts = np.asarray(t.get("counts", np.zeros_like(ids)), dtype=np.int64)
            weight = t["weight"]
            order = np.argsort(ids, kind="stable")
            ids, slots, counts = ids[order], slots[order], counts[order]
            pad = int(weight.shape[0]) - ids.size
            pfx = f"_managed_collision_collection._managed_collision_modules.{tbl}."
            sd[pfx + "_mch_sorted_raw_ids"] = torch.from_numpy(
                np.concatenate([ids, np.full(pad, _DELIM, dtype=np.int64)]))
            sd[pfx + "_mch_remapped_ids_mapping"] = torch.from_numpy(
                np.concatenate([slots, np.zeros(pad, dtype=np.int64)]))
            sd[pfx + "_mch_counts"] = torch.from_numpy(
                np.concatenate([counts, np.zeros(pad, dtype=np.int64)]))
            sd[pfx + "_delimiter"] = torch.tensor([_DELIM])
            sd[pfx + "_mch_slots"] = torch.tensor([int(weight.shape[0]) - 1])
            sd[f"_embedding_module.embeddings.{tbl}.weight"] = weight

        d = tmp_path / name
        d.mkdir()
        torch.save(sd, d / "state_dict.pt")
        (d / "meta.json").write_text(json.dumps({"global_step": step}))
        return str(d)

    return _write


@pytest.fixture
def real_fixture():
    runs = sorted(glob.glob("fixtures/run_*"))
    if not runs:
        pytest.skip("no generated trajectory under fixtures/ (run scripts/generate_checkpoints.py)")
    return runs[-1]
