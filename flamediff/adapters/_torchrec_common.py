"""Shared TorchRec MCH parsing: turn a flat {fqn: tensor} state_dict into a Checkpoint.

The single-device and sharded adapters differ only in *how* they obtain that flat dict
(torch.load vs DCP reassembly). The table-building is identical, because a reassembled sharded
checkpoint is structurally the same as a single-device one -- MCH shards row-wise with GLOBAL
slots and range-partitioned, globally-sorted ids (verified -- see discover_sharded_semantics.py).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from flamediff.types import Checkpoint, DenseTensor, EmbeddingTable, InMemoryTable

MCC_PREFIX = "_managed_collision_collection._managed_collision_modules."
EMB_PREFIX = "_embedding_module.embeddings."
EMB_SUFFIX = ".weight"
_DEFAULT_DELIMITER = np.iinfo(np.int64).max


def has_mc_keys(keys) -> bool:
    return any(k.startswith(MCC_PREFIX) for k in keys)


def read_step(path: str) -> int | None:
    if os.path.isdir(path):
        meta = os.path.join(path, "meta.json")
        if os.path.exists(meta):
            with open(meta) as fh:
                return json.load(fh).get("global_step")
    return None


def _build_table(name: str, bufs: dict, weight: torch.Tensor) -> InMemoryTable:
    raw = bufs["_mch_sorted_raw_ids"].cpu().numpy()
    slots = bufs["_mch_remapped_ids_mapping"].cpu().numpy()
    delim = int(bufs["_delimiter"].item()) if "_delimiter" in bufs else _DEFAULT_DELIMITER
    counts = bufs["_mch_counts"].cpu().numpy() if "_mch_counts" in bufs else None

    occupied = raw != delim
    sorted_ids, slots = raw[occupied], slots[occupied]
    counts = counts[occupied] if counts is not None else None

    # the format stores ids sorted; repair if a future variant doesn't
    if sorted_ids.size > 1 and not np.all(np.diff(sorted_ids) > 0):
        order = np.argsort(sorted_ids, kind="stable")
        sorted_ids, slots = sorted_ids[order], slots[order]
        counts = counts[order] if counts is not None else None

    return InMemoryTable(name, int(weight.shape[0]), sorted_ids, slots, weight, counts)


def assemble_checkpoint(sd: dict, path: str) -> Checkpoint:
    """Group an MCEC state_dict (flat {fqn: tensor}) into a Checkpoint."""
    mc_bufs: dict[str, dict] = {}
    emb_weights: dict[str, torch.Tensor] = {}
    leftover: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        if key.startswith(MCC_PREFIX):
            table, buf = key[len(MCC_PREFIX):].split(".", 1)
            mc_bufs.setdefault(table, {})[buf] = value
        elif key.startswith(EMB_PREFIX) and key.endswith(EMB_SUFFIX):
            emb_weights[key[len(EMB_PREFIX):-len(EMB_SUFFIX)]] = value
        elif torch.is_tensor(value):
            leftover[key] = value

    embedding_tables: dict[str, EmbeddingTable] = {}
    for table, bufs in mc_bufs.items():
        weight = emb_weights.pop(table, None)
        if weight is not None:
            embedding_tables[table] = _build_table(table, bufs, weight)
    # embedding tables without managed collision -> treat as dense for now
    for table, weight in emb_weights.items():
        leftover[f"{EMB_PREFIX}{table}{EMB_SUFFIX}"] = weight

    return Checkpoint(
        path=path,
        step=read_step(path),
        embedding_tables=embedding_tables,
        dense_tensors={k: DenseTensor(k, v) for k, v in leftover.items()},
    )
