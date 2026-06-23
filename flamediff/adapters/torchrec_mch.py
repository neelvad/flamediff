"""Adapter for TorchRec ManagedCollisionEmbeddingCollection (MCH/ZCH) checkpoints.

Parses the serialized *state_dict* directly -- it never imports torchrec -- so it works on
real TorchRec checkpoints and runs anywhere torch does (incl. arm64 macOS). Format confirmed
against a generated fixture (see scripts/inspect_fixtures.py and plan.md).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from flamediff.adapters.base import register
from flamediff.types import Checkpoint, DenseTensor, InMemoryTable

MCC_PREFIX = "_managed_collision_collection._managed_collision_modules."
EMB_PREFIX = "_embedding_module.embeddings."
EMB_SUFFIX = ".weight"
_STATE_DICT_NAMES = ("state_dict.pt", "checkpoint.pt")
_DEFAULT_DELIMITER = np.iinfo(np.int64).max


def _resolve_state_dict(path: str) -> str:
    if os.path.isdir(path):
        for name in _STATE_DICT_NAMES:
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"no state_dict ({_STATE_DICT_NAMES}) under {path!r}")
    return path


def _load_state_dict(sd_path: str) -> dict:
    try:
        return torch.load(sd_path, map_location="cpu", mmap=True, weights_only=True)
    except Exception:
        return torch.load(sd_path, map_location="cpu", weights_only=True)


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


def _read_step(path: str) -> int | None:
    if os.path.isdir(path):
        meta = os.path.join(path, "meta.json")
        if os.path.exists(meta):
            with open(meta) as fh:
                return json.load(fh).get("global_step")
    return None


class TorchRecMCHAdapter:
    name = "torchrec_mch"

    def can_load(self, path: str) -> bool:
        try:
            sd = _load_state_dict(_resolve_state_dict(path))
        except (FileNotFoundError, RuntimeError, OSError):
            return False
        return any(k.startswith(MCC_PREFIX) for k in sd)

    def load(self, path: str) -> Checkpoint:
        sd = _load_state_dict(_resolve_state_dict(path))

        mc_bufs: dict[str, dict] = {}
        emb_weights: dict[str, torch.Tensor] = {}
        leftover: dict[str, torch.Tensor] = {}
        for key, value in sd.items():
            if key.startswith(MCC_PREFIX):
                table, buf = key[len(MCC_PREFIX):].split(".", 1)
                mc_bufs.setdefault(table, {})[buf] = value
            elif key.startswith(EMB_PREFIX) and key.endswith(EMB_SUFFIX):
                table = key[len(EMB_PREFIX):-len(EMB_SUFFIX)]
                emb_weights[table] = value
            elif torch.is_tensor(value):
                leftover[key] = value

        embedding_tables = {}
        for table, bufs in mc_bufs.items():
            weight = emb_weights.pop(table, None)
            if weight is not None:
                embedding_tables[table] = _build_table(table, bufs, weight)

        # embedding tables without managed collision -> treat as dense for now
        for table, weight in emb_weights.items():
            leftover[f"{EMB_PREFIX}{table}{EMB_SUFFIX}"] = weight

        dense_tensors = {k: DenseTensor(k, v) for k, v in leftover.items()}
        return Checkpoint(
            path=path,
            step=_read_step(path),
            embedding_tables=embedding_tables,
            dense_tensors=dense_tensors,
        )


register(TorchRecMCHAdapter())
