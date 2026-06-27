"""Adapter for single-device TorchRec ManagedCollisionEmbeddingCollection (MCH/ZCH) checkpoints.

Parses the serialized *state_dict* directly -- it never imports torchrec -- so it works on real
TorchRec checkpoints and runs anywhere torch does (incl. arm64 macOS). See torchrec_mch_sharded.py
for the sharded (DCP) variant; both share the table-building in _torchrec_common.
"""
from __future__ import annotations

import os

import torch

from flamediff.adapters._torchrec_common import assemble_checkpoint, has_mc_keys
from flamediff.adapters.base import register
from flamediff.types import Checkpoint

_STATE_DICT_NAMES = ("state_dict.pt", "checkpoint.pt")


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


class TorchRecMCHAdapter:
    name = "torchrec_mch"

    def can_load(self, path: str) -> bool:
        try:
            sd = _load_state_dict(_resolve_state_dict(path))
        except (FileNotFoundError, RuntimeError, OSError):
            return False
        return has_mc_keys(sd)

    def load(self, path: str) -> Checkpoint:
        return assemble_checkpoint(_load_state_dict(_resolve_state_dict(path)), path)


register(TorchRecMCHAdapter())
