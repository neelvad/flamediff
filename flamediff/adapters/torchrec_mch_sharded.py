"""Adapter for *sharded* TorchRec MCEC checkpoints (torch.distributed.checkpoint / DCP dirs).

A sharded checkpoint is a DCP directory (`.metadata` + per-rank `__{rank}_{idx}.distcp`). `dcp.load`
reassembles every row-wise-sharded buffer (the weight and the id->slot map) into full tensors,
which -- because MCH shards row-wise with GLOBAL slots and range-partitioned (globally sorted) ids
-- are structurally identical to a single-device state_dict. So we reassemble, then reuse the
shared builder. Runs locally: DCP loads single-process, no GPU / torchrec / process group needed.

Stage 1 (this): full reassembly into an InMemoryTable -- correct, fits-in-RAM. Stage 2 (later):
a lazy out-of-core ShardedTable that reads only the rows it gathers.
"""
from __future__ import annotations

import os

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from flamediff.adapters._torchrec_common import assemble_checkpoint, has_mc_keys
from flamediff.adapters.base import register
from flamediff.types import Checkpoint

_DCP_METADATA = ".metadata"


class ShardedTorchRecMCHAdapter:
    name = "torchrec_mch_sharded"

    def can_load(self, path: str) -> bool:
        if not (os.path.isdir(path) and os.path.exists(os.path.join(path, _DCP_METADATA))):
            return False
        try:
            md = FileSystemReader(path).read_metadata()
        except Exception:
            return False
        return has_mc_keys(md.state_dict_metadata)

    def load(self, path: str) -> Checkpoint:
        md = FileSystemReader(path).read_metadata()
        target = {
            fqn: torch.empty(tuple(m.size), dtype=m.properties.dtype)
            for fqn, m in md.state_dict_metadata.items()
            if isinstance(m, TensorStorageMetadata)
        }
        dcp.load(target, checkpoint_id=path)  # reassembles all shards into full tensors
        return assemble_checkpoint(target, path)


register(ShardedTorchRecMCHAdapter())
