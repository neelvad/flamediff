"""Adapter for *sharded* TorchRec MCEC checkpoints (torch.distributed.checkpoint / DCP dirs).

A sharded checkpoint is a DCP directory (`.metadata` + per-rank `__{rank}_{idx}.distcp`). `dcp.load`
reassembles every row-wise-sharded buffer (the weight and the id->slot map) into full tensors,
which -- because MCH shards row-wise with GLOBAL slots and range-partitioned (globally sorted) ids
-- are structurally identical to a single-device state_dict. So we reassemble, then reuse the
shared builder. Runs locally: DCP loads single-process, no GPU / torchrec / process group needed.

Scale: the map (sorted_ids/slots/counts) is ~20x smaller than the weight, so it always loads to
RAM. A weight larger than ``out_of_core_bytes`` is reassembled into an **mmap-backed scratch file**
instead (peak RAM stays bounded by DCP's per-chunk buffer); InMemoryTable then gathers rows lazily
from the mmap. (Cost: a scratch copy of the weight + one streaming read. A future Stage 2.5 could
mmap the .distcp chunks directly to avoid the copy.)
"""
from __future__ import annotations

import atexit
import math
import os
import shutil
import tempfile
import uuid

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from flamediff.adapters._torchrec_common import assemble_checkpoint, has_mc_keys
from flamediff.adapters.base import register
from flamediff.types import Checkpoint

_DCP_METADATA = ".metadata"
_DEFAULT_OOC_BYTES = 2 * 1024**3  # weights above this reassemble out-of-core (mmap scratch)
_SCRATCH_DIR: str | None = None


def _scratch_dir() -> str:
    global _SCRATCH_DIR
    if _SCRATCH_DIR is None:
        _SCRATCH_DIR = tempfile.mkdtemp(prefix="flamediff-ooc-")
        atexit.register(shutil.rmtree, _SCRATCH_DIR, ignore_errors=True)
    return _SCRATCH_DIR


def _bytes(m: TensorStorageMetadata) -> int:
    return math.prod(tuple(m.size)) * torch.empty((), dtype=m.properties.dtype).element_size()


def _mmap_target(m: TensorStorageMetadata) -> torch.Tensor:
    """An mmap-backed scratch tensor to reassemble a large weight into (gathered lazily)."""
    size = tuple(m.size)
    path = os.path.join(_scratch_dir(), f"w_{uuid.uuid4().hex}.bin")
    return torch.from_file(path, shared=True, size=math.prod(size),
                           dtype=m.properties.dtype).reshape(size)


class ShardedTorchRecMCHAdapter:
    name = "torchrec_mch_sharded"

    def __init__(self, out_of_core_bytes: int = _DEFAULT_OOC_BYTES):
        self.out_of_core_bytes = out_of_core_bytes

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
        target = {}
        for fqn, m in md.state_dict_metadata.items():
            if not isinstance(m, TensorStorageMetadata):
                continue
            # large weights -> mmap scratch (lazy); maps and small weights -> RAM
            if fqn.endswith(".weight") and _bytes(m) > self.out_of_core_bytes:
                target[fqn] = _mmap_target(m)
            else:
                target[fqn] = torch.empty(tuple(m.size), dtype=m.properties.dtype)
        dcp.load(target, checkpoint_id=path)  # reassembles shards; big weights stream to scratch
        return assemble_checkpoint(target, path)


register(ShardedTorchRecMCHAdapter())
