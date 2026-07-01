"""Adapter for *sharded* TorchRec MCEC checkpoints (torch.distributed.checkpoint / DCP dirs).

A sharded checkpoint is a DCP directory (`.metadata` + per-rank `__{rank}_{idx}.distcp`). `dcp.load`
reassembles every row-wise-sharded buffer (the weight and the id->slot map) into full tensors,
which -- because MCH shards row-wise with GLOBAL slots and range-partitioned (globally sorted) ids
-- are structurally identical to a single-device state_dict. So we reassemble, then reuse the
shared builder. Runs locally: DCP loads single-process, no GPU / torchrec / process group needed.

Scale: the map (sorted_ids/slots/counts) is ~20x smaller than the weight, so it always loads to
RAM. A weight larger than ``out_of_core_bytes`` is read **out-of-core**: zero-copy by mmapping the
``.distcp`` chunks directly (Stage 2.5, see ``_dcp_zerocopy``), or -- if the on-disk framing isn't
the expected stored-zip layout -- a reassembled mmap-backed scratch copy (Stage 2). Either way
InMemoryTable gathers rows lazily; peak RAM is bounded by the rows actually touched.
"""
from __future__ import annotations

import atexit
import math
import os
import shutil
import tempfile
import uuid
from typing import TYPE_CHECKING

import torch

from flamediff.adapters._torchrec_common import assemble_checkpoint, has_mc_keys
from flamediff.adapters.base import register
from flamediff.types import Checkpoint

if TYPE_CHECKING:
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

# torch.distributed.checkpoint (and the zero-copy reader) are imported lazily inside the methods so
# `import flamediff` -- which imports this module to register the adapter -- doesn't pull in
# torch.distributed (keeps `flamediff --help` quiet and the package import light).

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
        from torch.distributed.checkpoint import FileSystemReader
        try:
            md = FileSystemReader(path).read_metadata()
        except Exception:
            return False
        return has_mc_keys(md.state_dict_metadata)

    def load(self, path: str) -> Checkpoint:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.metadata import TensorStorageMetadata

        from flamediff.adapters._dcp_zerocopy import open_zero_copy_weight

        md = FileSystemReader(path).read_metadata()
        target: dict = {}
        zero_copy: dict = {}
        for fqn, m in md.state_dict_metadata.items():
            if not isinstance(m, TensorStorageMetadata):
                continue
            # large weights: zero-copy mmap the .distcp chunks; fall back to a scratch copy.
            # maps and small weights: reassemble into RAM via dcp.load.
            if fqn.endswith(".weight") and _bytes(m) > self.out_of_core_bytes:
                try:
                    zero_copy[fqn] = open_zero_copy_weight(path, md, fqn)
                except Exception:
                    target[fqn] = _mmap_target(m)
            else:
                target[fqn] = torch.empty(tuple(m.size), dtype=m.properties.dtype)
        dcp.load(target, checkpoint_id=path)  # maps to RAM; any fallback weights to scratch
        target.update(zero_copy)              # inject the mmap-backed zero-copy weights
        return assemble_checkpoint(target, path)


register(ShardedTorchRecMCHAdapter())
