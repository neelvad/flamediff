"""Zero-copy reads of a row-wise-sharded DCP weight (Stage 2.5).

DCP writes each tensor storage *uncompressed* inside a torch.save zip in the per-rank `.distcp`
files. So instead of `dcp.load`-ing the weight into a scratch copy (Stage 2), we parse each chunk's
zip local header to find its raw-bytes offset and mmap the shard file there -- `gather(rows)` then
pages in only the rows it touches, with no copy of the weight. Anything unexpected (compression,
an odd dtype) raises, and the caller falls back to the scratch path -- so this is safe by default.
"""
from __future__ import annotations

import os
import struct

import numpy as np
import torch

_LOCAL_SIG, _CD_SIG, _EOCD_SIG = b"PK\x03\x04", b"PK\x01\x02", b"PK\x05\x06"
_LOCAL = struct.Struct("<4s5H3I2H")   # local file header (30 bytes fixed)
_CD = struct.Struct("<4s6H3I5H2I")    # central directory file header (46 bytes fixed)
_EOCD = struct.Struct("<4s4H2IH")     # end of central directory (22 bytes)
_NP_DTYPE = {torch.float32: np.float32, torch.float16: np.float16, torch.float64: np.float64}


def _stored_data_offset(path: str, zip_start: int, zip_len: int, expected_bytes: int) -> int:
    """Locate the raw bytes of the storage entry (uncompressed size == expected_bytes) inside the
    torch.save mini-zip embedded at [zip_start, zip_start+zip_len). Each .distcp tensor is its own
    mini-zip whose first entry is `data.pkl`, not the storage -- so walk the central directory.
    """
    with open(path, "rb") as fh:
        tail_at = zip_start + zip_len - _EOCD.size  # torch.save writes no zip comment
        fh.seek(tail_at)
        eocd = _EOCD.unpack(fh.read(_EOCD.size))
        if eocd[0] != _EOCD_SIG:
            raise ValueError("no EOCD at expected position")
        cd_size, cd_off = eocd[5], eocd[6]
        fh.seek(zip_start + cd_off)
        cd = fh.read(cd_size)

        pos = 0
        while pos + _CD.size <= len(cd):
            f = _CD.unpack_from(cd, pos)
            if f[0] != _CD_SIG:
                break
            usize, fnlen, extralen, commentlen, lho = f[9], f[10], f[11], f[12], f[16]
            if usize == expected_bytes:
                fh.seek(zip_start + lho)
                lh = _LOCAL.unpack(fh.read(_LOCAL.size))
                if lh[3] != 0:  # compression method
                    raise ValueError("storage entry is compressed")
                return zip_start + lho + _LOCAL.size + lh[9] + lh[10]
            pos += _CD.size + fnlen + extralen + commentlen
    raise ValueError("no stored entry matching expected size")


class ZeroCopyShardedWeight:
    """Tensor-like weight over per-shard mmap views; supports ``w[row_indices] -> [k, dim]``."""

    def __init__(self, chunks: list, dim: int, dtype: torch.dtype):
        self._chunks = chunks  # list of (row_start, row_end, memmap[rows, dim])
        self.dim = int(dim)
        self.dtype = dtype
        self.num_slots = max(end for _s, end, _v in chunks)
        self.shape = (self.num_slots, self.dim)

    def __getitem__(self, idx) -> torch.Tensor:
        idx_np = idx.numpy() if isinstance(idx, torch.Tensor) else np.asarray(idx, dtype=np.int64)
        out = np.empty((idx_np.size, self.dim), dtype=self._chunks[0][2].dtype)
        for r0, r1, view in self._chunks:
            m = (idx_np >= r0) & (idx_np < r1)
            if m.any():
                out[m] = view[idx_np[m] - r0]
        return torch.from_numpy(out)


def open_zero_copy_weight(checkpoint_dir: str, md, fqn: str) -> ZeroCopyShardedWeight:
    """Build a ZeroCopyShardedWeight for `fqn` from DCP metadata, or raise if not mmap-able."""
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    meta = md.state_dict_metadata[fqn]
    if not isinstance(meta, TensorStorageMetadata) or len(meta.size) != 2:
        raise ValueError("not a 2-D tensor")
    np_dtype = _NP_DTYPE[meta.properties.dtype]  # KeyError -> caller falls back
    itemsize = np.dtype(np_dtype).itemsize
    dim = int(meta.size[1])
    by_offset = {tuple(k.offset): md.storage_data[k]
                 for k in md.storage_data if getattr(k, "fqn", None) == fqn}
    chunks = []
    for ch in meta.chunks:
        row0, rows = int(ch.offsets[0]), int(ch.sizes[0])
        si = by_offset[tuple(ch.offsets)]
        path = os.path.join(checkpoint_dir, si.relative_path)
        data_off = _stored_data_offset(path, si.offset, si.length, rows * dim * itemsize)
        view = np.memmap(path, dtype=np_dtype, mode="r", offset=data_off, shape=(rows, dim))
        chunks.append((row0, row0 + rows, view))
    return ZeroCopyShardedWeight(chunks, dim, meta.properties.dtype)
