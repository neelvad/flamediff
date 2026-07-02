"""Pairwise structural diff over the normalized representation.

Pure measurement: it computes quantities, it does not decide what is anomalous (that needs
the run's own trajectory and is the deferred detection layer).
"""
from __future__ import annotations

import numpy as np
import torch

from flamediff import stats
from flamediff.types import (
    Checkpoint,
    CheckpointDiff,
    DenseTensor,
    DenseTensorDiff,
    EmbeddingTable,
    EmbeddingTableDiff,
    GeomStats,
)

_GEOM_SAMPLE = 8192  # cap rows gathered for the covariance geometry (avoids full-table reads)
_GATHER_BATCH = 1 << 20  # survivors gathered per batch -> bounds peak to batch x dim (out-of-core)


def _geom(table: EmbeddingTable) -> tuple[GeomStats, torch.Tensor, torch.Tensor]:
    """GeomStats plus the (descending) covariance eigenvalues/eigenvectors of a row sample --
    the eigenbasis feeds the prev-vs-cur subspace-overlap measurement in ``diff_table``."""
    ids = table.ids()
    n = int(ids.size)
    if n == 0:
        return GeomStats(0, 0.0, 0.0, 0.0), torch.zeros(table.dim), torch.eye(table.dim)
    if n > _GEOM_SAMPLE:  # subsample so geometry is O(sample), not O(whole table)
        ids = ids[np.random.default_rng(0).choice(n, _GEOM_SAMPLE, replace=False)]
    W = table.gather(ids).float()
    eigvals, eigvecs = stats.row_covariance_eig(W)
    geom = GeomStats(
        n=n,
        mean_row_norm=stats.mean_row_norm(W),
        effective_rank=stats.effective_rank_from_spectrum(eigvals),
        anisotropy=stats.anisotropy_from_spectrum(eigvals),
        rank95=stats.rank_at_energy(eigvals, 0.95),
    )
    return geom, eigvals, eigvecs


def _streamed_row_stats(
    prev: EmbeddingTable, cur: EmbeddingTable, ids: np.ndarray, batch: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-id ||delta|| and cosine, gathering survivors in batches so the [n, dim] gather (the
    dominant memory) never materializes whole -- peak stays batch x dim. Result is identical to a
    single-shot gather. (The [n]-length outputs are the next limit at true billion-id scale -- a
    streaming t-digest for the percentile reductions would remove it; noted, not built.)"""
    n = int(ids.size)
    delta_norm = np.empty(n, dtype=np.float64)
    cosine = np.empty(n, dtype=np.float64)
    for lo in range(0, n, batch):
        sl = slice(lo, min(lo + batch, n))
        wp = prev.gather(ids[sl]).float()
        wc = cur.gather(ids[sl]).float()
        delta_norm[sl] = stats.row_delta_norm(wp, wc).cpu().numpy()
        cosine[sl] = stats.row_cosine(wp, wc).cpu().numpy()
    return delta_norm, cosine


def diff_table(
    prev: EmbeddingTable, cur: EmbeddingTable, *, keep_ids: bool = True,
    gather_batch: int = _GATHER_BATCH,
) -> EmbeddingTableDiff:
    ids_prev, ids_cur = prev.ids(), cur.ids()
    survivors = np.intersect1d(ids_prev, ids_cur, assume_unique=True)
    inserted = np.setdiff1d(ids_cur, ids_prev, assume_unique=True)
    evicted = np.setdiff1d(ids_prev, ids_cur, assume_unique=True)

    # Clean comparison set = survivors whose slot is unchanged AND whose LFU count did not reset.
    # A slot change is one comparability break (eviction inherits the slot's vector); a count
    # reset (dcount < 0) is another -- it means the id was evicted and re-admitted, which leaks
    # even when the slot is reused. Both are excluded from the learning deltas.
    sp, sc = prev.slot_of(survivors), cur.slot_of(survivors)
    cp, cc = prev.counts(survivors), cur.counts(survivors)
    dcount_all = ((cc - cp).astype(np.int64) if cp is not None and cc is not None
                  else np.zeros(survivors.shape, dtype=np.int64))
    slot_stable = sp == sc
    clean_mask = slot_stable & (dcount_all >= 0)
    stable = survivors[clean_mask]
    moved = survivors[~slot_stable]
    readmitted = survivors[slot_stable & (dcount_all < 0)]

    if stable.size:
        delta_norm, cosine = _streamed_row_stats(prev, cur, stable, gather_batch)
        dcount = dcount_all[clean_mask]
        freq_resid = stats.freq_residual(delta_norm, dcount)
        frozen = stats.frozen_score(delta_norm, dcount)
    else:
        delta_norm = cosine = freq_resid = frozen = np.zeros(0, dtype=np.float64)
        dcount = np.zeros(0, dtype=np.int64)

    geom_prev, eig_p, vec_p = _geom(prev)
    geom_cur, eig_c, vec_c = _geom(cur)
    # overlap over prev's 90%-energy subspace: did the basis the table actually uses rotate?
    overlap = (stats.subspace_overlap(vec_p, eig_c, vec_c, stats.rank_at_energy(eig_p, 0.90))
               if geom_prev.n >= 2 and geom_cur.n >= 2 else 1.0)

    return EmbeddingTableDiff(
        name=cur.name,
        n_prev=int(ids_prev.size),
        n_cur=int(ids_cur.size),
        n_survivors=int(survivors.size),
        n_inserted=int(inserted.size),
        n_evicted=int(evicted.size),
        n_slot_stable=int(stable.size),
        n_slot_moved=int(moved.size),
        n_readmitted=int(readmitted.size),
        surv_ids=stable,
        delta_norm=delta_norm,
        cosine=cosine,
        dcount=dcount,
        freq_resid=freq_resid,
        frozen_score=frozen,
        geom_prev=geom_prev,
        geom_cur=geom_cur,
        subspace_overlap=overlap,
        inserted_ids=inserted if keep_ids else None,
        evicted_ids=evicted if keep_ids else None,
        slot_moved_ids=moved if keep_ids else None,
        readmitted_ids=readmitted if keep_ids else None,
    )


_DENSE_SVD_CAP = 4096  # skip O(min^2 . max) SVD stats on matrices larger than this (per side)


def diff_dense(prev: DenseTensor, cur: DenseTensor) -> DenseTensorDiff:
    a, b = prev.values().float(), cur.values().float()
    delta_norm = stats.tensor_delta_norm(a, b)
    a2 = a if a.ndim == 2 else a.reshape(a.shape[0], -1)
    if min(a2.shape) <= _DENSE_SVD_CAP:
        er_p, er_c = stats.matrix_effective_rank(a), stats.matrix_effective_rank(b)
        sn_p, sn_c = stats.spectral_norm(a), stats.spectral_norm(b)
    else:  # too large for a full SVD; report the cheap stats only
        er_p = er_c = sn_p = sn_c = float("nan")
    return DenseTensorDiff(
        name=cur.name,
        delta_norm=delta_norm,
        rel_delta_norm=delta_norm / (float(a.norm()) + 1e-12),
        cosine=stats.tensor_cosine(a, b),
        eff_rank_prev=er_p, eff_rank_cur=er_c,
        spectral_norm_prev=sn_p, spectral_norm_cur=sn_c,
    )


def diff_checkpoints(
    prev: Checkpoint, cur: Checkpoint, *, dense: bool = True, keep_ids: bool = True
) -> CheckpointDiff:
    embedding_diffs = {
        name: diff_table(prev.embedding_tables[name], cur.embedding_tables[name], keep_ids=keep_ids)
        for name in sorted(set(prev.embedding_tables) & set(cur.embedding_tables))
    }
    dense_diffs = {}
    if dense:
        for name in sorted(set(prev.dense_tensors) & set(cur.dense_tensors)):
            p, c = prev.dense_tensors[name], cur.dense_tensors[name]
            if p.shape == c.shape:
                dense_diffs[name] = diff_dense(p, c)
    return CheckpointDiff(
        step_prev=prev.step,
        step_cur=cur.step,
        embedding_diffs=embedding_diffs,
        dense_diffs=dense_diffs,
    )
