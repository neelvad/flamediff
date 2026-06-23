"""Pairwise structural diff over the normalized representation.

Pure measurement: it computes quantities, it does not decide what is anomalous (that needs
the run's own trajectory and is the deferred detection layer).
"""
from __future__ import annotations

import numpy as np

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


def _geom(table: EmbeddingTable) -> GeomStats:
    ids = table.ids()
    if ids.size == 0:
        return GeomStats(0, 0.0, 0.0, 0.0)
    W = table.gather(ids).float()
    eig = stats.row_covariance_eigvals(W)
    return GeomStats(
        n=int(ids.size),
        mean_row_norm=stats.mean_row_norm(W),
        effective_rank=stats.effective_rank_from_spectrum(eig),
        anisotropy=stats.anisotropy_from_spectrum(eig),
    )


def diff_table(
    prev: EmbeddingTable, cur: EmbeddingTable, *, keep_ids: bool = True
) -> EmbeddingTableDiff:
    ids_prev, ids_cur = prev.ids(), cur.ids()
    survivors = np.intersect1d(ids_prev, ids_cur, assume_unique=True)
    inserted = np.setdiff1d(ids_cur, ids_prev, assume_unique=True)
    evicted = np.setdiff1d(ids_prev, ids_cur, assume_unique=True)

    # slot-stable survivors are the clean comparison set; slot-moved are comparability
    # breaks (eviction inherits the slot's vector) -> excluded from learning deltas.
    stable_mask = prev.slot_of(survivors) == cur.slot_of(survivors)
    stable = survivors[stable_mask]
    moved = survivors[~stable_mask]

    if stable.size:
        Wp = prev.gather(stable).float()
        Wc = cur.gather(stable).float()
        delta_norm = stats.row_delta_norm(Wp, Wc).cpu().numpy()
        cosine = stats.row_cosine(Wp, Wc).cpu().numpy()
        cp, cc = prev.counts(stable), cur.counts(stable)
        dcount = (cc - cp).astype(np.int64) if cp is not None and cc is not None \
            else np.zeros(stable.shape, dtype=np.int64)
        freq_resid = stats.freq_residual(delta_norm, dcount)
    else:
        delta_norm = cosine = freq_resid = np.zeros(0, dtype=np.float64)
        dcount = np.zeros(0, dtype=np.int64)

    return EmbeddingTableDiff(
        name=cur.name,
        n_prev=int(ids_prev.size),
        n_cur=int(ids_cur.size),
        n_survivors=int(survivors.size),
        n_inserted=int(inserted.size),
        n_evicted=int(evicted.size),
        n_slot_stable=int(stable.size),
        n_slot_moved=int(moved.size),
        surv_ids=stable,
        delta_norm=delta_norm,
        cosine=cosine,
        dcount=dcount,
        freq_resid=freq_resid,
        geom_prev=_geom(prev),
        geom_cur=_geom(cur),
        inserted_ids=inserted if keep_ids else None,
        evicted_ids=evicted if keep_ids else None,
        slot_moved_ids=moved if keep_ids else None,
    )


def diff_dense(prev: DenseTensor, cur: DenseTensor) -> DenseTensorDiff:
    a, b = prev.values().float(), cur.values().float()
    delta_norm = stats.tensor_delta_norm(a, b)
    return DenseTensorDiff(
        name=cur.name,
        delta_norm=delta_norm,
        rel_delta_norm=delta_norm / (float(a.norm()) + 1e-12),
        cosine=stats.tensor_cosine(a, b),
        eff_rank_prev=stats.matrix_effective_rank(a),
        eff_rank_cur=stats.matrix_effective_rank(b),
        spectral_norm_prev=stats.spectral_norm(a),
        spectral_norm_cur=stats.spectral_norm(b),
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
