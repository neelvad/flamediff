"""Attribution validation by injection: planting known, popularity-INDEPENDENT representation
changes and showing the de-confounded residual recovers them better than raw ||delta||."""
import numpy as np
import torch

from flamediff.attribute import attribute_table
from flamediff.diff import diff_table
from flamediff.stats import _avg_rank
from flamediff.types import InMemoryTable


def _auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC = P(score[positive] > score[negative])."""
    n_pos = int(positive.sum())
    n_neg = score.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _avg_rank(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _table(name, ids, weights, counts):
    return InMemoryTable(name, int(ids.max()) + 1, ids, ids.copy(), weights, counts)


def test_attribution_deconfounds_popularity():
    # Drift is CONFOUNDED by popularity: high-dcount ids move a lot just from training. Plant
    # genuine change in LOW-traffic ids -- invisible to raw ||delta|| (buried under high-dcount
    # natural movers) but caught by the residual, which knows they barely trained.
    rng = np.random.default_rng(0)
    n, dim = 2000, 32
    ids = np.arange(n, dtype=np.int64)
    dcount = rng.integers(1, 400, size=n).astype(np.int64)
    prev_counts = rng.integers(10, 100, size=n).astype(np.int64)
    cur_counts = prev_counts + dcount

    Wp = torch.from_numpy(rng.standard_normal((n, dim)).astype(np.float32))
    confounded = (0.01 * dcount[:, None] * rng.standard_normal((n, dim))).astype(np.float32)
    Wc = Wp + torch.from_numpy(confounded)

    low_traffic = np.where(dcount < 30)[0]
    planted = rng.choice(low_traffic, size=60, replace=False)
    Wc[planted] = Wp[planted] + torch.from_numpy(  # popularity-independent scramble
        rng.standard_normal((planted.size, dim)).astype(np.float32))

    prev = _table("emb", ids, Wp, prev_counts)
    cur = _table("emb", ids, Wc, cur_counts)
    diff = diff_table(prev, cur)
    attr = attribute_table(prev, cur, diff)

    planted_surv = np.isin(diff.surv_ids, planted)
    auc_raw = _auc(diff.delta_norm, planted_surv)
    auc_idio = _auc(attr.idiosyncratic, planted_surv)
    assert auc_idio > auc_raw + 0.15  # de-confounding materially lifts recovery
    assert auc_idio > 0.9
    assert attr.popularity_r2 > 0.4   # popularity really does explain much of the raw drift


def test_alignment_removes_global_rotation():
    # A pure table-wide rotation is basis drift, not per-id change: alignment should absorb it.
    rng = np.random.default_rng(1)
    n, dim = 500, 16
    ids = np.arange(n, dtype=np.int64)
    Wp = torch.from_numpy(rng.standard_normal((n, dim)).astype(np.float32))
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    Wc = Wp @ torch.from_numpy(q.astype(np.float32))

    prev = _table("emb", ids, Wp, None)
    cur = _table("emb", ids, Wc, None)
    attr = attribute_table(prev, cur, diff_table(prev, cur))
    assert attr.frac_rotation > 0.9          # nearly all drift energy is the global rotation
    assert attr.frac_aligned_residual < 0.05  # ~nothing idiosyncratic remains


def test_attribution_empty_below_threshold():
    ids = np.arange(3, dtype=np.int64)
    W = torch.zeros((3, 4))
    prev = _table("emb", ids, W, None)
    attr = attribute_table(prev, prev, diff_table(prev, prev))
    assert attr.n == 0 and attr.idiosyncratic.size == 0
