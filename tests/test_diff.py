import numpy as np
import torch

from flamediff.diff import diff_dense, diff_table
from flamediff.types import DenseTensor


def test_partition_and_slot_stability(make_table):
    # prev ids {1,2,3}; cur ids {2,3,4}. id 3 keeps its slot, id 2 moves slot.
    prev = make_table(ids=[1, 2, 3], slots=[0, 1, 2], counts=[5, 5, 5],
                      weights=torch.zeros(10, 4), num_slots=10)
    cur = make_table(ids=[2, 3, 4], slots=[7, 2, 8], counts=[6, 9, 1],
                     weights=torch.zeros(10, 4), num_slots=10)
    d = diff_table(prev, cur)

    assert d.n_survivors == 2
    assert d.n_inserted == 1 and set(d.inserted_ids.tolist()) == {4}
    assert d.n_evicted == 1 and set(d.evicted_ids.tolist()) == {1}
    # id 2 moved slot (1 -> 7) => comparability break; id 3 stable (2 -> 2) => clean set
    assert d.n_slot_moved == 1 and set(d.slot_moved_ids.tolist()) == {2}
    assert d.n_slot_stable == 1 and set(d.surv_ids.tolist()) == {3}


def test_partition_invariants(make_table):
    prev = make_table(ids=[1, 2, 3, 4], slots=[0, 1, 2, 3], counts=[1, 1, 1, 1], num_slots=8)
    cur = make_table(ids=[3, 4, 5, 6], slots=[2, 3, 4, 5], counts=[2, 2, 2, 2], num_slots=8)
    d = diff_table(prev, cur)

    assert d.n_survivors + d.n_evicted == d.n_prev
    assert d.n_survivors + d.n_inserted == d.n_cur
    assert d.n_slot_stable + d.n_slot_moved == d.n_survivors


def test_freq_residual_flags_unexpected_mover(make_table):
    # most ids move proportionally to dcount; one rare id moves hugely -> top residual.
    n, dim = 200, 4
    ids = np.arange(1, n + 1)
    slots = np.arange(n)
    rng = np.random.default_rng(0)
    dcount = rng.integers(1, 100, size=n).astype(np.int64)

    Wp = torch.zeros(n, dim)
    Wc = torch.zeros(n, dim)
    for i in range(n):
        Wc[slots[i], 0] = 0.01 * dcount[i]   # expected: ||delta|| grows with dcount
    dcount[0] = 1                            # rare...
    Wc[slots[0], 0] = 5.0                    # ...but moved a lot -> anomaly

    prev = make_table(ids=ids, slots=slots, counts=np.zeros(n, dtype=np.int64),
                      weights=Wp, num_slots=n)
    cur = make_table(ids=ids, slots=slots, counts=dcount, weights=Wc, num_slots=n)
    d = diff_table(prev, cur)

    top_id, top_score, _, top_dcount = d.top_movers(1, by="freq_resid")[0]
    assert top_id == int(ids[0])
    assert top_dcount == 1
    assert top_score > 5.0  # well clear of the noise floor


def test_frozen_score_flags_trained_but_frozen(make_table):
    # most ids move proportionally to dcount; one heavily-trained id is frozen (no movement).
    n, dim = 100, 4
    ids = np.arange(1, n + 1)
    slots = np.arange(n)
    rng = np.random.default_rng(1)
    dcount = rng.integers(1, 50, size=n).astype(np.int64)

    Wp = torch.zeros(n, dim)
    Wc = torch.zeros(n, dim)
    for i in range(n):
        Wc[slots[i], 0] = 0.01 * dcount[i]
    dcount[0] = 1000          # trained the most...
    Wc[slots[0], 0] = 0.0     # ...but did not move at all -> frozen/saturated

    prev = make_table(ids=ids, slots=slots, counts=np.zeros(n, dtype=np.int64),
                      weights=Wp, num_slots=n)
    cur = make_table(ids=ids, slots=slots, counts=dcount, weights=Wc, num_slots=n)
    d = diff_table(prev, cur)

    top_id, top_frozen, top_dn, top_dc = d.top_frozen(1)[0]
    assert top_id == int(ids[0])
    assert top_dc == 1000 and top_dn == 0.0
    assert top_frozen > 0.5
    # and it is NOT a freq_resid mover (it did not move) -> the two scores are complementary
    assert int(d.top_movers(1, by="freq_resid")[0][0]) != int(ids[0])


def test_diff_dense_basic():
    a = torch.randn(16, 8)
    b = a + 0.1 * torch.randn(16, 8)
    dd = diff_dense(DenseTensor("w", a), DenseTensor("w", b))
    assert dd.delta_norm > 0
    assert -1.0001 <= dd.cosine <= 1.0001
    assert dd.eff_rank_cur > 0 and dd.spectral_norm_cur > 0
