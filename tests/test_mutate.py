import numpy as np
import torch

from flamediff.diff import diff_table
from flamediff.mutate import mutate_table


def test_scramble_is_detected_above_noise_floor(make_table):
    # all rows get small natural drift; a few ids are scrambled (the planted anomalies).
    n, dim = 400, 8
    ids = np.arange(1, n + 1)
    slots = np.arange(n)
    g = torch.Generator().manual_seed(0)
    Wp = torch.randn(n, dim, generator=g) * 0.1
    Wc = Wp + torch.randn(n, dim, generator=g) * 0.002  # background drift on every row
    counts = np.random.default_rng(0).integers(1, 200, n).astype(np.int64)
    prev = make_table(ids=ids, slots=slots, counts=np.zeros(n, dtype=np.int64),
                      weights=Wp, num_slots=n)
    cur = make_table(ids=ids, slots=slots, counts=counts, weights=Wc, num_slots=n)

    injected = np.array([11, 101, 201, 301, 399], dtype=np.int64)
    mutated, mut = mutate_table(cur, kind="scramble", ids=injected, magnitude=5.0, seed=1)
    d = diff_table(prev, mutated)

    # perfect recall at K: injected ids are exactly the top-5 by raw ||Δ|| ...
    assert {m[0] for m in d.top_movers(5, by="delta_norm")} == set(injected.tolist())
    # ... and by the de-confounded score (they moved far more than any dcount predicts).
    assert {m[0] for m in d.top_movers(5, by="freq_resid")} == set(injected.tolist())
    # ground truth recorded
    assert mut.kind == "scramble" and set(mut.ids.tolist()) == set(injected.tolist())


def test_mutation_isolation(make_table):
    # cur vs cur-with-mutation: exactly the mutated ids differ, nothing else.
    n, dim = 50, 4
    ids = np.arange(1, n + 1)
    slots = np.arange(n)
    g = torch.Generator().manual_seed(2)
    cur = make_table(ids=ids, slots=slots, counts=np.ones(n, dtype=np.int64),
                     weights=torch.randn(n, dim, generator=g) * 0.1, num_slots=n)

    injected = np.array([7, 22], dtype=np.int64)
    mutated, _ = mutate_table(cur, kind="zero", ids=injected)
    d = diff_table(cur, mutated)

    moved = {int(i) for i, dn in zip(d.surv_ids, d.delta_norm, strict=True) if dn > 0}
    assert moved == set(injected.tolist())


def test_mutation_ids_must_be_resident(make_table):
    cur = make_table(ids=[1, 2, 3], slots=[0, 1, 2], counts=[1, 1, 1],
                     weights=torch.randn(3, 4), num_slots=3)
    try:
        mutate_table(cur, kind="zero", ids=[999])
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-resident mutation id")
