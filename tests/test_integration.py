"""Gated integration test against the real generated trajectory (skipped if absent).

Checks ground-truth causality invariants from the fed_ids.json sidecar: you can only admit
ids you were fed.
"""
import glob
import json
import os

import numpy as np
import pytest

from flamediff import (
    detect_trajectory,
    diff_checkpoints,
    diff_trajectory,
    load_checkpoint,
    mutate_checkpoint,
)

TABLE = "author_id_emb"
FEATURE = "author_id"


def _fed(run, ckpt):
    with open(os.path.join(run, ckpt, "fed_ids.json")) as fh:
        return set(json.load(fh)[FEATURE])


@pytest.mark.integration
def test_real_fixture_partition_and_causality(real_fixture):
    ck0 = load_checkpoint(os.path.join(real_fixture, "ckpt_000"))
    ck1 = load_checkpoint(os.path.join(real_fixture, "ckpt_001"))
    d = diff_checkpoints(ck0, ck1)
    td = d.embedding_diffs[TABLE]

    # partition invariants
    assert td.n_survivors + td.n_evicted == td.n_prev
    assert td.n_survivors + td.n_inserted == td.n_cur
    assert td.n_slot_stable + td.n_slot_moved == td.n_survivors

    # causality: a resident id must have been fed in some interval up to now (admission can
    # lag feeding by an interval, so the check is against the *cumulative* fed set).
    fed_cum0 = _fed(real_fixture, "ckpt_000")
    fed_cum1 = fed_cum0 | _fed(real_fixture, "ckpt_001")
    assert set(ck0.embedding_tables[TABLE].ids().tolist()).issubset(fed_cum0)
    assert set(ck1.embedding_tables[TABLE].ids().tolist()).issubset(fed_cum1)
    assert set(td.inserted_ids.tolist()).issubset(fed_cum1)
    # nothing evicted yet while the table is still filling (confirmed in inspect_fixtures)
    assert td.n_evicted == 0


@pytest.mark.integration
def test_trajectory_detects_injected_corruption(real_fixture):
    # mutation test at trajectory level: scramble a mid-run checkpoint and assert the detector
    # flags the step(s) touching it, above the run's own noise floor.
    paths = sorted(glob.glob(os.path.join(real_fixture, "ckpt_*")))
    if len(paths) < 6:
        pytest.skip("trajectory detection needs a longer run")
    cks = [load_checkpoint(p) for p in paths]

    target = len(cks) // 2  # a mid-run checkpoint
    survivors = np.intersect1d(
        cks[target - 1].embedding_tables[TABLE].ids(),
        cks[target].embedding_tables[TABLE].ids(),
    )
    injected = np.random.default_rng(0).choice(survivors, size=20, replace=False)
    cks[target], _ = mutate_checkpoint(cks[target], TABLE, kind="scramble",
                                       ids=injected, magnitude=8.0)

    result = detect_trajectory(diff_trajectory(cks))
    # the diffs touching the mutated checkpoint are at positions target-1 and target
    hit = {(e.index, e.table) for e in result.events}
    assert any(idx in (target - 1, target) and tbl == TABLE for idx, tbl in hit)
