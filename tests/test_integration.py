"""Gated integration test against the real generated trajectory (skipped if absent).

Checks ground-truth causality invariants from the fed_ids.json sidecar: you can only admit
ids you were fed.
"""
import json
import os

import pytest

from flamediff import diff_checkpoints, load_checkpoint

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
