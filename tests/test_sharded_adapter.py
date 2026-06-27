import os

import pytest

from flamediff import diff_checkpoints, load_checkpoint

SHARDED_FIXTURE = "fixtures/sharded_sem"  # a DCP dir (2-rank, populated); see scripts/


def _fixture():
    meta = os.path.join(SHARDED_FIXTURE, ".metadata")
    if not (os.path.isdir(SHARDED_FIXTURE) and os.path.exists(meta)):
        pytest.skip("no sharded DCP fixture under fixtures/sharded_sem")
    return SHARDED_FIXTURE


@pytest.mark.integration
def test_loads_sharded_dcp_checkpoint():
    # the sharded adapter should win dispatch on a DCP dir and reassemble it locally.
    ck = load_checkpoint(_fixture())
    assert "author_id_emb" in ck.embedding_tables
    table = ck.embedding_tables["author_id_emb"]
    assert table.ids().size > 0                       # populated
    # the reassembled map is globally sorted (range-partitioned ids)
    import numpy as np
    assert bool(np.all(np.diff(table.ids()) > 0))


@pytest.mark.integration
def test_sharded_self_diff_is_zero():
    # diffing a reassembled sharded checkpoint against itself: no churn, no movement.
    ck = load_checkpoint(_fixture())
    td = diff_checkpoints(ck, ck).embedding_diffs["author_id_emb"]
    assert td.n_inserted == 0 and td.n_evicted == 0 and td.n_readmitted == 0
    assert td.n_slot_moved == 0
    assert td.n_survivors == ck.embedding_tables["author_id_emb"].ids().size
    assert (td.delta_norm.size == 0) or float(td.delta_norm.max()) == 0.0
