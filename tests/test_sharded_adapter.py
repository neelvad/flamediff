import os

import numpy as np
import pytest
import torch

from flamediff import diff_checkpoints, load_checkpoint
from flamediff.adapters.torchrec_mch_sharded import ShardedTorchRecMCHAdapter

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


@pytest.mark.integration
def test_out_of_core_weight_matches_in_ram():
    # forcing the out-of-core weight path must gather identically to a full RAM load.
    from flamediff.adapters._dcp_zerocopy import ZeroCopyShardedWeight

    path = _fixture()
    in_ram = ShardedTorchRecMCHAdapter(out_of_core_bytes=10**15).load(path)   # always RAM
    ooc = ShardedTorchRecMCHAdapter(out_of_core_bytes=0).load(path)           # force out-of-core
    t_ram = in_ram.embedding_tables["author_id_emb"]
    t_ooc = ooc.embedding_tables["author_id_emb"]
    assert isinstance(t_ooc._weights, ZeroCopyShardedWeight)  # zero-copy .distcp mmap, not RAM
    assert np.array_equal(t_ram.ids(), t_ooc.ids())
    ids = t_ram.ids()
    assert torch.equal(t_ram.gather(ids), t_ooc.gather(ids))  # zero-copy gather == RAM gather


@pytest.mark.integration
def test_out_of_core_falls_back_to_scratch(monkeypatch):
    # if the .distcp framing is ever unexpected, zero-copy must degrade to the scratch reassembly.
    from flamediff.adapters._dcp_zerocopy import ZeroCopyShardedWeight

    path = _fixture()
    ref = ShardedTorchRecMCHAdapter(out_of_core_bytes=10**15).load(path)  # RAM reference

    def boom(*a, **k):
        raise ValueError("simulated unexpected .distcp framing")

    monkeypatch.setattr("flamediff.adapters._dcp_zerocopy.open_zero_copy_weight", boom)
    fb = ShardedTorchRecMCHAdapter(out_of_core_bytes=0).load(path)  # zero-copy fails -> scratch
    t_ref, t_fb = ref.embedding_tables["author_id_emb"], fb.embedding_tables["author_id_emb"]
    assert not isinstance(t_fb._weights, ZeroCopyShardedWeight)  # took the scratch path
    ids = t_ref.ids()
    assert torch.equal(t_ref.gather(ids), t_fb.gather(ids))      # still identical to RAM
