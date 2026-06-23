import numpy as np
import torch

from flamediff import load_checkpoint


def test_adapter_parses_map_counts_and_weights(write_ckpt):
    weight = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    path = write_ckpt("ckpt", {
        "a_emb": {"ids": [10, 20, 30], "slots": [4, 2, 0], "counts": [7, 3, 1], "weight": weight},
    }, step=42)
    ck = load_checkpoint(path)

    t = ck.embedding_tables["a_emb"]
    assert t.name == "a_emb"
    assert t.num_slots == 5
    assert set(t.ids().tolist()) == {10, 20, 30}
    assert t.slot_of(np.array([10, 20, 30])).tolist() == [4, 2, 0]
    assert t.counts(np.array([10, 20, 30])).tolist() == [7, 3, 1]
    # gather follows the id -> slot map: id 10 -> slot 4 -> weight[4]
    assert torch.allclose(t.gather(np.array([10]))[0], weight[4])
    assert ck.step == 42


def test_non_resident_id_returns_minus_one(write_ckpt):
    path = write_ckpt("c", {
        "a_emb": {"ids": [1, 2], "slots": [0, 1], "counts": [1, 1], "weight": torch.randn(2, 4)},
    })
    t = load_checkpoint(path).embedding_tables["a_emb"]
    assert t.slot_of(np.array([99])).tolist() == [-1]
