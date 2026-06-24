import numpy as np
import torch

from flamediff.trajectory import diff_trajectory
from flamediff.types import Checkpoint


def _ckpt(make_table, step, weights):
    n = weights.shape[0]
    table = make_table(name="t", ids=np.arange(1, n + 1), slots=np.arange(n),
                       counts=np.full(n, step + 1, dtype=np.int64), weights=weights, num_slots=n)
    return Checkpoint(path=f"c{step}", step=step, embedding_tables={"t": table})


def test_diff_trajectory_builds_series(make_table):
    n, dim = 20, 4
    g = torch.Generator().manual_seed(0)
    w = [torch.randn(n, dim, generator=g) * 0.1]
    for _ in range(2):
        w.append(w[-1] + 0.01 * torch.randn(n, dim, generator=g))
    cks = [_ckpt(make_table, i, w[i].clone()) for i in range(3)]

    traj = diff_trajectory(cks)

    assert len(traj.diffs) == 2                      # 3 checkpoints -> 2 consecutive diffs
    s = traj.series_for("t", "delta_p95")
    assert s.value.size == 2
    assert list(s.step) == [1, 2]                    # series step = current checkpoint's step
    assert ("t", "freq_resid_max") in traj.series
    assert ("t", "effective_rank") in traj.series
