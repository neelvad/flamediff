import asyncio

import numpy as np
import torch
from textual.widgets import DataTable, Static

from flamediff.detect import detect_trajectory
from flamediff.mutate import mutate_checkpoint
from flamediff.trajectory import diff_trajectory
from flamediff.tui import FlamediffTUI
from flamediff.types import Checkpoint


def _trajectory(make_table):
    n, dim, T = 30, 4, 10
    g = torch.Generator().manual_seed(0)
    w = [torch.randn(n, dim, generator=g) * 0.1]
    for _ in range(T - 1):
        w.append(w[-1] + 0.005 * torch.randn(n, dim, generator=g))

    def ck(i):
        t = make_table(name="t", ids=np.arange(1, n + 1), slots=np.arange(n),
                       counts=np.full(n, i + 1, dtype=np.int64), weights=w[i].clone(), num_slots=n)
        return Checkpoint(path=f"c{i}", step=i * 10, embedding_tables={"t": t})

    cks = [ck(i) for i in range(T)]
    cks[5], _ = mutate_checkpoint(cks[5], "t", kind="scramble",
                                  ids=np.arange(1, 11), magnitude=10.0)
    traj = diff_trajectory(cks)
    return traj, detect_trajectory(traj)


def test_tui_mounts_and_populates(make_table):
    traj, result = _trajectory(make_table)
    assert len(result.events) > 0  # the injected scramble must produce events to browse

    async def _run():
        app = FlamediffTUI(traj, result)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one("#events", DataTable)
            assert table.row_count == len(result.events)
            assert "value=" in str(app.query_one("#chart", Static).render())  # populated on mount
            await pilot.press("down")  # exercise row-highlight -> chart/drill update path
            await pilot.pause()
            assert "drill-down" in str(app.query_one("#drill", Static).render())

    asyncio.run(_run())
