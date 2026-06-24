"""A terminal UI for browsing detected trajectory anomalies (textual).

Launch:  uv run flamediff-tui [run_dir]

Left: the ranked events. Right: the selected metric's series as a sparkline (flagged step in
red) over the drill-down (top movers / frozen ids for that step). Arrow keys navigate, [m]
cycles the method filter, [q] quits.
"""
from __future__ import annotations

import numpy as np
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Static

_BLOCKS = "▁▂▃▄▅▆▇█"


def _sparkline(values: np.ndarray, mark_pos: int) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ""
    lo, hi = float(finite.min()), float(finite.max())
    rng = hi - lo or 1.0
    chars = []
    for i, val in enumerate(values):
        ch = " " if not np.isfinite(val) else _BLOCKS[int((val - lo) / rng * (len(_BLOCKS) - 1))]
        chars.append(f"[red]{ch}[/red]" if i == mark_pos else ch)
    return "".join(chars)


class FlamediffTUI(App):
    TITLE = "flamediff"
    CSS = """
    Horizontal { height: 1fr; }
    #events { width: 2fr; }
    #right { width: 3fr; }
    #chart { height: 1fr; border: round $accent; padding: 0 1; }
    #drill { height: 1fr; border: round $accent; padding: 0 1; }
    """
    BINDINGS = [("q", "quit", "Quit"), ("m", "cycle_method", "Method filter")]

    def __init__(self, traj, result):
        super().__init__()
        self.traj = traj
        self._all_events = result.events
        self._methods = ["all", "robust_z", "page_hinkley", "pelt"]
        self._method_i = 0
        self.events = list(self._all_events)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield DataTable(id="events")
            with Vertical(id="right"):
                yield Static(id="chart")
                yield Static(id="drill")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#events", DataTable)
        table.cursor_type = "row"
        table.add_columns("idx", "step", "table.metric", "sev", "method", "dir")
        self.sub_title = f"{len(self.traj.diffs)} diffs · {len(self._all_events)} events"
        self._fill_table()

    def _fill_table(self) -> None:
        table = self.query_one("#events", DataTable)
        table.clear()
        for e in self.events:
            table.add_row(str(e.index), str(e.step), f"{e.table}.{e.metric}",
                          f"{e.score:+.1f}", e.method, e.direction)
        if self.events:
            self._show(0)
        else:
            self.query_one("#chart", Static).update("(no events for this filter)")
            self.query_one("#drill", Static).update("")

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if 0 <= event.cursor_row < len(self.events):
            self._show(event.cursor_row)

    def action_cycle_method(self) -> None:
        self._method_i = (self._method_i + 1) % len(self._methods)
        method = self._methods[self._method_i]
        self.events = (list(self._all_events) if method == "all"
                       else [e for e in self._all_events if e.method == method])
        self.sub_title = f"{len(self.traj.diffs)} diffs · {len(self.events)} events [{method}]"
        self._fill_table()

    def _show(self, row: int) -> None:
        e = self.events[row]
        self.query_one("#chart", Static).update(self._chart(e))
        self.query_one("#drill", Static).update(self._drill(e))

    def _chart(self, e) -> str:
        s = self.traj.series.get((e.table, e.metric))
        if s is None:
            return f"{e.table}.{e.metric}: (no series)"
        v = s.value
        match = np.where(s.index == e.index)[0]
        pos = int(match[0]) if match.size else -1
        head = (f"[b]{e.table}.{e.metric}[/b]\n"
                f"value={e.value:+.4f}  baseline={e.baseline:+.4f}  "
                f"sev={e.score:+.2f} ({e.method} {e.direction})")
        body = (f"{_sparkline(v, pos)}\n"
                f"min={np.nanmin(v):+.4f}  max={np.nanmax(v):+.4f}  steps={v.size}   "
                f"[red]●[/red] flagged @ idx {e.index}")
        return f"{head}\n\n{body}"

    def _drill(self, e) -> str:
        cd = self.traj.diffs[e.index]
        lines = [f"[b]drill-down @ idx {e.index} (step {e.step})[/b]"]
        if e.table in cd.embedding_diffs:
            td = cd.embedding_diffs[e.table]
            lines.append("[b]top movers (freq_resid):[/b]")
            lines += [f"  id {m:7d}  resid {sc:+6.2f}  ‖Δ‖ {dn:.4f}  dcount {dc}"
                      for m, sc, dn, dc in td.top_movers(6, by="freq_resid")]
            lines.append("[b]top frozen (trained, didn't move):[/b]")
            lines += [f"  id {m:7d}  frozen {sc:+5.2f}  ‖Δ‖ {dn:.4f}  dcount {dc}"
                      for m, sc, dn, dc in td.top_frozen(4)]
        else:
            lines.append("(dense tensor — no per-id drill-down)")
        return "\n".join(lines)


def main() -> None:
    import glob
    import sys

    from flamediff import detect_trajectory, diff_trajectory, load_checkpoint

    run = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("fixtures/run_*"))[-1]
    paths = sorted(glob.glob(f"{run}/ckpt_*"))
    print(f"loading {len(paths)} checkpoints from {run} ...")
    traj = diff_trajectory([load_checkpoint(p) for p in paths])
    FlamediffTUI(traj, detect_trajectory(traj)).run()
