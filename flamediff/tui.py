"""A terminal UI for browsing detected trajectory anomalies (textual).

Launch:  uv run flamediff-tui [run_dir]

Left: the ranked events (severity colour-coded). Right: the selected metric's series as a
sparkline (flagged step highlighted) over a drill-down to that step's top movers / frozen ids.
Arrow keys navigate, [m] cycles the method filter, [q] quits.
"""
from __future__ import annotations

import numpy as np
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Static

_BLOCKS = "▁▂▃▄▅▆▇█"
_THEME = "catppuccin-mocha"


def _sparkline(values: np.ndarray, mark_pos: int, theme) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ""
    lo, hi = float(finite.min()), float(finite.max())
    rng = hi - lo or 1.0
    out = []
    for i, val in enumerate(values):
        ch = " " if not np.isfinite(val) else _BLOCKS[int((val - lo) / rng * (len(_BLOCKS) - 1))]
        out.append(f"[b {theme.error}]{ch}[/]" if i == mark_pos else f"[{theme.secondary}]{ch}[/]")
    return "".join(out)


class FlamediffTUI(App):
    TITLE = "flamediff"
    CSS = """
    Horizontal { height: 1fr; padding: 1; }
    #events { width: 2fr; border: round $primary; padding: 0 1; border-title-color: $primary; }
    #right { width: 3fr; }
    #chart { height: 1fr; border: round $secondary; padding: 1; margin-left: 1;
             border-title-color: $secondary; }
    #drill { height: 1fr; border: round $accent; padding: 1; margin-left: 1;
             border-title-color: $accent; }
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
        self.theme = _THEME
        table = self.query_one("#events", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.border_title = "Events"
        table.add_columns("idx", "step", "table · metric", "sev", "method", "dir")
        self.query_one("#chart", Static).border_title = "Series"
        self.query_one("#drill", Static).border_title = "Drill-down"
        self.sub_title = f"{len(self.traj.diffs)} diffs · {len(self._all_events)} events"
        self._fill_table()

    def _sev_text(self, e) -> Text:
        # show what we actually rank by: the calibrated severity (x over the FPR bar). Fall back
        # to the raw score only when no calibration is loaded.
        th = self.current_theme
        cs = e.calibrated_severity
        if cs is not None:
            color = th.error if cs >= 8 else th.warning if cs >= 3 else th.success
            return Text(f"{cs:.1f}x", style=f"bold {color}")
        a = abs(e.score)
        color = th.error if a >= 12 else th.warning if a >= 5 else th.success
        return Text(f"{e.score:+.1f}", style=f"bold {color}")

    def _dir_text(self, direction: str) -> Text:
        th = self.current_theme
        return Text(direction, style=(th.error if direction == "up" else th.accent) or "")

    def _fill_table(self) -> None:
        table = self.query_one("#events", DataTable)
        table.clear()
        for e in self.events:
            table.add_row(str(e.index), str(e.step), f"{e.table} · {e.metric}",
                          self._sev_text(e), e.method, self._dir_text(e.direction))
        if self.events:
            self._show(0)
        else:
            self.query_one("#chart", Static).update("[dim](no events for this filter)[/]")
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
        th = self.current_theme
        s = self.traj.series.get((e.table, e.metric))
        if s is None:
            return f"{e.table}.{e.metric}: (no series)"
        v = s.value
        match = np.where(s.index == e.index)[0]
        pos = int(match[0]) if match.size else -1
        cal = f"[b {th.error}]cal {e.calibrated_severity:.1f}x[/]   " \
            if e.calibrated_severity is not None else ""
        head = (f"[b {th.primary}]{e.table}[/] · [b {th.secondary}]{e.metric}[/]\n"
                f"value={e.value:+.4f}   baseline={e.baseline:+.4f}\n"
                f"{cal}[{th.accent}]raw {e.score:+.2f} ({e.method} {e.direction})[/]")
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        foot = (f"[{th.panel}]min {lo:+.4f}  max {hi:+.4f}  steps {v.size}[/]   "
                f"[{th.error}]●[/] flagged @ idx {e.index}")
        return f"{head}\n\n{_sparkline(v, pos, th)}\n{foot}"

    def _drill(self, e) -> str:
        th = self.current_theme
        cd = self.traj.diffs[e.index]
        out = [f"[b {th.accent}]drill-down @ idx {e.index}[/] [{th.panel}](step {e.step})[/]"]
        if e.table not in cd.embedding_diffs:
            out.append("[dim](dense tensor — no per-id drill-down)[/]")
            return "\n".join(out)
        td = cd.embedding_diffs[e.table]

        def row(m, sc, dn, dc):
            return f"  id {m:7d}  [b]{sc:+6.2f}[/]  ‖Δ‖ {dn:.4f}  [{th.panel}]n={dc}[/]"

        out.append(f"[b {th.success}]top movers (freq_resid)[/]")
        out += [row(*x) for x in td.top_movers(6, by="freq_resid")]
        out.append(f"[b {th.warning}]top frozen (trained, didn't move)[/]")
        out += [row(*x) for x in td.top_frozen(4)]
        return "\n".join(out)


def main() -> None:
    import glob
    import sys

    from flamediff import detect_trajectory, diff_trajectory, load_checkpoint

    run = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("fixtures/run_*"))[-1]
    paths = sorted(glob.glob(f"{run}/ckpt_*"))
    print(f"loading {len(paths)} checkpoints from {run} ...")
    traj = diff_trajectory([load_checkpoint(p) for p in paths])
    FlamediffTUI(traj, detect_trajectory(traj)).run()
