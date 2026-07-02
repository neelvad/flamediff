"""The monitoring report: fuse detection (when/where drifted) with attribution (why) into one
consolidated, calibrated artifact -- the thing a user actually operates flamediff to produce.

`build_report(checkpoints)` runs the trajectory -> detection -> attribution pipeline and attaches a
*why* to every calibrated event, keyed by the metric's kind (churn / drift / geometry). The Report
renders to text, JSON, or markdown, and exposes `worst_severity()` for a CI gate.
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field

import numpy as np

from flamediff import detect as _detect
from flamediff.attribute import attribute_table
from flamediff.detect import Event, detect_trajectory
from flamediff.diff import diff_checkpoints
from flamediff.trajectory import TrajectoryDiff, diff_trajectory, series_from_rows, step_features
from flamediff.types import Checkpoint

_CHURN_METRICS = {"inserted_rate", "evicted_rate", "slot_moved_rate", "readmit_rate"}
_GEOMETRY_METRICS = {"effective_rank", "anisotropy", "mean_row_norm"}


def severity_of(e: Event) -> float:
    """Calibrated severity (x over the FPR bar) when calibration is active, else |raw score|."""
    return e.calibrated_severity if e.calibrated_severity is not None else abs(e.score)


@dataclass
class Why:
    kind: str            # "churn" | "drift" | "geometry" | "dense"
    text: str            # one-line human explanation
    detail: dict = field(default_factory=dict)  # structured, for JSON


@dataclass
class EnrichedEvent:
    event: Event
    why: Why

    @property
    def severity(self) -> float:
        return severity_of(self.event)

    def to_line(self) -> str:
        e = self.event
        return (f"  ● step {e.step}  {e.table}.{e.metric}  {self.severity:.1f}×  [{e.method}]\n"
                f"       why: {self.why.text}")

    def to_dict(self) -> dict:
        e = self.event
        return {
            "step": e.step, "index": e.index, "table": e.table, "metric": e.metric,
            "method": e.method, "direction": e.direction,
            "severity": round(self.severity, 3), "raw_score": round(e.score, 3),
            "value": round(e.value, 5), "baseline": round(e.baseline, 5),
            "why": {"kind": self.why.kind, "text": self.why.text, **self.why.detail},
        }


@dataclass
class Incident:
    """One underlying training incident: the events that fired together in a short index window,
    across tables / metrics / detectors, headlined by the strongest signal. One real cause (a data
    spike, a regime change) typically fires many series at once -- reporting the flat event list
    reads as dozens of anomalies when it is really a handful of incidents."""

    events: list[EnrichedEvent]  # severity desc; events[0] is the headline

    @property
    def headline(self) -> EnrichedEvent:
        return self.events[0]

    @property
    def severity(self) -> float:
        return self.headline.severity

    @property
    def steps(self) -> list[int]:
        return sorted({ee.event.step for ee in self.events})

    @property
    def tables(self) -> list[str]:
        return sorted({ee.event.table for ee in self.events})

    def span(self) -> str:
        s = self.steps
        return f"step {s[0]}" if len(s) == 1 else f"steps {s[0]}–{s[-1]}"

    def to_lines(self) -> list[str]:
        head = (f"  ▌ {self.span()}  worst {self.severity:.1f}×  "
                f"({len(self.events)} signals, {len(self.tables)} tables)")
        lines = [head] + ["  " + ln for ln in self.headline.to_line().split("\n")]
        rest = self.events[1:]
        if rest:
            shown = ", ".join(f"{ee.event.table}.{ee.event.metric} {ee.severity:.1f}×"
                              for ee in rest[:3])
            more = f", +{len(rest) - 3} more" if len(rest) > 3 else ""
            lines.append(f"         also: {shown}{more}")
        return lines

    def to_dict(self, event_pos: dict) -> dict:
        """JSON form; events are referenced by their position in the report's flat event list."""
        return {
            "steps": self.steps, "tables": self.tables,
            "severity": round(self.severity, 3), "n_events": len(self.events),
            "events": [event_pos[id(ee)] for ee in self.events],
        }


def group_incidents(events: list[EnrichedEvent], *, gap: int = 1) -> list[Incident]:
    """Cluster events into incidents by diff-index adjacency: chains of events whose indices are
    within ``gap`` of each other, across tables / metrics / detectors (PH and PELT flag the same
    cause a step or two after robust_z). Incidents and their events are sorted by severity desc."""
    if not events:
        return []
    by_index = sorted(events, key=lambda ee: ee.event.index)
    clusters = [[by_index[0]]]
    for ee in by_index[1:]:
        if ee.event.index - clusters[-1][-1].event.index <= gap:
            clusters[-1].append(ee)
        else:
            clusters.append([ee])
    incidents = [Incident(sorted(c, key=lambda ee: ee.severity, reverse=True)) for c in clusters]
    incidents.sort(key=lambda inc: inc.severity, reverse=True)
    return incidents


@dataclass
class Report:
    run: str
    n_checkpoints: int
    tables: list[str]
    calibration: str
    target_fpr: float | None
    min_severity: float
    events: list[EnrichedEvent]  # sorted by severity desc
    series: list = field(default_factory=list)  # [{table, metric, steps, values}] for the charts

    def worst_severity(self) -> float:
        return max((e.severity for e in self.events), default=0.0)

    def incidents(self, *, gap: int = 1) -> list[Incident]:
        return group_incidents(self.events, gap=gap)

    def to_dict(self) -> dict:
        incidents = self.incidents()
        event_pos = {id(ee): i for i, ee in enumerate(self.events)}
        return {
            "run": self.run, "n_checkpoints": self.n_checkpoints, "tables": self.tables,
            "calibration": self.calibration, "target_fpr": self.target_fpr,
            "min_severity": self.min_severity, "n_events": len(self.events),
            "n_incidents": len(incidents),
            "worst_severity": round(self.worst_severity(), 3),
            "events": [e.to_dict() for e in self.events],
            "incidents": [inc.to_dict(event_pos) for inc in incidents],
            "series": self.series,
        }

    def to_html(self, live_poll_ms: int = 0) -> str:
        from flamediff._html import render_html
        return render_html(self.to_dict(), live_poll_ms)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_text(self) -> str:
        head = (f"flamediff report — {self.run}  "
                f"({self.n_checkpoints} ckpts, {len(self.tables)} tables)  "
                f"cal={self.calibration}")
        lines = [head, ""]
        if not self.events:
            lines.append(f"no anomalies (calibrated severity ≥ {self.min_severity:g})")
            return "\n".join(lines)
        incidents = self.incidents()
        lines.append(f"INCIDENTS (calibrated severity ≥ {self.min_severity:g}, "
                     f"most severe first):")
        for inc in incidents:
            lines += inc.to_lines()
        worst = self.events[0]
        lines += ["", f"SUMMARY: {len(incidents)} incidents ({len(self.events)} signals) across "
                  f"{len(self.tables)} tables; "
                  f"worst step {worst.event.step} ({worst.event.table}.{worst.event.metric} "
                  f"{worst.severity:.1f}×)"]
        return "\n".join(lines)

    def to_markdown(self) -> str:
        incidents = self.incidents()
        out = [f"# flamediff report — `{self.run}`", "",
               f"- checkpoints: **{self.n_checkpoints}**, tables: {', '.join(self.tables)}",
               f"- calibration: {self.calibration}",
               f"- incidents (≥ {self.min_severity:g}×): **{len(incidents)}** "
               f"({len(self.events)} signals), worst **{self.worst_severity():.1f}×**", ""]
        for i, inc in enumerate(incidents, 1):
            out += [f"## incident {i} — {inc.span()}, worst {inc.severity:.1f}× "
                    f"({len(inc.events)} signals)", "",
                    "| step | table.metric | severity | method | why |",
                    "|---|---|---|---|---|"]
            for ee in inc.events:
                e = ee.event
                out.append(f"| {e.step} | `{e.table}.{e.metric}` | {ee.severity:.1f}× | "
                           f"{e.method} | {ee.why.text} |")
            out.append("")
        return "\n".join(out).rstrip("\n") + "\n"


def _why_ctx_entry(prev_table, cur_table, td) -> dict:
    """Reduce a table diff + its attribution to the small why-context the report needs: the churn
    summary (scalars) and the attribution summary (scalars) + top movers (ids) -- no per-id arrays.
    This is what the watcher retains per (table, step) instead of the full diff/attribution."""
    attr = attribute_table(prev_table, cur_table, td)
    return {"churn": td.churn_summary(),
            "attr": (attr.summary(), [int(m[0]) for m in attr.top_movers(10)])}


def _explain(e: Event, ctx: dict | None) -> Why:
    """Build the *why* for an event from its reduced context (from ``_why_ctx_entry``), or None
    for a dense-tensor metric (no managed-collision diff for that table)."""
    if ctx is None:
        return Why("dense", f"{e.table} {e.direction} ({e.value:.3g} vs {e.baseline:.3g})")

    if e.metric in _CHURN_METRICS:
        c = ctx["churn"]
        return Why("churn",
                   f"churn {e.direction}: {c['inserted']} inserted / {c['evicted']} evicted / "
                   f"{c['readmitted']} re-admitted / {c['slot_moved']} slot-moved",
                   {"churn": c})

    summary, movers = ctx["attr"]
    g, p, i = summary["global"], summary["popularity_r2"], summary["aligned_residual"]
    if e.metric in _GEOMETRY_METRICS:
        return Why("geometry",
                   f"geometry shift ({e.value:.3g} vs {e.baseline:.3g}); "
                   f"global basis drift {g:.0%}, rotation_mag {summary['rotation_magnitude']:.2g}",
                   summary)
    return Why("drift",
               f"idiosyncratic drift (global {g:.0%}, popularity r²={p:.2f}, "
               f"residual {i:.0%}); movers {', '.join(str(m) for m in movers[:5])}",
               {**summary, "top_movers": movers})


def build_report(
    checkpoints: list[Checkpoint], *, run: str = "trajectory",
    table: str | None = None, min_severity: float = 1.0,
) -> Report:
    """Run the full pipeline and fuse calibrated events with their attribution."""
    traj = diff_trajectory(checkpoints)
    det = detect_trajectory(traj)
    cal = _detect._DEFAULT_CALIBRATION

    ctx_cache: dict = {}

    def ctx_of(tbl: str, index: int) -> dict | None:
        key = (tbl, index)
        if key not in ctx_cache:
            td = traj.diffs[index].embedding_diffs.get(tbl)
            ctx_cache[key] = None if td is None else _why_ctx_entry(
                checkpoints[index].embedding_tables[tbl],
                checkpoints[index + 1].embedding_tables[tbl], td)
        return ctx_cache[key]

    enriched = _enrich(det.events, ctx_of, min_severity=min_severity, table=table)
    series = _series_payload(traj.series, table)

    tables = sorted(set(checkpoints[0].embedding_tables) | set(checkpoints[0].dense_tensors))
    return Report(
        run=run,
        n_checkpoints=len(checkpoints),
        tables=[t for t in tables if t in checkpoints[0].embedding_tables] or tables,
        calibration=(cal.provenance or "loaded") if cal else "uncalibrated (raw |score|)",
        target_fpr=cal.target_fpr if cal else None,
        min_severity=min_severity,
        events=enriched,
        series=series,
    )


def _enrich(events: list[Event], ctx_of, *, min_severity: float,
            table: str | None) -> list[EnrichedEvent]:
    sel = [e for e in events
           if severity_of(e) >= min_severity and (table is None or e.table == table)]
    sel.sort(key=severity_of, reverse=True)
    return [EnrichedEvent(e, _explain(e, ctx_of(e.table, e.index))) for e in sel]


def _series_payload(series_map: dict, table: str | None) -> list:
    out = []
    for (tbl, metric), s in sorted(series_map.items()):
        if table is not None and tbl != table:
            continue
        vals = [None if not np.isfinite(v) else round(float(v), 6) for v in s.value]
        out.append({"table": tbl, "metric": metric,
                    "steps": [int(x) for x in s.step], "values": vals})
    return out


class Watcher:
    """Incremental drift watch over a run dir: `poll()` ingests any new ``ckpt_*`` and returns the
    events surfaced for the first time.

    Bounded in run length: per step it keeps only the scalar step-features (for the series) and a
    reduced per-(table, step) why-context (churn summary + attribution summary + <=10 mover ids) --
    never the diffs' or attribution's per-id arrays. Only the *last* checkpoint is held (to diff the
    next). So memory grows as O(steps x tables x small), and holds at most one checkpoint's weights.
    """

    def __init__(self, run_dir: str, *, table: str | None = None, min_severity: float = 1.0):
        self.run_dir = run_dir
        self.table = table
        self.min_severity = min_severity
        self._last: Checkpoint | None = None
        self._rows: list = []             # (index, step, step_features) per step -> the series
        self._why: dict = {}              # (table, index) -> reduced why-context (scalars + ids)
        self._events: list = []           # all current enriched events (for current_report)
        self._seen: set = set()           # event keys already surfaced
        self._processed: set = set()      # checkpoint paths already ingested

    def poll(self) -> list[EnrichedEvent]:
        from flamediff import load_checkpoint

        for path in sorted(glob.glob(f"{self.run_dir}/ckpt_*")):
            if path in self._processed:
                continue
            self._processed.add(path)
            ck = load_checkpoint(path)
            if self._last is None:
                self._last = ck
                continue
            idx = len(self._rows)
            cd = diff_checkpoints(self._last, ck)
            step = ck.step if ck.step is not None else idx
            self._rows.append((idx, step, step_features(cd)))
            for name, td in cd.embedding_diffs.items():
                self._why[(name, idx)] = _why_ctx_entry(
                    self._last.embedding_tables[name], ck.embedding_tables[name], td)
            self._last = ck
            # cd (and its per-id arrays) is now dropped -- only the reduced forms above are kept.

        if not self._rows:
            return []
        traj = TrajectoryDiff([], [], series_from_rows(self._rows))
        det = detect_trajectory(traj)
        self._events = _enrich(det.events, lambda t, i: self._why.get((t, i)),
                               min_severity=self.min_severity, table=self.table)
        fresh = []
        for ee in self._events:
            e = ee.event
            key = (e.index, e.table, e.metric, e.method)
            if key not in self._seen:
                self._seen.add(key)
                fresh.append(ee)
        return fresh

    def current_report(self) -> Report:
        """A full Report snapshot of the watcher's current state (for `flamediff serve`)."""
        series = _series_payload(series_from_rows(self._rows), self.table) if self._rows else []
        cal = _detect._DEFAULT_CALIBRATION
        tables = sorted(self._last.embedding_tables) if self._last is not None else []
        return Report(
            run=self.run_dir, n_checkpoints=len(self._processed), tables=tables,
            calibration=(cal.provenance or "loaded") if cal else "uncalibrated (raw |score|)",
            target_fpr=cal.target_fpr if cal else None, min_severity=self.min_severity,
            events=list(self._events), series=series,
        )
