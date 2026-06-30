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

from flamediff import detect as _detect
from flamediff.attribute import attribute_table
from flamediff.detect import Event, detect_trajectory
from flamediff.diff import diff_checkpoints
from flamediff.trajectory import TrajectoryDiff, build_series, diff_trajectory
from flamediff.types import Attribution, Checkpoint

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
class Report:
    run: str
    n_checkpoints: int
    tables: list[str]
    calibration: str
    target_fpr: float | None
    min_severity: float
    events: list[EnrichedEvent]  # sorted by severity desc

    def worst_severity(self) -> float:
        return max((e.severity for e in self.events), default=0.0)

    def to_dict(self) -> dict:
        return {
            "run": self.run, "n_checkpoints": self.n_checkpoints, "tables": self.tables,
            "calibration": self.calibration, "target_fpr": self.target_fpr,
            "min_severity": self.min_severity, "n_events": len(self.events),
            "worst_severity": round(self.worst_severity(), 3),
            "events": [e.to_dict() for e in self.events],
        }

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
        lines.append(f"ANOMALIES (calibrated severity ≥ {self.min_severity:g}, most severe first):")
        lines += [ee.to_line() for ee in self.events]
        worst = self.events[0]
        lines += ["", f"SUMMARY: {len(self.events)} anomalies across {len(self.tables)} tables; "
                  f"worst step {worst.event.step} ({worst.event.table}.{worst.event.metric} "
                  f"{worst.severity:.1f}×)"]
        return "\n".join(lines)

    def to_markdown(self) -> str:
        out = [f"# flamediff report — `{self.run}`", "",
               f"- checkpoints: **{self.n_checkpoints}**, tables: {', '.join(self.tables)}",
               f"- calibration: {self.calibration}",
               f"- anomalies (≥ {self.min_severity:g}×): **{len(self.events)}**, "
               f"worst **{self.worst_severity():.1f}×**", ""]
        if self.events:
            out += ["| step | table.metric | severity | method | why |",
                    "|---|---|---|---|---|"]
            for ee in self.events:
                e = ee.event
                out.append(f"| {e.step} | `{e.table}.{e.metric}` | {ee.severity:.1f}× | "
                           f"{e.method} | {ee.why.text} |")
        return "\n".join(out) + "\n"


def _explain(e: Event, checkpoints: list[Checkpoint], diffs: list,
             attr_cache: dict) -> Why:
    cd = diffs[e.index]
    td = cd.embedding_diffs.get(e.table)
    if td is None:  # a dense tensor metric
        return Why("dense", f"{e.table} {e.direction} ({e.value:.3g} vs {e.baseline:.3g})")

    if e.metric in _CHURN_METRICS:
        c = td.churn_summary()
        return Why("churn",
                   f"churn {e.direction}: {c['inserted']} inserted / {c['evicted']} evicted / "
                   f"{c['readmitted']} re-admitted / {c['slot_moved']} slot-moved",
                   {"churn": c})

    attr = _attribution(e.table, e.index, checkpoints, diffs, attr_cache)
    movers = ", ".join(str(m[0]) for m in attr.top_movers(5))
    g, p, i = attr.frac_global, attr.popularity_r2, attr.frac_aligned_residual
    if e.metric in _GEOMETRY_METRICS:
        return Why("geometry",
                   f"geometry shift ({e.value:.3g} vs {e.baseline:.3g}); "
                   f"global basis drift {g:.0%}, rotation_mag {attr.rotation_magnitude:.2g}",
                   attr.summary())
    return Why("drift",
               f"idiosyncratic drift (global {g:.0%}, popularity r²={p:.2f}, "
               f"residual {i:.0%}); movers {movers}",
               {**attr.summary(), "top_movers": [m[0] for m in attr.top_movers(10)]})


def _attribution(table: str, index: int, checkpoints: list[Checkpoint], diffs: list,
                 cache: dict) -> Attribution:
    key = (table, index)
    if key not in cache:
        td = diffs[index].embedding_diffs[table]
        prev = checkpoints[index].embedding_tables[table]
        cur = checkpoints[index + 1].embedding_tables[table]
        cache[key] = attribute_table(prev, cur, td)
    return cache[key]


def build_report(
    checkpoints: list[Checkpoint], *, run: str = "trajectory",
    table: str | None = None, min_severity: float = 1.0,
) -> Report:
    """Run the full pipeline and fuse calibrated events with their attribution."""
    traj = diff_trajectory(checkpoints)
    det = detect_trajectory(traj)
    cal = _detect._DEFAULT_CALIBRATION

    enriched = _enrich(det.events, checkpoints, traj.diffs,
                       min_severity=min_severity, table=table, attr_cache={})

    tables = sorted(set(checkpoints[0].embedding_tables) | set(checkpoints[0].dense_tensors))
    return Report(
        run=run,
        n_checkpoints=len(checkpoints),
        tables=[t for t in tables if t in checkpoints[0].embedding_tables] or tables,
        calibration=(cal.provenance or "loaded") if cal else "uncalibrated (raw |score|)",
        target_fpr=cal.target_fpr if cal else None,
        min_severity=min_severity,
        events=enriched,
    )


def _enrich(events: list[Event], checkpoints: list[Checkpoint], diffs: list, *,
            min_severity: float, table: str | None, attr_cache: dict) -> list[EnrichedEvent]:
    sel = [e for e in events
           if severity_of(e) >= min_severity and (table is None or e.table == table)]
    sel.sort(key=severity_of, reverse=True)
    return [EnrichedEvent(e, _explain(e, checkpoints, diffs, attr_cache)) for e in sel]


class Watcher:
    """Incremental drift watch over a run dir: `poll()` ingests any new ``ckpt_*`` and returns the
    events surfaced for the first time. Bounded memory -- it keeps the scalar diffs + per-step
    attribution and only the *last* checkpoint (to diff the next against), never the whole run.
    """

    def __init__(self, run_dir: str, *, table: str | None = None, min_severity: float = 1.0):
        self.run_dir = run_dir
        self.table = table
        self.min_severity = min_severity
        self._last: Checkpoint | None = None
        self._steps: list = []
        self._diffs: list = []
        self._attr_cache: dict = {}       # (table, index) -> Attribution, pre-filled per step
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
                self._last, self._steps = ck, [ck.step]
                continue
            idx = len(self._diffs)
            cd = diff_checkpoints(self._last, ck)
            self._diffs.append(cd)
            self._steps.append(ck.step)
            for name, td in cd.embedding_diffs.items():
                self._attr_cache[(name, idx)] = attribute_table(
                    self._last.embedding_tables[name], ck.embedding_tables[name], td)
            self._last = ck

        if not self._diffs:
            return []
        traj = TrajectoryDiff(self._steps, self._diffs, build_series(self._steps, self._diffs))
        det = detect_trajectory(traj)
        fresh = []
        for ee in _enrich(det.events, [], self._diffs, min_severity=self.min_severity,
                          table=self.table, attr_cache=self._attr_cache):
            e = ee.event
            key = (e.index, e.table, e.metric, e.method)
            if key not in self._seen:
                self._seen.add(key)
                fresh.append(ee)
        return fresh
