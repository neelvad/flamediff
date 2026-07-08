"""Calibration sweep: characterize detection power (TPR vs effect size) and the null
(clean) severity distributions, all offline / in-memory.

Method (two phases, score-once / threshold-analytically):
  - Null: run the detectors permissively on stationary CLEAN trajectories; pool the severities
    (for severity->p-value normalization) and the per-run max severity (for per-run FPR).
  - Power: inject a known corruption at a random step of a clean trajectory and record the max
    severity at the injected location, per method. TPR at any threshold is then a tail fraction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from flamediff.detect import detect_trajectory
from flamediff.mutate import set_rows
from flamediff.trajectory import TrajectoryDiff, diff_trajectory
from flamediff.types import Checkpoint, InMemoryTable

METHODS = ("robust_z", "page_hinkley", "pelt")
# permissive thresholds so every detector emits a severity at every candidate location
_PERMISSIVE = dict(k=0.0, window=6, min_history=3, ph_lam=0.3, pelt_min_size=3, pelt_pen=1.0)


@dataclass
class MutationSpec:
    kind: str            # scramble | zero | noise | freeze
    magnitude: float
    n_ids: int = 16
    persistent: bool = False

    @property
    def label(self) -> str:
        return f"{self.kind}{'/persist' if self.persistent else ''}"


@dataclass
class TrialResult:
    spec: MutationSpec
    step: int
    severity: dict           # method -> max |severity| at the injected location


@dataclass
class NullResult:
    pooled: dict             # method -> np.ndarray of |severity| on clean data
    run_max: dict            # method -> np.ndarray of per-clean-run max |severity|


@dataclass
class CalibrationResult:
    trials: list = field(default_factory=list)
    null: NullResult | None = None

    def operating_threshold(self, method: str, target_fpr: float) -> float:
        rm = self.null.run_max.get(method, np.array([])) if self.null else np.array([])
        if rm.size == 0:
            return float("inf")
        return float(np.quantile(rm, 1.0 - target_fpr))

    def normalize(self, method: str, severity: float) -> float:
        """Right-tail p-value of a severity against the clean null pool."""
        pool = self.null.pooled.get(method, np.array([])) if self.null else np.array([])
        if pool.size == 0:
            return 1.0
        return float(np.mean(pool >= severity))

    def tpr(self, spec_label: str, magnitude: float, method: str, threshold: float) -> float:
        sev = [t.severity.get(method, 0.0) for t in self.trials
               if t.spec.label == spec_label and t.spec.magnitude == magnitude]
        return float(np.mean(np.array(sev) >= threshold)) if sev else float("nan")

    def magnitudes(self, spec_label: str) -> list:
        return sorted({t.spec.magnitude for t in self.trials if t.spec.label == spec_label})

    def labels(self) -> list:
        return sorted({t.spec.label for t in self.trials})

    def min_detectable_effect(self, spec_label, method, *, tpr_bar=0.9, target_fpr=0.05):
        t = self.operating_threshold(method, target_fpr)
        for m in self.magnitudes(spec_label):
            if self.tpr(spec_label, m, method, t) >= tpr_bar:
                return m
        return float("inf")


# --- synthetic clean data -----------------------------------------------------------------
def synthetic_trajectory(seed: int, *, n: int = 48, dim: int = 8, steps: int = 20,
                         drift: float = 0.01, table: str = "emb") -> list[Checkpoint]:
    """A stationary clean trajectory: heterogeneous per-id frequency drives both movement and
    update counts, so the movement/scorer/geometry series have realistic, stable noise floors."""
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    ids = np.arange(1, n + 1, dtype=np.int64)
    slots = np.arange(n)
    freq = rng.integers(1, 40, size=n).astype(np.int64)          # per-id update frequency
    scale = torch.from_numpy((freq / freq.mean()).astype(np.float32)).unsqueeze(1)
    W = torch.randn(n, dim, generator=g) * 0.2
    counts = np.zeros(n, dtype=np.int64)
    cks = []
    for i in range(steps):
        if i > 0:
            W = W + drift * torch.randn(n, dim, generator=g) * scale  # movement ~ frequency
        counts = counts + freq
        t = InMemoryTable(table, n, ids, slots, W.clone(), counts.copy())
        cks.append(Checkpoint(path=f"syn{seed}-{i}", step=i, embedding_tables={table: t}))
    return cks


# --- injection ----------------------------------------------------------------------------
def _choose_ids(table: InMemoryTable, spec: MutationSpec, g: torch.Generator) -> np.ndarray:
    ids = table.ids()
    k = min(spec.n_ids, ids.size)
    if spec.kind == "freeze":  # freeze the most-trained ids (a meaningful "didn't move")
        counts = table.counts(ids)
        order = np.argsort(-counts) if counts is not None else np.arange(ids.size)
        return np.sort(ids[order[:k]])
    pick = torch.randperm(ids.size, generator=g)[:k].numpy()
    return np.sort(ids[pick])


def inject(checkpoints, spec: MutationSpec, step: int, *, table: str = "emb", seed: int = 0):
    """Inject a corruption at `step` (transient: that checkpoint; persistent: step..end)."""
    cks = list(checkpoints)
    g = torch.Generator().manual_seed(seed)
    base = cks[step].embedding_tables[table]
    chosen = _choose_ids(base, spec, g)
    typical = float(base.gather(base.ids()).float().norm(dim=1).median()) or 1.0

    if spec.kind == "freeze":
        target = cks[step - 1].embedding_tables[table].gather(chosen).float().clone()
    else:
        orig = base.gather(chosen).float()
        if spec.kind == "zero":
            target = torch.zeros_like(orig)
        elif spec.kind == "scramble":
            r = torch.randn(len(chosen), base.dim, generator=g)
            target = r / (r.norm(dim=1, keepdim=True) + 1e-12) * (spec.magnitude * typical)
        elif spec.kind == "noise":
            noise = torch.randn(len(chosen), base.dim, generator=g)
            target = orig + noise * (spec.magnitude * typical)
        else:
            raise ValueError(f"unknown kind {spec.kind!r}")

    end = len(cks) if spec.persistent else step + 1
    for j in range(step, end):
        t = cks[j].embedding_tables[table]
        res = np.intersect1d(chosen, t.ids())
        if res.size == 0:
            continue
        rows = target[np.searchsorted(chosen, res)]
        tables = dict(cks[j].embedding_tables)
        tables[table] = set_rows(t, res, rows)
        cks[j] = Checkpoint(path=cks[j].path, step=cks[j].step,
                            embedding_tables=tables, dense_tensors=cks[j].dense_tensors)
    return cks


# --- the sweep ----------------------------------------------------------------------------
def _events_permissive(traj: TrajectoryDiff) -> list:
    return detect_trajectory(traj, calibration=None, **_PERMISSIVE).events


def _loc_severity(events, indices: set, table: str) -> dict:
    out = {m: 0.0 for m in METHODS}
    for e in events:
        if e.index in indices and e.table == table:
            out[e.method] = max(out[e.method], abs(e.score))
    return out


def characterize_null(clean_trajectories, *, table: str = "emb") -> NullResult:
    pooled: dict[str, list[float]] = {m: [] for m in METHODS}
    run_max: dict[str, list[float]] = {m: [] for m in METHODS}
    for cks in clean_trajectories:
        events = _events_permissive(diff_trajectory(cks))
        per_run = {m: 0.0 for m in METHODS}
        for e in events:
            pooled[e.method].append(abs(e.score))
            per_run[e.method] = max(per_run[e.method], abs(e.score))
        for m in METHODS:
            run_max[m].append(per_run[m])
    return NullResult(
        pooled={m: np.array(v) for m, v in pooled.items()},
        run_max={m: np.array(v) for m, v in run_max.items()},
    )


def run_power(clean_trajectories, battery, *, trials_per_cell: int, table: str = "emb",
              seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    trials = []
    for spec in battery:
        for _ in range(trials_per_cell):
            cks = clean_trajectories[rng.integers(len(clean_trajectories))]
            step = int(rng.integers(2, len(cks) - 1))  # avoid warmup / final edges
            indices = {step - 1} | ({step} if not spec.persistent else
                                    set(range(step - 1, len(cks) - 1)))
            mutated = inject(cks, spec, step, table=table, seed=int(rng.integers(1 << 31)))
            events = _events_permissive(diff_trajectory(mutated))
            trials.append(TrialResult(spec, step, _loc_severity(events, indices, table)))
    return trials


def run_calibration(clean_trajectories, battery, *, trials_per_cell: int = 40,
                    table: str = "emb", seed: int = 0) -> CalibrationResult:
    null = characterize_null(clean_trajectories, table=table)
    trials = run_power(clean_trajectories, battery, trials_per_cell=trials_per_cell,
                       table=table, seed=seed)
    return CalibrationResult(trials=trials, null=null)


def derive_params(result: CalibrationResult, *, target_fpr: float = 0.05) -> dict:
    """Compact JSON the detector loads: per-method FPR-calibrated threshold."""
    methods = {m: {"threshold": result.operating_threshold(m, target_fpr)} for m in METHODS}
    return {
        "provenance": ("SYNTHETIC (stationary trajectories, dim=8, n=48, steps=20) -- thresholds "
                       "are scale-transferable but the per-run-max null reflects the synthetic "
                       "look-elsewhere count; regenerate for your data via scripts/calibrate.py"),
        "target_fpr": target_fpr,
        "methods": methods,
    }
