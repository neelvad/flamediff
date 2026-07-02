"""Stage 1+2 of the detection layer: run consecutive pairwise diffs over a trajectory and
reduce each step to a vector of scalar features -> per-(table, metric) time series."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flamediff.diff import diff_checkpoints
from flamediff.types import Checkpoint, CheckpointDiff, EmbeddingTableDiff

# Count ids whose |freq_resid| clears this bar -> a stable (count-style) detector feature.
FREQ_RESID_BAR = 4.0


@dataclass
class MetricSeries:
    """One scalar metric for one table, over the trajectory."""

    table: str
    metric: str
    index: np.ndarray  # 0..T-1, the diff's position (maps to TrajectoryDiff.diffs[index])
    step: np.ndarray   # global step of the current checkpoint, or index if unknown
    value: np.ndarray


@dataclass
class TrajectoryDiff:
    steps: list             # global step per checkpoint (len = n_checkpoints)
    diffs: list             # CheckpointDiff per consecutive pair (len = n_checkpoints - 1)
    series: dict            # (table, metric) -> MetricSeries

    def series_for(self, table: str, metric: str) -> MetricSeries:
        return self.series[(table, metric)]


def _emb_features(td: EmbeddingTableDiff) -> dict[str, float]:
    dn, fr = td.delta_norm, td.freq_resid
    return {
        # churn (count-style, robust)
        "inserted_rate": td.n_inserted / max(td.n_cur, 1),
        "evicted_rate": td.n_evicted / max(td.n_cur, 1),
        "slot_moved_rate": td.n_slot_moved / max(td.n_survivors, 1),
        "readmit_rate": td.n_readmitted / max(td.n_survivors, 1),
        "mover_frac": float((dn > 0).mean()) if dn.size else 0.0,
        # movement (point-style, sensitive)
        "delta_p50": float(np.median(dn)) if dn.size else 0.0,
        "delta_p95": float(np.percentile(dn, 95)) if dn.size else 0.0,
        "delta_max": float(dn.max()) if dn.size else 0.0,
        # scorer tails
        "freq_resid_max": float(fr.max()) if fr.size else 0.0,
        "n_freq_resid_hi": float((np.abs(fr) >= FREQ_RESID_BAR).sum()) if fr.size else 0.0,
        "frozen_max": float(td.frozen_score.max()) if td.frozen_score.size else 0.0,
        # geometry
        "effective_rank": td.geom_cur.effective_rank,
        "anisotropy": td.geom_cur.anisotropy,
        "mean_row_norm": td.geom_cur.mean_row_norm,
        # NOT series-ized: rank95 (integer-quantized) and subspace_overlap (pinned near 1.0) have
        # heavy-tailed standardized scores on clean runs, which inflates the pooled per-METHOD
        # null the calibration thresholds share and desensitizes every other metric. They live in
        # the diff result + the `flamediff rank` advisory; detection-ready versions need the
        # per-metric calibration noted in ARCHITECTURE.md.
    }


def step_features(cd: CheckpointDiff) -> dict[tuple[str, str], float]:
    """Collapse one CheckpointDiff to its scalar (table, metric) -> value features."""
    out: dict[tuple[str, str], float] = {}
    for name, td in cd.embedding_diffs.items():
        for metric, value in _emb_features(td).items():
            out[(name, metric)] = value
    for name, dd in cd.dense_diffs.items():
        out[(name, "rel_delta_norm")] = dd.rel_delta_norm
        out[(name, "cosine")] = dd.cosine
        out[(name, "effective_rank")] = dd.eff_rank_cur
    return out


def series_from_rows(rows: list) -> dict:
    """Assemble per-(table, metric) MetricSeries from precomputed feature rows -- each row is
    ``(index, step, {(table, metric): value})``. This lets the incremental watcher keep only the
    scalar step features per step (a few floats), never the diffs' per-id arrays.
    """
    index = np.array([r[0] for r in rows], dtype=np.int64)
    step_arr = np.array([r[1] for r in rows], dtype=np.int64)
    keys = sorted({k for _, _, feats in rows for k in feats})
    series = {}
    for table, metric in keys:
        vals = np.array([feats.get((table, metric), np.nan) for _, _, feats in rows],
                        dtype=np.float64)
        series[(table, metric)] = MetricSeries(table, metric, index, step_arr, vals)
    return series


def build_series(steps: list, diffs: list) -> dict:
    """Series from consecutive diffs + per-checkpoint steps (diff j spans checkpoints j and j+1)."""
    rows = [(j, steps[j + 1] if steps[j + 1] is not None else j, step_features(cd))
            for j, cd in enumerate(diffs)]
    return series_from_rows(rows)


def diff_trajectory(checkpoints: list[Checkpoint], *, keep_ids: bool = True) -> TrajectoryDiff:
    """Diff consecutive checkpoints and assemble per-(table, metric) time series."""
    if len(checkpoints) < 2:
        raise ValueError("need at least 2 checkpoints to form a trajectory")
    diffs = [diff_checkpoints(checkpoints[i - 1], checkpoints[i], keep_ids=keep_ids)
             for i in range(1, len(checkpoints))]
    steps = [c.step for c in checkpoints]
    return TrajectoryDiff(steps=steps, diffs=diffs, series=build_series(steps, diffs))
