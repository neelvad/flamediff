"""Stage 3: detect anomalies in each per-(table, metric) series, judged against the series'
own trailing/overall history (its noise floor).

Methods:
  - robust_z     : causal trailing-window control chart (point spikes)
  - page_hinkley : online sequential drift detector (persistent shifts)
  - pelt         : offline changepoint segmentation via ruptures (whole-series)

Severities are each method's natural standardized statistic; ranking mixes them, which a later
calibration pass / the view can normalize per method.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import ruptures as rpt

from flamediff.trajectory import MetricSeries, TrajectoryDiff

_EPS = 1e-12
_METHODS = ("robust_z", "page_hinkley", "pelt")


@dataclass
class Event:
    index: int
    step: int | None
    table: str
    metric: str
    value: float
    baseline: float
    score: float       # signed severity (robust-z units / standardized shift)
    direction: str     # "up" | "down"
    method: str
    calibrated_severity: float | None = None  # null p-value; set when calibration is active


@dataclass
class DetectionResult:
    events: list        # ranked by |score| desc
    series: dict        # (table, metric) -> MetricSeries, carried through for the view

    def top(self, k: int = 10) -> list:
        return self.events[:k]


def _robust_center_scale(x: np.ndarray) -> tuple[float, float]:
    finite = x[np.isfinite(x)]
    if finite.size < 2:
        return 0.0, 0.0
    med = float(np.median(finite))
    scale = 1.4826 * float(np.median(np.abs(finite - med)))
    if scale < _EPS:
        scale = float(finite.std())
    return med, scale


def _event(s: MetricSeries, i: int, baseline: float, score: float, method: str) -> Event:
    return Event(
        index=int(s.index[i]), step=int(s.step[i]), table=s.table, metric=s.metric,
        value=float(s.value[i]), baseline=baseline, score=float(score),
        direction="up" if score > 0 else "down", method=method,
    )


def _robust_z_events(s: MetricSeries, *, window: int, k: float, min_history: int) -> list[Event]:
    # Center on the trailing median (tracks level, so slow drift is PH/PELT's job) but scale by
    # the series' global robust scale -- a stable noise floor, unlike a tiny trailing-window MAD
    # that can collapse to ~0 and explode the z. (Global scale is offline; an online variant
    # would use an expanding estimate.)
    v = s.value
    _, gscale = _robust_center_scale(v)
    if gscale < _EPS:
        return []
    events = []
    for i in range(min_history, v.size):
        base = v[max(0, i - window):i]
        base = base[np.isfinite(base)]
        if base.size < 1 or not np.isfinite(v[i]):
            continue
        center = float(np.median(base))
        z = (v[i] - center) / gscale
        if abs(z) >= k:
            events.append(_event(s, i, center, float(z), "robust_z"))
    return events


def _page_hinkley_events(
    s: MetricSeries, *, lam: float, delta: float, min_history: int
) -> list[Event]:
    med, scale = _robust_center_scale(s.value)
    if scale < _EPS:
        return []
    x = (s.value - med) / scale
    events = []
    for sign in (1.0, -1.0):
        mean = cum = extreme = 0.0
        count = 0
        for i in range(x.size):
            xi = x[i] * sign
            if not np.isfinite(xi):
                continue
            count += 1
            mean += (xi - mean) / count
            cum += xi - mean - delta
            extreme = min(extreme, cum)
            if count > min_history and (cum - extreme) > lam:
                events.append(_event(s, i, med, (cum - extreme) * sign, "page_hinkley"))
                mean = cum = extreme = 0.0
                count = 0
    return events


def _pelt_events(s: MetricSeries, *, min_size: int, pen: float) -> list[Event]:
    v = s.value
    med, scale = _robust_center_scale(v)
    if scale < _EPS or v[np.isfinite(v)].size < 2 * min_size:
        return []
    x = np.nan_to_num((v - med) / scale, nan=0.0).reshape(-1, 1)
    bkps = rpt.Pelt(model="l2", min_size=min_size).fit(x).predict(pen=pen)
    events = []
    for b in bkps[:-1]:  # the final breakpoint is the series end, not a changepoint
        before = v[max(0, b - min_size):b]
        after = v[b:b + min_size]
        before = before[np.isfinite(before)]
        after = after[np.isfinite(after)]
        if before.size == 0 or after.size == 0:
            continue
        shift = (after.mean() - before.mean()) / scale
        events.append(_event(s, min(b, v.size - 1), float(before.mean()), float(shift), "pelt"))
    return events


def detect_series(
    s: MetricSeries, *, window: int = 6, k: float = 4.0, min_history: int = 3,
    ph_lam: float = 5.0, ph_delta: float = 0.0, pelt_min_size: int = 3, pelt_pen: float = 8.0,
    methods: tuple = _METHODS,
) -> list[Event]:
    events: list[Event] = []
    if "robust_z" in methods:
        events += _robust_z_events(s, window=window, k=k, min_history=min_history)
    if "page_hinkley" in methods:
        events += _page_hinkley_events(s, lam=ph_lam, delta=ph_delta, min_history=min_history)
    if "pelt" in methods:
        events += _pelt_events(s, min_size=pelt_min_size, pen=pelt_pen)
    return events


# --- calibration (optional; produced by scripts/calibrate.py) -----------------------------
_PERMISSIVE_DEFAULTS = {"k": 0.0, "ph_lam": 0.3, "pelt_pen": 1.0}


class Calibration:
    """FPR-calibrated thresholds + a severity->null-p-value map, from calibration.json."""

    def __init__(self, params: dict):
        self.target_fpr = params.get("target_fpr")
        self._methods = params.get("methods", {})

    def threshold(self, method: str) -> float:
        return self._methods.get(method, {}).get("threshold", 0.0)

    def calibrated(self, method: str, severity: float) -> float:
        """Severity in units of the method's FPR-calibrated threshold (>=1 clears the bar).
        Comparable across methods and tail-resolving, so ranking isn't skewed by raw scale."""
        t = self.threshold(method)
        return float(severity / t) if 0 < t < float("inf") else 0.0

    def p_value(self, method: str, severity: float) -> float:
        """Right-tail p-value vs the clean null (for display; saturates to 0 far in the tail)."""
        q = self._methods.get(method, {}).get("null_quantiles", [])
        if not q:
            return 1.0
        pos = int(np.searchsorted(np.asarray(q), severity, side="right"))
        return float(1.0 - pos / len(q))


def _load_default_calibration() -> Calibration | None:
    path = os.path.join(os.path.dirname(__file__), "calibration.json")
    if os.path.exists(path):
        with open(path) as fh:
            return Calibration(json.load(fh))
    return None


_DEFAULT_CALIBRATION = _load_default_calibration()


def detect_trajectory(traj: TrajectoryDiff, *, calibration=_DEFAULT_CALIBRATION,
                      **cfg) -> DetectionResult:
    """Detect over every series. With a calibration (default: the committed one), generate
    candidates permissively, keep those clearing each method's FPR-calibrated threshold, and
    rank by the calibrated severity (null p-value) -- comparable across methods. Without one,
    use the raw per-method thresholds in `cfg` and rank by |score|.
    """
    if calibration is None:
        events: list[Event] = []
        for series in traj.series.values():
            events.extend(detect_series(series, **cfg))
        events.sort(key=lambda e: -abs(e.score))
        return DetectionResult(events=events, series=traj.series)

    perm = {**_PERMISSIVE_DEFAULTS, **cfg}
    events = []
    for series in traj.series.values():
        for e in detect_series(series, **perm):
            cs = calibration.calibrated(e.method, abs(e.score))
            e.calibrated_severity = cs
            if cs >= 1.0:  # clears the FPR-calibrated bar
                events.append(e)
    events.sort(key=lambda e: -e.calibrated_severity)
    return DetectionResult(events=events, series=traj.series)
