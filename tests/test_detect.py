import numpy as np

from flamediff.detect import detect_series
from flamediff.trajectory import MetricSeries


def _series(values):
    v = np.asarray(values, dtype=float)
    idx = np.arange(v.size)
    return MetricSeries("t", "m", idx, idx, v)


def test_robust_z_flags_spike():
    rng = np.random.default_rng(0)
    v = 1.0 + rng.normal(0, 0.01, 20)
    v[12] = 10.0
    evs = detect_series(_series(v), methods=("robust_z",), k=5.0)
    assert 12 in [e.index for e in evs]


def test_robust_z_quiet_on_clean_noise():
    rng = np.random.default_rng(1)
    v = 5.0 + rng.normal(0, 0.02, 50)
    evs = detect_series(_series(v), methods=("robust_z",), k=5.0)
    assert len(evs) == 0  # global-scale denominator -> no spurious flags on clean noise


def test_pelt_flags_level_shift():
    rng = np.random.default_rng(2)
    v = np.concatenate([np.zeros(15), np.full(15, 5.0)]) + rng.normal(0, 0.02, 30)
    evs = detect_series(_series(v), methods=("pelt",), pelt_pen=5.0, pelt_min_size=3)
    assert any(12 <= e.index <= 18 for e in evs)
    assert all(e.method == "pelt" for e in evs)


def test_page_hinkley_flags_drift():
    rng = np.random.default_rng(3)
    v = np.concatenate([np.zeros(15), np.full(15, 3.0)]) + rng.normal(0, 0.05, 30)
    evs = detect_series(_series(v), methods=("page_hinkley",), ph_lam=3.0)
    assert any(e.index >= 15 for e in evs)


def test_no_methods_no_events():
    s = _series(np.arange(20.0))
    assert detect_series(s, methods=()) == []
