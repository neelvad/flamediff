import numpy as np

from flamediff.calibrate import MutationSpec, run_calibration, synthetic_trajectory


def test_null_fpr_is_honest():
    # the operating threshold at a target FPR should actually give ~that FPR on clean runs.
    clean = [synthetic_trajectory(s) for s in range(20)]
    res = run_calibration(clean, [MutationSpec("scramble", 4.0)], trials_per_cell=4)
    for method in ("robust_z", "page_hinkley"):
        t = res.operating_threshold(method, 0.2)
        fpr = float(np.mean(res.null.run_max[method] >= t))
        assert fpr <= 0.35  # ~0.2 target (in-sample quantile), comfortably not a storm


def test_power_increases_with_magnitude():
    clean = [synthetic_trajectory(s) for s in range(8)]
    battery = [MutationSpec("scramble", m) for m in (0.25, 1.0, 8.0)]
    res = run_calibration(clean, battery, trials_per_cell=20)
    t = res.operating_threshold("robust_z", 0.05)
    tprs = [res.tpr("scramble", m, "robust_z", t) for m in (0.25, 1.0, 8.0)]
    assert tprs[-1] >= tprs[0] - 0.1   # non-decreasing-ish
    assert tprs[-1] >= 0.8             # a big scramble is reliably caught


def test_freeze_is_detectable_if_weaker():
    clean = [synthetic_trajectory(s) for s in range(8)]
    res = run_calibration(clean, [MutationSpec("freeze", 1.0)], trials_per_cell=20)
    t = res.operating_threshold("robust_z", 0.05)
    assert res.tpr("freeze", 1.0, "robust_z", t) > 0.3


def test_calibrated_severity_is_comparable_across_methods():
    from flamediff.detect import Calibration

    calib = Calibration({"methods": {
        "robust_z": {"threshold": 6.0}, "page_hinkley": {"threshold": 20.0},
    }})
    # raw 30 (PH) vs raw 12 (robust_z): PH is bigger raw, but robust_z is more over its bar.
    assert calib.calibrated("page_hinkley", 30.0) < calib.calibrated("robust_z", 12.0)
