"""Run the calibration sweep, print the report, and write flamediff/calibration.json.

    uv run scripts/calibrate.py
"""
from __future__ import annotations

import json
import os

import numpy as np

from flamediff.calibrate import (
    MutationSpec,
    derive_params,
    run_calibration,
    synthetic_trajectory,
)

TARGET_FPR = 0.05
N_CLEAN = 24
TRIALS = 50
MAGS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
METHODS = ("robust_z", "page_hinkley", "pelt")


def battery():
    specs = []
    for persistent in (False, True):
        specs += [MutationSpec("scramble", m, persistent=persistent) for m in MAGS]
        specs.append(MutationSpec("freeze", 1.0, persistent=persistent))
        specs.append(MutationSpec("zero", 1.0, persistent=persistent))
    return specs


def main() -> None:
    clean = [synthetic_trajectory(seed) for seed in range(N_CLEAN)]
    result = run_calibration(clean, battery(), trials_per_cell=TRIALS, seed=0)

    print(f"=== calibration (target per-run FPR = {TARGET_FPR}) ===")
    print("operating thresholds & null severity (per detector):")
    for m in METHODS:
        t = result.operating_threshold(m, TARGET_FPR)
        pool = result.null.pooled[m]
        med = float(np.median(pool)) if pool.size else float("nan")
        p99 = float(np.quantile(pool, 0.99)) if pool.size else float("nan")
        print(f"  {m:13s} thresh={t:8.2f}   null med={med:6.2f}  p99={p99:8.2f}  n={pool.size}")

    print("\npower = TPR at the operating point:")
    for label in result.labels():
        print(f"  [{label}]")
        for mag in result.magnitudes(label):
            cells = []
            for m in METHODS:
                t = result.operating_threshold(m, TARGET_FPR)
                cells.append(f"{m[:4]}={result.tpr(label, mag, m, t):.2f}")
            print(f"      mag={mag:<5g} " + " ".join(cells))
        for m in METHODS:
            mde = result.min_detectable_effect(label, m, target_fpr=TARGET_FPR)
            if mde != float("inf"):
                print(f"      min-detectable[{m}] = {mde:g}x row-norm")

    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "flamediff", "calibration.json")
    with open(out, "w") as fh:
        json.dump(derive_params(result, target_fpr=TARGET_FPR), fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
