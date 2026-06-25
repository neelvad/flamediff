"""Run the full trajectory detection over a real run and print the ranked anomaly events.

    uv run scripts/detect_demo.py [run_dir]
"""
from __future__ import annotations

import glob
import sys

from flamediff import detect_trajectory, diff_trajectory, load_checkpoint


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("fixtures/run_*"))[-1]
    paths = sorted(glob.glob(f"{run}/ckpt_*"))
    print(f"trajectory: {run}  ({len(paths)} checkpoints)")
    traj = diff_trajectory([load_checkpoint(p) for p in paths])
    result = detect_trajectory(traj)
    print(f"series: {len(traj.series)}   events: {len(result.events)}\n")
    print("top events (most severe first; cal = severity / FPR-calibrated bar):")
    for e in result.top(15):
        loc = f"idx={e.index:2d} step={e.step}"
        cal = f"{e.calibrated_severity:5.1f}x" if e.calibrated_severity is not None else "   -  "
        print(f"  {loc:16s} {e.table}.{e.metric:16s} "
              f"cal={cal} raw={e.score:+7.2f} {e.method}")


if __name__ == "__main__":
    main()
