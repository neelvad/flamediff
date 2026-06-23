"""Walk a generated trajectory and print a per-step structural diff report.

    .venv/bin/python scripts/run_diff.py [run_dir]

This is the "see it work" driver: it runs the real adapter + diff over consecutive
checkpoints and surfaces churn, geometry drift, and the top frequency-residual movers.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

from flamediff import diff_checkpoints, load_checkpoint


def main() -> None:
    run_dir = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("fixtures/run_*"))[-1]
    ckpts = sorted(glob.glob(os.path.join(run_dir, "ckpt_*")))
    print(f"trajectory: {run_dir}  ({len(ckpts)} checkpoints)\n")

    prev = load_checkpoint(ckpts[0])
    for path in ckpts[1:]:
        cur = load_checkpoint(path)
        d = diff_checkpoints(prev, cur)
        print(f"=== {os.path.basename(ckpts[ckpts.index(path) - 1])} -> {os.path.basename(path)}"
              f"  (step {d.step_prev} -> {d.step_cur}) ===")
        for name, td in d.embedding_diffs.items():
            c = td.churn_summary()
            print(f"  [{name}]  resident {c['prev']} -> {c['cur']}   "
                  f"survivors={c['survivors']} inserted={c['inserted']} evicted={c['evicted']}  "
                  f"(slot_stable={c['slot_stable']} slot_moved={c['slot_moved']})")
            gp, gc = td.geom_prev, td.geom_cur
            print(f"        geometry: eff_rank {gp.effective_rank:.2f}->{gc.effective_rank:.2f}  "
                  f"anisotropy {gp.anisotropy:.2f}->{gc.anisotropy:.2f}  "
                  f"mean_norm {gp.mean_row_norm:.4f}->{gc.mean_row_norm:.4f}")
            if td.surv_ids.size:
                print(f"        clean survivor ||Δ||: median={float(np.median(td.delta_norm)):.5f}"
                      f"  max={float(td.delta_norm.max()):.5f}")
                print("        top freq_resid movers (id, score, ||Δ||, dcount):")
                for mid, score, dn, dc in td.top_movers(5, by="freq_resid"):
                    print(f"        id={mid:6d}  score={score:+6.2f}  ||Δ||={dn:.5f}  dcount={dc}")
        print()
        prev = cur


if __name__ == "__main__":
    main()
