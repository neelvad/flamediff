"""Detection-power demo: inject a known corruption into a real checkpoint and show the diff
catches it above the natural-drift noise floor.

    uv run scripts/mutation_demo.py [run_dir]
"""
from __future__ import annotations

import glob
import sys

import numpy as np

from flamediff import diff_checkpoints, load_checkpoint
from flamediff.mutate import mutate_checkpoint

TABLE = "author_id_emb"
MAGNITUDE = 5.0


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("fixtures/run_*"))[-1]
    prev = load_checkpoint(f"{run}/ckpt_002")
    cur = load_checkpoint(f"{run}/ckpt_003")

    # scramble a handful of resident survivor ids (the planted anomalies)
    survivors = np.intersect1d(
        prev.embedding_tables[TABLE].ids(), cur.embedding_tables[TABLE].ids()
    )
    injected = np.random.default_rng(0).choice(survivors, size=5, replace=False)
    mutated, mut = mutate_checkpoint(cur, TABLE, kind="scramble", ids=injected, magnitude=MAGNITUDE)

    base = diff_checkpoints(prev, cur).embedding_diffs[TABLE]       # natural drift only
    test = diff_checkpoints(prev, mutated).embedding_diffs[TABLE]   # drift + injection
    inj = set(int(i) for i in injected)

    print(f"trajectory: {run}")
    print(f"injected (scramble x{mut.magnitude:g}) ids: {sorted(inj)}")
    print(f"natural-drift noise floor: max ||Δ||={base.delta_norm.max():.5f}  "
          f"median={np.median(base.delta_norm):.5f}")
    print("top-8 by ||Δ|| after injection (* = injected):")
    for mid, _, dn, dc in test.top_movers(8, by="delta_norm"):
        mark = "*" if mid in inj else " "
        print(f"  {mark} id={mid:6d}  ||Δ||={dn:.5f}  dcount={dc}")

    caught = inj & {m[0] for m in test.top_movers(len(inj), by="freq_resid")}
    print(f"detection power: {len(caught)}/{len(inj)} injected ids in top-{len(inj)} by freq_resid")


if __name__ == "__main__":
    main()
