"""Attribution demo: *why* embeddings drift across a real run, plus a proof on real data that the
de-confounded residual finds genuine change that raw ||delta|| misses.

    uv run scripts/attribution_demo.py [run_dir]
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import torch

from flamediff import attribute_table, diff_table, load_checkpoint
from flamediff.stats import _avg_rank


def _auc(score: np.ndarray, positive: np.ndarray) -> float:
    n_pos = int(positive.sum())
    n_neg = score.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _avg_rank(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("fixtures/run_*"))[-1]
    cks = [load_checkpoint(p) for p in sorted(glob.glob(f"{run}/ckpt_*"))]
    table = sorted(cks[0].embedding_tables)[0]
    print(f"trajectory: {run}  ({len(cks)} checkpoints)  table={table}\n")

    print("per-step drift attribution (energy fractions; pop_r2 = popularity's hold on the drift):")
    print(f"  {'step':>11}  {'global':>7} {'pop_r2':>7} {'idio':>7}   top idiosyncratic movers")
    for i in range(1, len(cks)):
        prev, cur = cks[i - 1].embedding_tables[table], cks[i].embedding_tables[table]
        attr = attribute_table(prev, cur, diff_table(prev, cur))
        movers = ", ".join(str(m[0]) for m in attr.top_movers(5))
        step = f"{cks[i - 1].step}->{cks[i].step}"
        print(f"  {step:>11}  {attr.frac_global:7.2f} {attr.popularity_r2:7.2f} "
              f"{attr.frac_aligned_residual:7.2f}   {movers}")

    # --- injection validation on the middle pair: plant popularity-independent change ---
    j = len(cks) // 2
    prev, cur = cks[j - 1].embedding_tables[table], cks[j].embedding_tables[table]
    base = diff_table(prev, cur)
    surv, dcount = base.surv_ids, base.dcount
    low = surv[dcount <= np.quantile(dcount, 0.3)]  # low-traffic survivors
    rng = np.random.default_rng(0)
    planted = rng.choice(low, size=min(50, low.size), replace=False)

    moved = base.delta_norm[base.delta_norm > 0]
    typ = float(np.median(moved)) if moved.size else 1.0  # a "typical" drift magnitude
    g = torch.from_numpy(rng.standard_normal((planted.size, cur.dim)).astype(np.float32))
    scramble = g / g.norm(dim=1, keepdim=True) * typ  # normal-sized move, but in low-traffic ids
    W = cur.copy_weights()
    W[torch.from_numpy(cur.slot_of(planted))] = prev.gather(planted).float() + scramble
    cur_mut = cur.with_weights(W)

    inj = diff_table(prev, cur_mut)
    attr = attribute_table(prev, cur_mut, inj)
    planted_surv = np.isin(inj.surv_ids, planted)
    auc_raw = _auc(inj.delta_norm, planted_surv)
    auc_idio = _auc(attr.idiosyncratic, planted_surv)
    print(f"\ninjection validation (plant {planted.size} typical-sized changes in low-traffic ids):")
    print(f"  recover via raw ||delta||  AUC={auc_raw:.3f}")
    print(f"  recover via de-confounded  AUC={auc_idio:.3f}   (lift {auc_idio - auc_raw:+.3f})")


if __name__ == "__main__":
    main()
