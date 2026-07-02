"""Generate a synthetic MCH-format trajectory with real low-rank structure, for the
`flamediff rank` demo page (docs/rank.html). Local + CPU-only: it writes the same single-device
state_dict layout the TorchRec adapter parses, so it exercises the real load path -- but the
weights are synthesized, not trained (the demo is labeled as such).

Two stories at dim=64:
  - `author_id_emb` grows into rank ~12 and plateaus -> a clear factorization win (advisory fires:
    top-12 factors keep 95% at ~19% of the parameters).
  - `video_id_emb` keeps growing rank through the end of the run -> the advisory correctly warns
    that sizing a factorization now would bake in an unfinished dimensionality.

Run:  uv run scripts/generate_rank_demo.py [out_dir]     (default fixtures/rank_demo)
Then: uv run flamediff rank fixtures/rank_demo --html docs/rank.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

DELIM = np.iinfo(np.int64).max
DIM, N_IDS, NUM_SLOTS, CKPTS = 64, 6000, 8000, 16
NOISE = 0.06  # small vs unit factor strength, so rank95 sits at the active-factor count


def rank_schedule(table: str, t: int) -> int:
    if table == "author_id_emb":
        return min(3 + t, 12)      # ramps, then 7 plateau checkpoints -> "stable since"
    return min(6 + 3 * t, DIM - 4)  # still climbing at the last checkpoint -> "still moving"


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)

    tables = {}
    for name in ("author_id_emb", "video_id_emb"):
        U = torch.randn(N_IDS, DIM, generator=g)            # per-id factor loadings
        V, _ = torch.linalg.qr(torch.randn(DIM, DIM, generator=g))  # orthonormal factor basis
        counts = np.sort(rng.zipf(1.3, N_IDS).astype(np.int64))[::-1].copy()
        tables[name] = (U, V, counts)

    ids = np.arange(N_IDS, dtype=np.int64) * 7 + 3          # arbitrary raw-id space
    slots = np.arange(N_IDS, dtype=np.int64)
    pad = NUM_SLOTS - N_IDS

    for t in range(CKPTS):
        sd = {}
        for name, (U, V, counts) in tables.items():
            r = rank_schedule(name, t)
            # active factors at unit strength + a shared small isotropic noise floor; jitter the
            # loadings a little per step so consecutive diffs look like training, not teleporting
            jitter = 0.03 * torch.randn(N_IDS, r, generator=g)
            W = torch.zeros(NUM_SLOTS, DIM)
            W[:N_IDS] = (U[:, :r] + jitter) @ V[:r] + NOISE * torch.randn(
                N_IDS, DIM, generator=g)
            pfx = f"_managed_collision_collection._managed_collision_modules.{name}."
            sd[pfx + "_mch_sorted_raw_ids"] = torch.from_numpy(
                np.concatenate([ids, np.full(pad, DELIM, dtype=np.int64)]))
            sd[pfx + "_mch_remapped_ids_mapping"] = torch.from_numpy(
                np.concatenate([slots, np.zeros(pad, dtype=np.int64)]))
            sd[pfx + "_mch_counts"] = torch.from_numpy(
                np.concatenate([counts + t * (1 + counts // 4), np.zeros(pad, dtype=np.int64)]))
            sd[pfx + "_delimiter"] = torch.tensor([DELIM])
            sd[pfx + "_mch_slots"] = torch.tensor([NUM_SLOTS - 1])
            sd[f"_embedding_module.embeddings.{name}.weight"] = W
        d = out / f"ckpt_{t:03d}"
        d.mkdir(exist_ok=True)
        torch.save(sd, d / "state_dict.pt")
        (d / "meta.json").write_text(json.dumps({"global_step": (t + 1) * 100}))
    print(f"wrote {CKPTS} checkpoints under {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fixtures/rank_demo")
