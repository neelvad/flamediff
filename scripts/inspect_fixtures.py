"""
Inspect a generated managed-collision checkpoint trajectory to confirm the MCH
serialization semantics the flamediff parser relies on (the id->slot map, the LFU
counts, churn across checkpoints, re-admission, and why an id-keyed gather is
required instead of a row-index subtract).

Run with the local CPU-torch venv:
    .venv/bin/python scripts/inspect_fixtures.py [run_dir]
"""
import json
import sys

import torch

RUN = sys.argv[1] if len(sys.argv) > 1 else "fixtures/run_1782180451"
NCKPT = 6
TBL = "author_id_emb"
FEAT = "author_id"


def mch(c):
    return torch.load(f"{RUN}/ckpt_{c:03d}/mch_buffers.pt",
                      weights_only=True, map_location="cpu")[TBL]


def weights(c):
    sd = torch.load(f"{RUN}/ckpt_{c:03d}/state_dict.pt",
                    weights_only=True, map_location="cpu")
    return sd[f"_embedding_module.embeddings.{TBL}.weight"]


def id_to_slot(c):
    b = mch(c)
    delim = int(b["_delimiter"].item())
    raw, slot = b["_mch_sorted_raw_ids"], b["_mch_remapped_ids_mapping"]
    occ = raw != delim
    return {int(r): int(s) for r, s in zip(raw[occ].tolist(), slot[occ].tolist())}


# 1) buffer layout + scalars at ckpt 0
b0 = mch(0)
delim = int(b0["_delimiter"].item())
print(f"[layout] table={TBL}")
for k, v in b0.items():
    print(f"  {k:28s} {str(tuple(v.shape)):9s} {str(v.dtype):12s} head={v.flatten()[:4].tolist()}")
print(f"[scalars] delimiter={delim}  mch_slots={int(b0['_mch_slots'].item())}  "
      f"current_iter={int(b0['_current_iter_tensor'].item())}")

m0 = id_to_slot(0)
print(f"[map] occupied={len(m0)}  raw_id=[{min(m0)},{max(m0)}]  "
      f"slot=[{min(m0.values())},{max(m0.values())}]  unique_slots={len(set(m0.values()))}")

# 2) is _mch_counts the LFU frequency? under zipf+offset0 the hottest raw ids are
#    near 0; if counts are parallel to the sorted-raw-id array, counts should be
#    largest at the smallest raw ids (front of the array).
raw, counts = b0["_mch_sorted_raw_ids"], b0["_mch_counts"]
occ = raw != delim
ro, co = raw[occ], counts[occ]
print(f"[counts] front-of-array (smallest raw ids): "
      f"{[(int(ro[i]), int(co[i])) for i in range(5)]}")
print(f"[counts] global max count={int(co.max())} at raw_id={int(ro[co.argmax()])}; "
      f"median={int(co.median())}")

# 3) id-keyed churn across consecutive checkpoints
print("[churn] id-keyed survivors / inserted / evicted per step:")
maps = [id_to_slot(c) for c in range(NCKPT)]
for c in range(1, NCKPT):
    a, b = set(maps[c - 1]), set(maps[c])
    print(f"   {c-1}->{c}: survivors={len(a & b):5d}  inserted={len(b - a):5d}  evicted={len(a - b):5d}")

# 4) re-admission: present@0, gone somewhere in the middle, back@3 (offset-0 revisit)
present = {i: [i in maps[c] for c in range(NCKPT)] for i in maps[0]}
readmit = [i for i, p in present.items() if p[0] and not all(p[:4]) and p[3]]
print(f"[re-admission] present@0, evicted mid-run, back@3: {len(readmit)} ids "
      f"(e.g. {readmit[:5]})")

# 5) the v1 thesis on real bytes: a row-index subtract is wrong; gather by id.
W = [weights(c) for c in range(NCKPT)]


def slot_to_id(c):
    return {s: i for i, s in maps[c].items()}


for c in range(1, NCKPT):  # find a slot reused by a different id (via eviction)
    prev, cur = slot_to_id(c - 1), slot_to_id(c)
    reused = [s for s in cur if s in prev and prev[s] != cur[s]]
    if reused:
        s = reused[0]
        d_rowidx = (W[c][s] - W[c - 1][s]).norm().item()
        print(f"[slot-reuse] {c-1}->{c} slot {s}: id {prev[s]} -> {cur[s]}; a "
              f"row-index subtract there ||Δ||={d_rowidx:.6f} conflates 2 entities")
        break

if readmit:  # re-admission inflates an id-keyed ||Δ|| vs true survivors
    i = readmit[0]
    s0, s3 = maps[0][i], maps[3][i]
    d_reborn = (W[3][s3] - W[0][s0]).norm().item()
    pool = list((set(maps[0]) & set(maps[3])) - set(readmit))[:300]
    surv_med = torch.stack(
        [W[3][maps[3][j]] - W[0][maps[0][j]] for j in pool]).norm(dim=1).median().item()
    print(f"[re-admission] id {i}: slot {s0}->{s3} (moved={s0 != s3}); id-keyed "
          f"||Δ|| 0->3 reborn={d_reborn:.6f} vs survivor median={surv_med:.6f} "
          f"({d_reborn / max(surv_med, 1e-9):.0f}x inflation)")
