"""Modal (GPU): behavioral-probe experiment (branch E) -- does flamediff's weight-space drift
predict REAL behavioral drift, and how does that depend on embedding over-parameterization?

Train a full recsys model (MCEC embeddings + a dot-product interaction, matrix-factorization style)
on a rank-RANK latent task. Midway, flip the target relationship (latent -> -latent) for a random,
popularity-spanning subset of author ids: they must relearn -- a genuine meaning change that is, by
construction, independent of popularity. Checkpoint the MCEC and record canary-grid predictions.
Then, holding the TASK fixed, SWEEP the embedding dim and measure, per dim, how well weight-space
drift identifies the ids whose behavior actually changed. Scorers: raw ||delta||, flamediff's
de-confounded residual, own-basis subspace projection (a shipped negative -- kept as baseline),
and the INTERACTION-WEIGHTED variants: projection onto the CO-tower's covariance eigenbasis, and
the full covariance-weighted norm sqrt(d' C_video d) -- for a dot-product model, movement only
matters where the co-embeddings have mass, so the co-tower defines behavioral relevance.

Computes on Modal (flamediff mounted); returns the curve as JSON.

Run:  modal run scripts/behavioral_probe.py
"""
import modal

CUDA_INDEX = "https://download.pytorch.org/whl/cu121"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "ruptures>=1.1")
    .pip_install("torch==2.5.1", extra_index_url=CUDA_INDEX)
    .pip_install("fbgemm-gpu==1.0.0", extra_index_url=CUDA_INDEX)
    .pip_install("torchrec==1.0.0", extra_index_url=CUDA_INDEX)
    .add_local_python_source("flamediff")
)
vol = modal.Volume.from_name("flamediff-fixtures", create_if_missing=True)
app = modal.App("flamediff-behavioral", image=image)

FEATURES = ["author_id", "video_id"]
ZCH, VOCAB, HOT = 20_000, 50_000, 10_000
CKPTS, WARMUP, STEPS_PER_CKPT, BATCH = 20, 3, 40, 1024
FLIP_AT, SHIFT_FRAC, RANK = 6, 0.35, 4
N_CANARY, N_PANEL = 500, 128
DIMS = [8, 16, 32, 64]           # embedding dim swept against the true latent rank (RANK=4)
POST_WINDOW = 4                  # accumulate over the immediate post-flip relearning window


def _auc(score, positive):
    from flamediff.stats import _avg_rank
    npos = int(positive.sum())
    nneg = score.size - npos
    if npos == 0 or nneg == 0:
        return 0.5
    r = _avg_rank(score)
    return float((r[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg))


@app.function(gpu="A10G", volumes={"/data": vol}, timeout=60 * 60)
def experiment(seed: int = 0) -> str:
    import json
    import os
    import tempfile

    import numpy as np
    import torch
    from torchrec.modules.embedding_configs import EmbeddingConfig
    from torchrec.modules.embedding_modules import EmbeddingCollection
    from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingCollection
    from torchrec.modules.mc_modules import (
        LFU_EvictionPolicy,
        ManagedCollisionCollection,
        MCHManagedCollisionModule,
    )
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    device = torch.device("cuda")

    # ---- the TASK, fixed across all dims (but varied per seed -- each seed is a full
    # independent replication: new latents, flip set, canaries, data stream, and init) ----
    trng = np.random.default_rng(1000 + seed)
    latent = {f: trng.standard_normal((HOT, RANK)).astype(np.float32) for f in FEATURES}
    shift_mask = np.zeros(VOCAB, dtype=bool)
    shift_mask[trng.choice(HOT, size=int(HOT * SHIFT_FRAC), replace=False)] = True
    canary = {f: trng.choice(HOT, size=N_CANARY, replace=False).astype(np.int64) for f in FEATURES}
    panel = {f: trng.choice(HOT, size=N_PANEL, replace=False).astype(np.int64) for f in FEATURES}

    def other(feat):
        return "video_id" if feat == "author_id" else "author_id"

    def target(raw_a, raw_v, flipped):
        t = (latent["author_id"][raw_a] * latent["video_id"][raw_v]).sum(1)
        return np.where(shift_mask[raw_a], -t, t) if flipped else t

    def run_one(dim):
        rng = np.random.default_rng(2000 + seed)  # identical data stream for every dim
        torch.manual_seed(seed)
        configs = [EmbeddingConfig(name=f"{f}_emb", embedding_dim=dim, num_embeddings=ZCH,
                                   feature_names=[f]) for f in FEATURES]
        ec = EmbeddingCollection(tables=configs, device=device)
        mc = {f"{f}_emb": MCHManagedCollisionModule(
            zch_size=ZCH, device=device, input_hash_size=VOCAB,
            eviction_policy=LFU_EvictionPolicy(), eviction_interval=20) for f in FEATURES}
        model = ManagedCollisionEmbeddingCollection(
            ec, ManagedCollisionCollection(mc, configs)).to(device)
        bias = torch.nn.Parameter(torch.zeros(1, device=device))
        opt = torch.optim.Adam(list(model.parameters()) + [bias], lr=0.02)

        def interact(kjt):
            emb = model(kjt)
            emb = emb[0] if isinstance(emb, tuple) else emb
            return (emb["author_id"].values() * emb["video_id"].values()).sum(-1) + bias

        def train_batch(flipped):
            raw = {f: (rng.zipf(1.2, size=BATCH) - 1) % HOT for f in FEATURES}
            vals = torch.from_numpy(np.concatenate([raw[f] for f in FEATURES]).astype(np.int64))
            kjt = KeyedJaggedTensor(
                keys=list(FEATURES), values=vals.to(device),
                lengths=torch.ones(len(FEATURES) * BATCH, dtype=torch.long, device=device))
            y = target(raw["author_id"], raw["video_id"], flipped).astype(np.float32)
            return kjt, torch.from_numpy(y).to(device)

        @torch.no_grad()
        def embed(ids, feat):
            per = {feat: torch.from_numpy(ids.astype(np.int64)),
                   other(feat): torch.zeros(ids.size, dtype=torch.long)}
            kjt = KeyedJaggedTensor(
                keys=list(FEATURES),
                values=torch.cat([per["author_id"], per["video_id"]]).to(device),
                lengths=torch.ones(2 * ids.size, dtype=torch.long, device=device))
            emb = model(kjt)
            emb = emb[0] if isinstance(emb, tuple) else emb
            return emb[feat].values()

        @torch.no_grad()
        def fingerprints(frozen):
            model.eval()
            out = {f: (embed(canary[f], f) @ frozen[f].T).cpu().numpy() for f in FEATURES}
            model.train()
            return out

        tmp = tempfile.mkdtemp()
        fps, ck_loss, frozen, step = [], [], {}, 0
        for c in range(CKPTS):
            flipped = c >= FLIP_AT
            losses = []
            for _ in range(STEPS_PER_CKPT):
                kjt, tgt = train_batch(flipped)
                loss = ((interact(kjt) - tgt) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                step += 1
                losses.append(float(loss))
            ck_loss.append(float(np.mean(losses)))
            d = os.path.join(tmp, f"ckpt_{c:03d}")
            os.makedirs(d, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(d, "state_dict.pt"))
            with open(os.path.join(d, "meta.json"), "w") as fh:
                json.dump({"global_step": step}, fh)
            if c == 0:
                frozen = {f: embed(panel[other(f)], other(f)) for f in FEATURES}
            fps.append(fingerprints(frozen))

        from flamediff import diff_table, load_checkpoint
        from flamediff.attribute import attribute_table
        from flamediff.spectral import covariance_weighted_deltas, project_deltas, table_spectrum
        from flamediff.stats import rank_at_energy
        cks = [load_checkpoint(os.path.join(tmp, f"ckpt_{i:03d}")) for i in range(WARMUP, CKPTS)]
        is_shifted = shift_mask[canary["author_id"]]
        scores = ("resid", "raw", "proj_e90", "proj_r8", "xproj_e90", "xproj_r8", "xweight",
                  "beh", "n")
        acc = {k: np.zeros(N_CANARY) for k in scores}
        rank90 = 0
        for i in range(1, len(cks)):
            gi = WARMUP + i
            prev = cks[i - 1].embedding_tables["author_id_emb"]
            cur = cks[i].embedding_tables["author_id_emb"]
            co = cks[i].embedding_tables["video_id_emb"]  # the co-tower defines what gets READ
            diff = diff_table(prev, cur)
            attr = attribute_table(prev, cur, diff)
            beh = np.linalg.norm(fps[gi]["author_id"] - fps[gi - 1]["author_id"], axis=1)
            if FLIP_AT < gi <= FLIP_AT + POST_WINDOW:  # the immediate relearning window
                s = diff.surv_ids
                per = {
                    # own-basis projection: drop the table's own null-space motion (negative
                    # result -- kept as the baseline the interaction-weighted variants must beat)
                    "proj_e90": project_deltas(prev, cur, s, energy=0.90),
                    "proj_r8": project_deltas(prev, cur, s, rank=2 * RANK),
                    # interaction-weighted: behavioral relevance = where the CO-tower has mass
                    "xproj_e90": project_deltas(prev, cur, s, energy=0.90, basis_table=co),
                    "xproj_r8": project_deltas(prev, cur, s, rank=2 * RANK, basis_table=co),
                    "xweight": covariance_weighted_deltas(prev, cur, s, co),
                    "resid": attr.idiosyncratic, "raw": diff.delta_norm,
                }
                rank90 = rank_at_energy(table_spectrum(cur), 0.90)
                o = {int(x): j for j, x in enumerate(s)}
                j = np.array([o.get(int(a), -1) for a in canary["author_id"]])
                m = j >= 0
                for k, v in per.items():
                    acc[k][m] += v[j[m]]
                acc["beh"][m] += beh[m]
                acc["n"][m] += 1
        ok = acc["n"] > 0
        row = {"dim": dim, "final_loss": round(ck_loss[-1], 4), "n": int(ok.sum()),
               "rank90": rank90,
               "auc_behavior": round(_auc(acc["beh"][ok], is_shifted[ok]), 3)}
        for k in ("raw", "resid", "proj_e90", "proj_r8", "xproj_e90", "xproj_r8", "xweight"):
            row[f"auc_{k}"] = round(_auc(acc[k][ok], is_shifted[ok]), 3)
        return row

    curve = []
    for dim in DIMS:
        row = run_one(dim)
        print(f"seed={seed}  " + "  ".join(f"{k}={v}" for k, v in row.items()))
        curve.append(row)

    result = {"seed": seed,
              "config": {"rank": RANK, "flip_at": FLIP_AT, "shift_frac": SHIFT_FRAC,
                         "canary": N_CANARY, "post_window": POST_WINDOW}, "curve": curve}
    with open(f"/data/behavioral_curve_seed{seed}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    vol.commit()
    return json.dumps(result)


N_SEEDS = 5


@app.local_entrypoint()
def main():
    """Fan the seeds out as parallel containers, then aggregate mean +- std and the PAIRED
    within-seed contrasts that test the two claims (dilution and its attempted repair).
    Stdlib-only aggregation: the local entrypoint runs in the modal CLI's env, not the venv."""
    import json as _json
    from statistics import mean, stdev

    runs = [_json.loads(r) for r in experiment.map(range(N_SEEDS))]
    by_dim = {dim: [next(row for row in r["curve"] if row["dim"] == dim) for r in runs]
              for dim in DIMS}
    keys = [k for k in by_dim[DIMS[0]][0] if k.startswith("auc_")] + ["final_loss", "rank90"]

    print(f"\n=== mean +- std over {N_SEEDS} seeds ===")
    for dim in DIMS:
        cells = []
        for k in keys:
            v = [float(row[k]) for row in by_dim[dim]]
            label = k[4:] if k.startswith("auc_") else k
            cells.append(f"{label}={mean(v):.3f}±{stdev(v):.3f}")
        print(f"dim={dim:2d}  " + "  ".join(cells))

    def paired(a_dim, a_key, b_dim, b_key):
        d = [by_dim[a_dim][s][a_key] - by_dim[b_dim][s][b_key] for s in range(N_SEEDS)]
        wins = sum(1 for x in d if x > 0)
        return f"{mean(d):+.3f}±{stdev(d):.3f} ({wins}/{N_SEEDS} seeds >0)"

    print("\n=== paired within-seed contrasts ===")
    print("dilution   raw@16 - raw@64:      ", paired(16, "auc_raw", 64, "auc_raw"))
    print("repair(own) proj_r8@64 - raw@64:  ", paired(64, "auc_proj_r8", 64, "auc_raw"))
    print("repair(x)  xproj_r8@64 - raw@64:  ", paired(64, "auc_xproj_r8", 64, "auc_raw"))
    print("repair(xw) xweight@64 - raw@64:   ", paired(64, "auc_xweight", 64, "auc_raw"))
    print("resid      resid@64 - raw@64:     ", paired(64, "auc_resid", 64, "auc_raw"))

    out = {"n_seeds": N_SEEDS, "runs": runs}
    with open("fixtures/behavioral_multiseed.json", "w") as fh:
        _json.dump(out, fh, indent=2)
    print("\nwrote fixtures/behavioral_multiseed.json")
