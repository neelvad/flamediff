"""Modal (GPU): behavioral-probe experiment (branch E) -- does flamediff's weight-space drift
predict REAL behavioral drift, and how does that depend on embedding over-parameterization?

Train a full recsys model (MCEC embeddings + a dot-product interaction, matrix-factorization style)
on a rank-RANK latent task. Midway, flip the target relationship (latent -> -latent) for a random,
popularity-spanning subset of author ids: they must relearn -- a genuine meaning change that is, by
construction, independent of popularity. Checkpoint the MCEC and record canary-grid predictions.
Then, holding the TASK fixed, SWEEP the embedding dim and measure, per dim, how well weight-space
drift (raw ||delta||, flamediff's de-confounded residual, and the SUBSPACE-PROJECTED ||delta|| --
drift with the null-space component removed, at an automatic 90%-energy rank and at a fixed
2*RANK oracle) identifies the ids whose behavior actually changed -- yielding a degradation curve
vs over-parameterization (dim >> latent rank) and a test of whether null-space removal repairs it.

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
def experiment() -> str:
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

    # ---- the TASK, fixed across all dims: latents, the flipped subset, the canary grid ----
    trng = np.random.default_rng(0)
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
        rng = np.random.default_rng(1)  # identical data stream for every dim
        torch.manual_seed(0)
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
        from flamediff.spectral import project_deltas, table_spectrum
        from flamediff.stats import rank_at_energy
        cks = [load_checkpoint(os.path.join(tmp, f"ckpt_{i:03d}")) for i in range(WARMUP, CKPTS)]
        is_shifted = shift_mask[canary["author_id"]]
        acc = {k: np.zeros(N_CANARY) for k in ("resid", "raw", "proj_e90", "proj_r8", "beh", "n")}
        rank90 = 0
        for i in range(1, len(cks)):
            gi = WARMUP + i
            prev = cks[i - 1].embedding_tables["author_id_emb"]
            cur = cks[i].embedding_tables["author_id_emb"]
            diff = diff_table(prev, cur)
            attr = attribute_table(prev, cur, diff)
            beh = np.linalg.norm(fps[gi]["author_id"] - fps[gi - 1]["author_id"], axis=1)
            if FLIP_AT < gi <= FLIP_AT + POST_WINDOW:  # the immediate relearning window
                # subspace-projected ||delta||: drop null-space motion (the dilution hypothesis).
                # e90 = automatic rank (90% energy); r8 = fixed 2*RANK (a true-rank oracle).
                proj_e90 = project_deltas(prev, cur, diff.surv_ids, energy=0.90)
                proj_r8 = project_deltas(prev, cur, diff.surv_ids, rank=2 * RANK)
                rank90 = rank_at_energy(table_spectrum(cur), 0.90)
                o = {int(s): j for j, s in enumerate(diff.surv_ids)}
                j = np.array([o.get(int(a), -1) for a in canary["author_id"]])
                m = j >= 0
                acc["resid"][m] += attr.idiosyncratic[j[m]]
                acc["raw"][m] += diff.delta_norm[j[m]]
                acc["proj_e90"][m] += proj_e90[j[m]]
                acc["proj_r8"][m] += proj_r8[j[m]]
                acc["beh"][m] += beh[m]
                acc["n"][m] += 1
        ok = acc["n"] > 0
        return {
            "dim": dim, "final_loss": round(ck_loss[-1], 4), "n": int(ok.sum()),
            "rank90": rank90,
            "auc_behavior": round(_auc(acc["beh"][ok], is_shifted[ok]), 3),
            "auc_raw_delta": round(_auc(acc["raw"][ok], is_shifted[ok]), 3),
            "auc_residual": round(_auc(acc["resid"][ok], is_shifted[ok]), 3),
            "auc_proj_e90": round(_auc(acc["proj_e90"][ok], is_shifted[ok]), 3),
            "auc_proj_r8": round(_auc(acc["proj_r8"][ok], is_shifted[ok]), 3),
        }

    curve = []
    for dim in DIMS:
        row = run_one(dim)
        print(f"dim={row['dim']:2d}  loss={row['final_loss']:.3f}  rank90={row['rank90']:2d}  "
              f"auc_behavior={row['auc_behavior']:.3f}  auc_raw_delta={row['auc_raw_delta']:.3f}  "
              f"auc_residual={row['auc_residual']:.3f}  auc_proj_e90={row['auc_proj_e90']:.3f}  "
              f"auc_proj_r8={row['auc_proj_r8']:.3f}")
        curve.append(row)

    result = {"config": {"rank": RANK, "flip_at": FLIP_AT, "shift_frac": SHIFT_FRAC,
                         "canary": N_CANARY, "post_window": POST_WINDOW}, "curve": curve}
    with open("/data/behavioral_curve.json", "w") as fh:
        json.dump(result, fh, indent=2)
    vol.commit()
    return json.dumps(result)


@app.local_entrypoint()
def main():
    print(experiment.remote())
