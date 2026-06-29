"""Modal: REAL-data calibration. Generate several CLEAN, stationary TorchRec MCH runs at real
scale (dim=64, 2 tables, fixed hot region -> no engineered events), then run flamediff's own
calibration sweep on them and write a real-derived calibration.json. Computes on Modal (flamediff
is mounted) so only the small JSON comes back -- no checkpoint download.

Run:  modal run scripts/calibrate_real.py
Then: modal volume get flamediff-fixtures calibration_real.json flamediff/calibration.json
"""
import modal

CUDA_INDEX = "https://download.pytorch.org/whl/cu121"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "ruptures>=1.1")
    .pip_install("torch==2.5.1", extra_index_url=CUDA_INDEX)
    .pip_install("fbgemm-gpu==1.0.0", extra_index_url=CUDA_INDEX)
    .pip_install("torchrec==1.0.0", extra_index_url=CUDA_INDEX)
    .add_local_python_source("flamediff")  # [CONFIRM] mount the local flamediff package
)
vol = modal.Volume.from_name("flamediff-fixtures", create_if_missing=True)
app = modal.App("flamediff-calibrate-real", image=image)

FEATURES = ["author_id", "video_id"]
ZCH, DIM, VOCAB = 20_000, 64, 50_000           # real scale (matches the demo fixture)
N_RUNS, CKPTS, WARMUP = 10, 15, 4              # 10 clean runs; drop the table-filling transient
STEPS_PER_CKPT, BATCH = 40, 1024
TARGET_FPR = 0.05
METHODS = ("robust_z", "page_hinkley", "pelt")


def _build_run(seed: int, outdir: str) -> None:
    """Train one STATIONARY MCH run (fixed hot region) and save state_dict + meta per checkpoint."""
    import json
    import os

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
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    configs = [EmbeddingConfig(name=f"{f}_emb", embedding_dim=DIM, num_embeddings=ZCH,
                               feature_names=[f]) for f in FEATURES]
    ec = EmbeddingCollection(tables=configs, device=device)
    mc = {f"{f}_emb": MCHManagedCollisionModule(zch_size=ZCH, device=device, input_hash_size=VOCAB,
              eviction_policy=LFU_EvictionPolicy(), eviction_interval=20) for f in FEATURES}
    model = ManagedCollisionEmbeddingCollection(
        ec, ManagedCollisionCollection(mc, configs)).to(device)
    head = torch.nn.Linear(DIM * len(FEATURES), 1).to(device)
    opt = torch.optim.SGD(list(model.parameters()) + list(head.parameters()), lr=0.1)

    def make_batch():
        per, vals = {}, []
        for f in FEATURES:
            raw = ((rng.zipf(1.2, size=BATCH) - 1) % VOCAB).astype(np.int64)  # FIXED region
            ids = torch.from_numpy(raw)
            per[f] = ids
            vals.append(ids)
        kjt = KeyedJaggedTensor(
            keys=list(FEATURES), values=torch.cat(vals).to(device),
            lengths=torch.ones(len(FEATURES) * BATCH, dtype=torch.long, device=device))
        tgt = torch.sin((per[FEATURES[0]].float() + per[FEATURES[1]].float()) / VOCAB)
        return kjt, tgt.unsqueeze(1).to(device)

    step = 0
    for c in range(CKPTS):
        for _ in range(STEPS_PER_CKPT):
            kjt, tgt = make_batch()
            out = model(kjt)
            emb = out[0] if isinstance(out, tuple) else out
            x = torch.cat([emb[f].values() for f in FEATURES], dim=1)
            loss = ((head(x) - tgt) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        d = os.path.join(outdir, f"ckpt_{c:03d}")
        os.makedirs(d, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(d, "state_dict.pt"))
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump({"global_step": step}, fh)


@app.function(gpu="T4", volumes={"/data": vol}, timeout=60 * 40)
def calibrate_real() -> str:
    import json
    import os
    import tempfile

    import numpy as np

    from flamediff import load_checkpoint
    from flamediff.calibrate import MutationSpec, derive_params, run_calibration

    tmp = tempfile.mkdtemp()
    clean = []
    for seed in range(N_RUNS):
        run_dir = os.path.join(tmp, f"run_{seed}")
        _build_run(seed, run_dir)
        cks = [load_checkpoint(os.path.join(run_dir, f"ckpt_{c:03d}"))
               for c in range(WARMUP, CKPTS)]
        clean.append(cks)
        print(f"[run {seed}] {len(cks)} stationary checkpoints")

    battery = []
    for persistent in (False, True):
        battery += [MutationSpec("scramble", m, persistent=persistent) for m in (0.5, 2.0, 8.0)]
        battery.append(MutationSpec("freeze", 1.0, persistent=persistent))
    result = run_calibration(clean, battery, trials_per_cell=20, table="author_id_emb", seed=0)

    print(f"=== REAL calibration (target per-run FPR={TARGET_FPR}) ===")
    for m in METHODS:
        t = result.operating_threshold(m, TARGET_FPR)
        pool = result.null.pooled[m]
        print(f"  {m:13s} thresh={t:8.2f}  null med={float(np.median(pool)):.2f}  n={pool.size}")
    print("power = TPR at the operating point:")
    for label in result.labels():
        for mag in result.magnitudes(label):
            parts = []
            for m in METHODS:
                tpr = result.tpr(label, mag, m, result.operating_threshold(m, TARGET_FPR))
                parts.append(f"{m[:4]}={tpr:.2f}")
            print(f"  [{label}] mag={mag:<4g} " + "  ".join(parts))

    params = derive_params(result, target_fpr=TARGET_FPR)
    params["provenance"] = (
        f"REAL: {N_RUNS} clean stationary TorchRec MCH runs (dim={DIM}, zch={ZCH}, 2 tables, "
        f"{CKPTS - WARMUP} steady-state ckpts each); regenerate via scripts/calibrate_real.py")
    with open("/data/calibration_real.json", "w") as fh:
        json.dump(params, fh, indent=2)
    vol.commit()
    print("wrote /data/calibration_real.json")
    return json.dumps(params, indent=2)


@app.local_entrypoint()
def main():
    print(calibrate_real.remote())
