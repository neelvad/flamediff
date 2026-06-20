"""
Modal app: generate a TorchRec managed-collision (MCH/ZCH) checkpoint *trajectory*
to serve as the reference fixture for flamediff's embedding-diff parser.

Why a trajectory and not a single checkpoint:
    flamediff diffs *consecutive* checkpoints, so we need >= 2 from the same run with
    insertion / eviction / re-admission actually happening between them, plus
    ground-truth about which raw ids were fed each interval so tests can assert against
    known answers (mutation-testing style).

What this run produces, per checkpoint, under /data/run_<ts>/ckpt_NNN/:
    state_dict.pt            - full model state_dict (embedding weights + MCH buffers)
    mch_buffers.pt           - the managed-collision module buffers per feature (the
                               id->slot map + eviction metadata the parser must read)
    mch_buffers_summary.json - human-readable buffer layout (name/shape/dtype)
    fed_ids.json             - exact set of raw ids fed this interval (ground truth)
    meta.json                - config + step bookkeeping

Run:
    modal run scripts/generate_checkpoints.py
Download the trajectory it prints:
    modal volume get flamediff-fixtures run_<ts> ./fixtures/

NOTE — this is also a *discovery* run. TorchRec's managed-collision API and the
serialized buffer names are version-dependent, and we can't build TorchRec locally
(arm64 macOS). So the first thing this does is print torch/torchrec/fbgemm versions
and dump the MCH buffer layout. Lines marked [CONFIRM] use API surface that should be
validated against the installed torchrec version on first run; if one is wrong, the
run fails loudly with versions already printed, and we fix the pin/name and re-run.

Design note: this builds the modules on a single device (NOT DistributedModelParallel),
so the serialization is the un-sharded form. A sharded trajectory is a later fixture.
"""
import modal

# TorchRec is Linux/CUDA-first; pull CUDA wheels from the pytorch index. torchrec is
# pinned because the parser is written against its exact serialization; torch/fbgemm
# are pinned to the coherent release set for torchrec 1.0.0. [CONFIRM on first run]
CUDA_INDEX = "https://download.pytorch.org/whl/cu121"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy")
    .pip_install("torch==2.5.1", extra_index_url=CUDA_INDEX)
    .pip_install("fbgemm-gpu==1.0.0", extra_index_url=CUDA_INDEX)
    .pip_install("torchrec==1.0.0", extra_index_url=CUDA_INDEX)
)

vol = modal.Volume.from_name("flamediff-fixtures", create_if_missing=True)
app = modal.App("flamediff-ckpt-gen", image=image)


# ---- generation config (small dev scale; a T4 has huge headroom) -------------------
class Cfg:
    features = ["author_id", "video_id"]
    raw_vocab = 50_000      # distinct possible raw ids per feature
    zch_size = 20_000       # table capacity / slots (< raw_vocab => eviction forced)
    embedding_dim = 64
    num_checkpoints = 6
    steps_per_ckpt = 50
    batch_size = 1024
    eviction_interval = 100  # MCH evicts every N forward calls
    lr = 0.1
    seed = 0
    # Per-checkpoint offset of the Zipfian "hot" region. Shifting it forces eviction of
    # now-cold ids and admission of newly-hot ids. ckpt 3 revisits ckpt 0's region and
    # ckpt 5 revisits ckpt 1's -> deliberately exercises RE-ADMISSION (the confound).
    hot_offsets = [0, 10_000, 20_000, 0, 30_000, 10_000]


@app.function(gpu="T4", volumes={"/data": vol}, timeout=60 * 30)
def generate() -> str:
    import json
    import os
    import time

    import numpy as np
    import torch

    cfg = Cfg()
    device = torch.device("cuda")

    # --- pin reality: print versions before touching the finicky API ----------------
    import torchrec  # noqa: E402
    try:
        import fbgemm_gpu  # noqa: F401
        fbgemm_ver = getattr(fbgemm_gpu, "__version__", "?")
    except Exception as e:  # pragma: no cover
        fbgemm_ver = f"import-failed: {e}"
    print(f"[versions] torch={torch.__version__} torchrec={torchrec.__version__} "
          f"fbgemm_gpu={fbgemm_ver} cuda={torch.version.cuda} "
          f"device={torch.cuda.get_device_name(0)}")

    # [CONFIRM] managed-collision API surface (paths/args may shift across versions)
    from torchrec.modules.embedding_configs import EmbeddingConfig
    from torchrec.modules.embedding_modules import EmbeddingCollection
    from torchrec.modules.mc_modules import (
        LFU_EvictionPolicy,
        ManagedCollisionCollection,
        MCHManagedCollisionModule,
    )
    from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingCollection
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    # --- model: EmbeddingCollection wrapped in managed collision (MCH/ZCH) -----------
    embedding_configs = [
        EmbeddingConfig(
            name=f"{f}_emb",
            embedding_dim=cfg.embedding_dim,
            num_embeddings=cfg.zch_size,
            feature_names=[f],
        )
        for f in cfg.features
    ]
    ec = EmbeddingCollection(tables=embedding_configs, device=device)

    mc_modules = {
        f: MCHManagedCollisionModule(  # [CONFIRM] arg names
            zch_size=cfg.zch_size,
            device=device,
            input_hash_size=cfg.raw_vocab,
            eviction_policy=LFU_EvictionPolicy(),
            eviction_interval=cfg.eviction_interval,
        )
        for f in cfg.features
    }
    mcc = ManagedCollisionCollection(  # [CONFIRM] arg names
        managed_collision_modules=mc_modules,
        embedding_configs=embedding_configs,
    )
    model = ManagedCollisionEmbeddingCollection(ec, mcc).to(device)  # [CONFIRM] signature

    head = torch.nn.Linear(cfg.embedding_dim * len(cfg.features), 1).to(device)
    opt = torch.optim.SGD(list(model.parameters()) + list(head.parameters()), lr=cfg.lr)

    # --- synthetic Zipfian id stream with a per-checkpoint shifting hot region -------
    def make_ids(ckpt_idx: int, n: int) -> np.ndarray:
        offset = cfg.hot_offsets[ckpt_idx % len(cfg.hot_offsets)]
        raw = (rng.zipf(1.2, size=n) - 1) % cfg.raw_vocab  # zipf -> popularity skew
        return ((raw + offset) % cfg.raw_vocab).astype(np.int64)

    def make_batch(ckpt_idx: int):
        per_feat = {}
        values = []
        for f in cfg.features:
            ids = torch.from_numpy(make_ids(ckpt_idx, cfg.batch_size))
            per_feat[f] = ids
            values.append(ids)
        kjt = KeyedJaggedTensor(  # feature-major, one id per example (lengths all 1)
            keys=list(cfg.features),
            values=torch.cat(values).to(device),
            lengths=torch.ones(len(cfg.features) * cfg.batch_size,
                               dtype=torch.long, device=device),
        )
        # synthetic learnable target so embeddings actually move
        target = torch.sin(
            (per_feat[cfg.features[0]].float() + per_feat[cfg.features[1]].float())
            / cfg.raw_vocab
        ).unsqueeze(1).to(device)
        return kjt, per_feat, target

    def dump_checkpoint(run_dir: str, idx: int, fed_ids: dict, step: int):
        ckpt_dir = os.path.join(run_dir, f"ckpt_{idx:03d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "state_dict.pt"))

        # the MCH buffers ARE the id->slot map + eviction metadata the parser reads
        mch_state, summary = {}, {}
        for f, m in mc_modules.items():
            bufs = {name: t.detach().cpu() for name, t in m.named_buffers()}
            mch_state[f] = bufs
            summary[f] = {name: {"shape": list(t.shape), "dtype": str(t.dtype)}
                          for name, t in bufs.items()}
        torch.save(mch_state, os.path.join(ckpt_dir, "mch_buffers.pt"))
        with open(os.path.join(ckpt_dir, "mch_buffers_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        with open(os.path.join(ckpt_dir, "fed_ids.json"), "w") as fh:
            json.dump({f: sorted(int(x) for x in set(v.tolist()))
                       for f, v in fed_ids.items()}, fh)
        with open(os.path.join(ckpt_dir, "meta.json"), "w") as fh:
            json.dump({"ckpt_idx": idx, "global_step": step,
                       "hot_offset": cfg.hot_offsets[idx % len(cfg.hot_offsets)],
                       "config": {k: v for k, v in vars(Cfg).items()
                                  if not k.startswith("_")}}, fh, indent=2)
        if idx == 0:
            print(f"[discovery] state_dict keys:\n  " +
                  "\n  ".join(model.state_dict().keys()))
            print(f"[discovery] MCH buffer layout:\n{json.dumps(summary, indent=2)}")
        return ckpt_dir

    # --- run the trajectory ----------------------------------------------------------
    run_dir = f"/data/run_{int(time.time())}"
    os.makedirs(run_dir, exist_ok=True)
    global_step = 0
    for c in range(cfg.num_checkpoints):
        fed = {f: [] for f in cfg.features}
        last_loss = None
        for _ in range(cfg.steps_per_ckpt):
            kjt, per_feat, target = make_batch(c)
            out = model(kjt)
            emb_dict = out[0] if isinstance(out, tuple) else out  # [CONFIRM] return
            x = torch.cat([emb_dict[f].values() for f in cfg.features], dim=1)
            loss = ((head(x) - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss)
            for f in cfg.features:
                fed[f].append(per_feat[f])
            global_step += 1
        fed = {f: torch.cat(v) for f, v in fed.items()}
        ckpt_dir = dump_checkpoint(run_dir, c, fed, global_step)
        vol.commit()
        print(f"[ckpt {c}] step={global_step} loss={last_loss:.4f} -> {ckpt_dir}")

    print(f"DONE run_dir={run_dir}")
    return os.path.basename(run_dir)


@app.local_entrypoint()
def main():
    run_name = generate.remote()
    print("\nGenerated trajectory:", run_name)
    print("Download it with:")
    print(f"    modal volume get flamediff-fixtures {run_name} ./fixtures/")
