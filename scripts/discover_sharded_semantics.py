"""Modal (2 GPUs): discover the row-wise managed-collision SLOT SEMANTICS -- the one fact
ShardedTable needs before it can correctly reassemble a multi-rank checkpoint.

Shards an MCEC across 2 ranks/GPUs, admits known ids via a forward pass (so the map buffers are
populated), DCP-saves, then reassembles the full buffers and checks PER RANK whether
_mch_remapped_ids_mapping values are LOCAL (0..zch/2) or GLOBAL (rank_offset + local), and whether
_mch_sorted_raw_ids is globally sorted or per-rank-sorted. [CONFIRM] marks risky API surface.

Run:  modal run scripts/discover_sharded_semantics.py
"""
import modal

CUDA_INDEX = "https://download.pytorch.org/whl/cu121"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy")
    .pip_install("torch==2.5.1", extra_index_url=CUDA_INDEX)
    .pip_install("fbgemm-gpu==1.0.0", extra_index_url=CUDA_INDEX)
    .pip_install("torchrec==1.0.0", extra_index_url=CUDA_INDEX)
)
vol = modal.Volume.from_name("flamediff-fixtures", create_if_missing=True)
app = modal.App("flamediff-sharded-sem", image=image)

WORLD = 2
OUT = "/data/sharded_sem"
FEATURES = ["author_id", "video_id"]
ZCH, DIM, VOCAB = 2000, 16, 5000


def _worker(rank: int, world: int) -> None:
    import os

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29502")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    from torchrec.distributed import DistributedModelParallel
    from torchrec.distributed.embedding import EmbeddingCollectionSharder
    from torchrec.distributed.mc_embedding import ManagedCollisionEmbeddingCollectionSharder
    from torchrec.distributed.mc_modules import ManagedCollisionCollectionSharder
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.modules.embedding_configs import EmbeddingConfig
    from torchrec.modules.embedding_modules import EmbeddingCollection
    from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingCollection
    from torchrec.modules.mc_modules import (
        LFU_EvictionPolicy,
        ManagedCollisionCollection,
        MCHManagedCollisionModule,
    )
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    meta = torch.device("meta")
    configs = [EmbeddingConfig(name=f"{f}_emb", embedding_dim=DIM, num_embeddings=ZCH,
                               feature_names=[f]) for f in FEATURES]
    ec = EmbeddingCollection(tables=configs, device=meta)
    mc_modules = {f"{f}_emb": MCHManagedCollisionModule(
        zch_size=ZCH, device=meta, input_hash_size=VOCAB,
        eviction_policy=LFU_EvictionPolicy(), eviction_interval=1) for f in FEATURES}
    model = ManagedCollisionEmbeddingCollection(ec, ManagedCollisionCollection(mc_modules, configs))

    sharder = ManagedCollisionEmbeddingCollectionSharder(
        EmbeddingCollectionSharder(), ManagedCollisionCollectionSharder())
    planner = EmbeddingShardingPlanner(topology=Topology(world_size=world, compute_device="cuda"))
    plan = planner.collective_plan(model, [sharder], dist.group.WORLD)
    dmp = DistributedModelParallel(module=model, device=device, plan=plan, sharders=[sharder])
    if rank == 0:
        print(f"[plan]\n{plan}")

    # admit a known spread of ids so the maps populate (same batch on both ranks; DMP routes)
    dmp.train()
    g = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(4):
        n = 512
        vals = torch.randint(0, VOCAB, (len(FEATURES) * n,), generator=g).to(device)
        kjt = KeyedJaggedTensor(  # [CONFIRM] forward input on a sharded MCEC
            keys=list(FEATURES), values=vals,
            lengths=torch.ones(len(FEATURES) * n, dtype=torch.long, device=device))
        dmp(kjt)
    dist.barrier()

    dcp.save(dmp.state_dict(), checkpoint_id=OUT)  # collective
    dist.destroy_process_group()


@app.function(gpu="T4:2", volumes={"/data": vol}, timeout=60 * 20)
def generate() -> str:
    import os
    import shutil

    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    import torch.multiprocessing as mp
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)
    mp.spawn(_worker, args=(WORLD,), nprocs=WORLD, join=True)
    vol.commit()

    # single-rank PG so dcp.load can reassemble the full tensors here
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = "29599"
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

    md = FileSystemReader(OUT).read_metadata()
    target = {fqn: torch.empty(tuple(m.size), dtype=m.properties.dtype)
              for fqn, m in md.state_dict_metadata.items() if isinstance(m, TensorStorageMetadata)}
    dcp.load(target, checkpoint_id=OUT)

    delim = np.iinfo(np.int64).max
    pfx = "_managed_collision_collection._managed_collision_modules.author_id_emb."
    raw = target[pfx + "_mch_sorted_raw_ids"].numpy()
    rem = target[pfx + "_mch_remapped_ids_mapping"].numpy()
    chunks = md.state_dict_metadata[pfx + "_mch_remapped_ids_mapping"].chunks
    chunk_spans = [(int(c.offsets[0]), int(c.sizes[0])) for c in chunks]
    print(f"[author_id_emb] global size={raw.shape}  chunks={chunk_spans}")
    for c in chunks:
        o, s = int(c.offsets[0]), int(c.sizes[0])
        raw_seg, rem_seg = raw[o:o + s], rem[o:o + s]
        occ = raw_seg != delim
        if not occ.any():
            print(f"  rank-chunk @offset {o} size {s}: EMPTY (no ids admitted)")
            continue
        rmin, rmax = int(rem_seg[occ].min()), int(rem_seg[occ].max())
        local = rmax < s
        sorted_local = bool(np.all(np.diff(raw_seg[occ]) > 0))
        print(f"  rank-chunk @offset {o} size {s}: occupied={int(occ.sum())}  "
              f"remapped_range=[{rmin},{rmax}]  -> slots are {'LOCAL' if local else 'GLOBAL'}  "
              f"raw_ids_sorted_within_chunk={sorted_local}")
        print(f"      sample (raw_id -> remapped): "
              f"{list(zip(raw_seg[occ][:5].tolist(), rem_seg[occ][:5].tolist(), strict=True))}")
    print(f"[globally sorted across chunks?] {bool(np.all(np.diff(raw[raw != delim]) > 0))}")
    dist.destroy_process_group()
    print(f"DONE {OUT}")
    return OUT


@app.local_entrypoint()
def main():
    print("sharded-semantics discovery ->", generate.remote())
