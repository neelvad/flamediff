"""Modal app: produce a *sharded* TorchRec checkpoint and discover its on-disk serialization
-- the foundation for flamediff's ShardedTable / v1.0 real-scale work.

DISCOVERY run (like generate_checkpoints.py was). It shards a ManagedCollisionEmbeddingCollection
with DistributedModelParallel, saves via torch.distributed.checkpoint (DCP), and dumps: versions,
the sharding plan, the sharded state_dict structure, and the DCP layout. Findings:
  - MC tables shard ONLY row_wise + a fused GPU kernel (so GPU is required; we use world_size=1).
  - The weight AND the map buffers (_mch_sorted_raw_ids / _mch_remapped_ids_mapping / _mch_counts)
    are each a row-wise ShardedTensor; scalars (_current_iter_tensor, _output_segments_tensor)
    stay replicated plain Tensors.
  - On disk: a DCP dir (.metadata + __{rank}_{idx}.distcp); read via FileSystemReader.read_metadata,
    which gives per-fqn TensorStorageMetadata (global size + chunks of offset/size).

[CONFIRM] marks sharding/DCP API surface validated against torchrec 1.0.0.

Run:  modal run scripts/generate_sharded.py
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
app = modal.App("flamediff-sharded-gen", image=image)

WORLD = 1  # MC sharding requires GPU (row_wise + fused kernel); 1 rank on the single T4
OUT = "/data/sharded_run"


def _worker(rank: int, world: int) -> None:
    import os

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    # MC embedding collections only shard row_wise + fused kernel -> GPU required.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    import torchrec
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

    if rank == 0:
        print(f"[versions] torch={torch.__version__} torchrec={torchrec.__version__} "
              f"world_size={world}")

    features = ["author_id", "video_id"]
    zch, dim, vocab = 2000, 16, 5000
    meta = torch.device("meta")
    configs = [EmbeddingConfig(name=f"{f}_emb", embedding_dim=dim, num_embeddings=zch,
                               feature_names=[f]) for f in features]
    ec = EmbeddingCollection(tables=configs, device=meta)
    mc_modules = {f"{f}_emb": MCHManagedCollisionModule(  # [CONFIRM] meta-device MC module
        zch_size=zch, device=meta, input_hash_size=vocab,
        eviction_policy=LFU_EvictionPolicy(), eviction_interval=10) for f in features}
    model = ManagedCollisionEmbeddingCollection(ec, ManagedCollisionCollection(mc_modules, configs))

    # [CONFIRM] managed-collision sharder construction + module paths
    sharder = ManagedCollisionEmbeddingCollectionSharder(
        EmbeddingCollectionSharder(), ManagedCollisionCollectionSharder())
    topology = Topology(world_size=world, compute_device="cuda")
    planner = EmbeddingShardingPlanner(topology=topology)
    plan = planner.collective_plan(model, [sharder], dist.group.WORLD)  # row_wise + fused on GPU
    if rank == 0:
        print(f"[plan]\n{plan}")

    dmp = DistributedModelParallel(  # [CONFIRM] signature
        module=model, device=device, plan=plan, sharders=[sharder],
    )

    sd = dmp.state_dict()
    if rank == 0:
        print("[state_dict] key :: type :: metadata")
        for k, v in sd.items():
            print(f"  {k} :: {type(v).__name__}")
            if hasattr(v, "metadata"):  # ShardedTensor
                md = v.metadata()
                print(f"      global size={tuple(md.size)}")
                for s in md.shards_metadata:
                    print(f"      shard offset={tuple(s.shard_offsets)} "
                          f"size={tuple(s.shard_sizes)} @ {s.placement}")
            elif torch.is_tensor(v):
                print(f"      shape={tuple(v.shape)} dtype={v.dtype}")

    dcp.save(sd, checkpoint_id=OUT)  # [CONFIRM] collective DCP save
    dist.destroy_process_group()


@app.function(gpu="T4", volumes={"/data": vol}, timeout=60 * 20)
def generate() -> str:
    import os
    import shutil

    import torch.multiprocessing as mp

    shutil.rmtree(OUT, ignore_errors=True)  # avoid stale shards from a previous run
    os.makedirs(OUT, exist_ok=True)
    mp.spawn(_worker, args=(WORLD,), nprocs=WORLD, join=True)
    vol.commit()

    print(f"[dcp] {OUT} contents:")
    for f in sorted(os.listdir(OUT)):
        print(f"  {f}  ({os.path.getsize(os.path.join(OUT, f))} bytes)")
    from torch.distributed.checkpoint import FileSystemReader
    md = FileSystemReader(OUT).read_metadata()  # DCP's own metadata format (not torch.load-able)
    print("[dcp metadata] fqn :: type :: global size :: chunks (offset/size):")
    for fqn, m in md.state_dict_metadata.items():
        size = getattr(m, "size", None)
        print(f"  {fqn} :: {type(m).__name__} :: size={tuple(size) if size is not None else None}")
        for c in getattr(m, "chunks", None) or []:
            print(f"      chunk offset={tuple(c.offsets)} size={tuple(c.sizes)}")
    print(f"DONE {OUT}")
    return OUT


@app.local_entrypoint()
def main():
    print("sharded discovery ->", generate.remote())
