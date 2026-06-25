"""Modal app: produce a *sharded* TorchRec checkpoint and discover its on-disk serialization
-- the foundation for flamediff's ShardedTable / v1.0 real-scale work.

DISCOVERY run (like generate_checkpoints.py was). It shards a plain EmbeddingBagCollection across
2 ranks (gloo / CPU via torch.multiprocessing.spawn -- robust, avoids NCCL-on-one-GPU), saves via
torch.distributed.checkpoint (DCP), and dumps: torch/torchrec versions, the sharding plan, the
sharded state_dict structure (ShardedTensor shard metadata), and the DCP output layout (files +
the .metadata planner record). Managed-collision sharding is the follow-up once the format is known.

[CONFIRM] marks sharding/DCP API surface to validate against torchrec 1.0.0 on first run.

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

WORLD = 2
OUT = "/data/sharded_run"


def _worker(rank: int, world: int) -> None:
    import os

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)  # CPU/gloo: robust

    import torchrec
    from torchrec.distributed import DistributedModelParallel
    from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.modules.embedding_configs import EmbeddingBagConfig
    from torchrec.modules.embedding_modules import EmbeddingBagCollection

    if rank == 0:
        print(f"[versions] torch={torch.__version__} torchrec={torchrec.__version__} "
              f"world_size={world}")

    features = ["author_id", "video_id"]
    zch, dim, _vocab = 2000, 16, 5000
    configs = [EmbeddingBagConfig(name=f"{f}_emb", embedding_dim=dim, num_embeddings=zch,
                                  feature_names=[f]) for f in features]
    ebc = EmbeddingBagCollection(tables=configs, device=torch.device("meta"))  # [CONFIRM] meta

    sharder = EmbeddingBagCollectionSharder()
    # Planner defaults to table-wise here (each table whole on a rank) -> a real cross-rank
    # sharded checkpoint. (ROW_WISE -- the realistic recsys intra-table split -- needs the GPU
    # fused kernel / multi-GPU; the on-disk DCP chunk metadata generalises to N shards either way.)
    topology = Topology(world_size=world, compute_device="cpu")
    planner = EmbeddingShardingPlanner(topology=topology)
    plan = planner.collective_plan(ebc, [sharder], dist.group.WORLD)  # [CONFIRM]
    if rank == 0:
        print(f"[plan]\n{plan}")

    dmp = DistributedModelParallel(  # [CONFIRM] signature
        module=ebc, device=torch.device("cpu"), plan=plan, sharders=[sharder],
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
