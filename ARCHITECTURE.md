# flamediff architecture

v1 = a pairwise **structural diff** of two model checkpoints, specialised for recsys
dynamic managed-collision (TorchRec MCH/ZCH) embedding tables. See `plan.md` for scope.

## Data flow

```
checkpoint dir ──[adapter]──> Checkpoint ──┐
checkpoint dir ──[adapter]──> Checkpoint ──┴──[diff]──> CheckpointDiff ──> (future: detection)
                                  │                          │
                              types.py                    types.py
                                  └────── stats.py ────────┘
```

Two load-bearing boundaries:

1. **The diff *measures*; it does not decide "anomalous."** Ranking against a noise floor
   needs the run's own trajectory, so that is the deferred detection layer, which consumes a
   *sequence* of `CheckpointDiff`. Keeping the pairwise diff pure-measurement makes it
   shippable and composable.
2. **`gather(ids)` + `mmap` is the scale seam.** All heavy data access goes through by-id
   gather; an `MmapTable`/`ShardedTable` later satisfies the same `EmbeddingTable` Protocol
   and `diff.py` never changes.

## Modules

- `flamediff/types.py` — the normalized representation (adapter↔diff contract) and the
  diff-result dataclasses.
  - `EmbeddingTable` (Protocol): `ids()` (sorted), `slot_of(ids)`, `counts(ids)`, `gather(ids)`.
  - `InMemoryTable`: the v1 impl, backed by the same sorted parallel-array layout the format
    uses (`_sorted_ids`, `_slots`, `_counts`) plus an mmap-able `[num_slots, dim]` weight
    tensor — so `gather` reads only the rows it needs.
  - `DenseTensor`, `Checkpoint`, and the result types
    `EmbeddingTableDiff` / `DenseTensorDiff` / `CheckpointDiff` / `GeomStats`.
- `flamediff/stats.py` — owned commodity functions (torch + numpy): row delta-norm / cosine,
  covariance-eigenspectrum geometry (effective rank, anisotropy), dense spectral norm /
  effective rank, and the frequency-residual scorer.
- `flamediff/adapters/base.py` — `CheckpointAdapter` Protocol + a tiny registry.
- `flamediff/adapters/torchrec_mch.py` — parses a TorchRec MCEC **state_dict** (no
  `import torchrec`): groups keys by table, lifts the MCH buffers into `InMemoryTable`s.
- `flamediff/diff.py` — the pairwise algorithm.

## Diff algorithm (per managed-collision table)

1. **Id-keyed join** (sorted-array set ops): survivors / inserted / evicted.
2. **Slot-stability split**: survivors whose slot is unchanged are the *clean* set; survivors
   whose slot moved are comparability breaks (eviction inherits the slot's vector — a
   re-admitted id inherits a stranger's embedding), flagged and excluded from learning deltas.
3. **Clean deltas**: row `||Δ||` and cosine for slot-stable survivors, gathered by id.
4. **Frequency-residual score** (the differentiated signal): the popularity confound is that
   `||Δ||` tracks how often an id was trained. `dcount = count_cur - count_prev` (from the LFU
   `_mch_counts`) is the per-interval update count, free in the checkpoint. Fit
   `log(||Δ||) ≈ a + b·log1p(dcount)` and take the MAD-z-scored residual: high-positive moved
   more than its training predicts; high-negative trained but barely moved.
5. **Geometry** (scale-safe): `dim×dim` row-covariance eigenspectrum → effective rank,
   anisotropy, mean row-norm; reported prev vs cur.

Dense tensors get the standard `||Δ||`, relative `||Δ||`, cosine, spectral norm, effective rank.

## Public API

```python
from flamediff import load_checkpoint, diff_checkpoints
a = load_checkpoint("fixtures/run_.../ckpt_000")   # sniffs the adapter registry
b = load_checkpoint("fixtures/run_.../ckpt_001")
d = diff_checkpoints(a, b)
d.embedding_diffs["author_id_emb"].top_movers(10)   # by frequency-residual
```
