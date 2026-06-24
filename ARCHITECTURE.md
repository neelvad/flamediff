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
   `_mch_counts`) is the per-interval update count, free in the checkpoint. Over the ids that
   actually moved, fit `log(||Δ||) ≈ a + b·log1p(dcount)` and take the MAD-z-scored residual,
   clipped: high-positive = moved more than its training predicts. (Fit on movers only; the
   zipf tail of non-movers would otherwise collapse the regression and the scale.)
5. **Frozen score** (the complementary signal): `pctrank(dcount) − pctrank(||Δ||)`, a rank-based
   "trained but didn't move" (saturated/dead) measure that, unlike the residual, also scores the
   zero-inflated non-movers.
6. **Geometry** (scale-safe): `dim×dim` row-covariance eigenspectrum → effective rank,
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

## Trajectory detection (stages 1–3)

Turns the pairwise diff into run-level monitoring — *which step* deviates from the run's own
history. The pairwise `diff.py` is untouched and pure-measurement; detection is a separate consumer.

- `flamediff/trajectory.py` — **stage 1+2**: `diff_trajectory(checkpoints)` runs consecutive
  pairwise diffs; `step_features` collapses each `CheckpointDiff` to scalar `(table, metric)`
  features → a `MetricSeries` per metric. Features: churn rates, movement percentiles, scorer
  tails (`freq_resid_max`, `n_freq_resid_hi`, `frozen_max`), geometry; dense tensors too.
- `flamediff/detect.py` — **stage 3**: `detect_trajectory(traj)` runs three detectors per series
  and returns `Event`s ranked by severity:
  - `robust_z` — point spikes: `(value − trailing_median) / global_robust_scale`. Trailing
    median tracks level (slow drift is PH/PELT's job); the *global* robust scale is a stable
    noise floor that avoids the tiny-trailing-window-MAD blow-up.
  - `page_hinkley` — online sequential drift on the standardized series.
  - `pelt` — offline changepoint segmentation (`ruptures`), severity = standardized mean-shift.

**Two-level drill-down:** an `Event` carries `(index, table, metric)`; `traj.diffs[index]` is the
pairwise diff for that step, whose per-id arrays (`top_movers`/`top_frozen`) give attribution.

Detection judges deviation in **noise-floor units**, so `k` / `window` / `pelt_pen` are the knobs
a future calibration sweep tunes. Cross-series joint detection (Mahalanobis) is a deliberate v2.

**Testing:** synthetic series with planted spikes / level-shifts / drift (unit); a trajectory
mutation test injects a known corruption at a mid-run checkpoint and asserts the detector flags
that step above the run's noise floor (integration, gated on `fixtures/`).
