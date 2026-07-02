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
- `flamediff/adapters/` — `base.py` (the `CheckpointAdapter` Protocol + registry) and two
  readers, both torchrec-free and sharing `_torchrec_common.assemble_checkpoint`:
  - `torchrec_mch.py` — single-device: `torch.load` a `state_dict.pt`.
  - `torchrec_mch_sharded.py` — **sharded (production)**: a DCP directory (`.metadata` +
    `__{rank}_{idx}.distcp`); `dcp.load` reassembles the row-wise `ShardedTensor`s (weight + map)
    into full tensors. Works because reassembled MCH buffers are structurally identical to
    single-device — **global** slots, **globally-sorted** ids (verified on 2 GPUs). Runs locally
    (DCP single-process; no GPU/torchrec/PG). Maps load to RAM; a weight above `out_of_core_bytes`
    is read out-of-core (automatic by size). **Stage 2.5 — zero-copy** (`_dcp_zerocopy.py`): DCP
    stores each tensor storage *uncompressed* inside a torch.save mini-zip per `.distcp`; we walk
    each chunk's central directory to find the storage entry and **mmap the shard at its raw bytes**,
    so `gather` pages in only the rows it touches with no copy. Falls back to a reassembled mmap
    **scratch** copy (Stage 2) on any unexpected framing.
- `flamediff/diff.py` — the pairwise algorithm.

## Diff algorithm (per managed-collision table)

1. **Id-keyed join** (sorted-array set ops): survivors / inserted / evicted.
2. **Slot-stability split**: survivors whose slot is unchanged are the *clean* set; survivors
   whose slot moved are comparability breaks (eviction inherits the slot's vector — a
   re-admitted id inherits a stranger's embedding), flagged and excluded from learning deltas.
3. **Clean deltas**: row `||Δ||` and cosine for slot-stable survivors, gathered by id — in
   **batches** (`gather_batch`, default 1<<20) so the dominant `[n, dim]` gather peaks at
   `batch × dim`, not `n × dim` (Stage 2.5; pairs with the zero-copy mmap weight for a bounded
   out-of-core diff). Bit-identical to a single-shot gather.
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

## Calibration sweep (detection power + FPR-calibrated thresholds)

`flamediff/calibrate.py` runs an offline mutation sweep to (1) measure detection power and
(2) calibrate the detector. Method — score once, threshold analytically:
- **Null:** run detectors permissively on stationary synthetic clean trajectories; pool the
  severities per method and the per-run max severity.
- **Power:** inject a known corruption (scramble / zero / noise / freeze; transient or
  persistent) at a random step and record the max severity at the injected location per method.
  TPR at any threshold is then a tail fraction — so power curves, ROC, and minimum-detectable
  effect fall out without re-running.

`scripts/calibrate.py` runs it and writes `flamediff/calibration.json`: per method, the
**FPR-calibrated threshold** (the `1 − target_fpr` quantile of the per-run-max null) and the
null severity quantiles.

**Wire-back (`detect.py`):** with `calibration.json` present (the default), `detect_trajectory`
generates candidates permissively, keeps those clearing each method's calibrated threshold, and
sets `Event.calibrated_severity = severity / threshold` (≥1 = over the bar) — a comparable,
tail-resolving severity it ranks by, so Page-Hinkley's large raw scale no longer skews the order.
Falls back to raw per-method thresholds + `|score|` ranking when no calibration is loaded.

Findings on the synthetic battery: scramble is detectable to ~0.25× row-norm; freeze is weaker;
PELT is weak on transient spikes but strong on persistent shifts (detector specialization).
(The shipped `calibration.json` is now **real-data-derived** — see `scripts/calibrate_real.py`.)

## Attribution — *why* a table drifted (`attribute.py`)

A separate analysis layer over a table diff: `attribute_table(prev, cur, diff)` re-gathers the
clean-survivor rows (via the Protocol, so it works on sharded/out-of-core tables) and decomposes
their drift, nested:
1. **Global basis drift** — orthogonal Procrustes (`stats.procrustes_align`) removes the table-wide
   rotation + mean shift; an *energy* split reports `frac_translation / frac_rotation /
   frac_aligned_residual` (sums to 1). A big rotation is itself a checkpoint-level health signal.
2. **Popularity churn** — `stats.loglog_residual` regresses the *aligned* per-id drift on
   `log dcount / log count` (multi-covariate `freq_resid`); `popularity_r2` is how much of the
   per-id drift popularity explains.
3. **Idiosyncratic residual** — the de-confounded score: ids whose representation genuinely changed,
   independent of how much they trained. `Attribution.top_movers()` ranks them.

Correlational, not causal — made trustworthy by **injection validation**: planting known
popularity-*independent* change and showing the residual recovers it where raw `‖Δ‖` can't. With a
strong confound the raw scorer is fooled (AUC 0.21) while the residual nails it (0.999); on the real
run (modest confound, popularity ≈ 25% of drift) the lift is smaller but real (0.884 → 0.964).
Honest limit: *coordinated* genuine drift is indistinguishable from basis drift (identifiability).
Scaling (similarity Procrustes) is a deferred toggle.

## Monitoring report (`report.py`, `cli.py`) — the operated product

One command fuses *when/where* (detection) with *why* (attribution). `build_report(checkpoints)`
runs trajectory → detect → attribute and attaches a `Why` to each calibrated event by metric kind:
- **churn** metrics (insert/evict/readmit/slot-move rates) → the diff's churn breakdown;
- **movement / geometry / scorer** metrics → the table's `Attribution` (global / popularity /
  idiosyncratic + top movers).

**Incident grouping:** one underlying cause (a data spike, a regime change) fires many series at
once — robust_z at the spike, PH/PELT a step or two later, across tables and metrics — so the flat
event list reads as dozens of anomalies when it is really a handful. `group_incidents` clusters
events by diff-index adjacency (chained, `gap=1`) into `Incident`s, each headlined by its strongest
signal; all renderers (text / JSON / markdown / HTML) present incidents as the primary unit, with
the constituent signals nested under them.

`Report` renders to text / JSON / markdown and exposes `worst_severity()`. CLI (Typer):
`flamediff report <run_dir>` with `--json`, `--md`, `--table`, `--min-severity`, and
`--fail-on <sev>` — a **CI gate** that exits nonzero past a severity (fail a training run on drift).
JSON + exit code are the integration seam; a cron/CI wires alerting.

`flamediff watch <run_dir>` streams **new** anomalies as checkpoints are dropped. `Watcher.poll()`
ingests new `ckpt_*`, re-detects on the accumulated scalar series, and surfaces each event once;
attribution is computed per step and cached, and only the **last checkpoint** is held (to diff the
next) — so memory is bounded regardless of run length. Polling is a cheap `glob`, so `--interval`
(default 60s) can be raised freely to match checkpoint cadence; `--fail-on` guards a live run.

`flamediff report --html` (`Report.to_html`, `_html.py`) renders the same `Report` into a
**self-contained static page** — vanilla JS + inline SVG, no server / build / deps, works offline —
mirroring the TUI: trajectory sparklines with anomaly markers, the ranked event list, and per-event
drill-down (attribution bars + movers, or churn). `flamediff serve` (`serve.py`) makes it **live**:
a thin stdlib HTTP server whose background thread polls the run dir via a `Watcher` and caches the
current `Report`; the page fetches `/data.json` on `--interval` and re-renders in place (no reload,
keeps the selected event). `render_html(live_poll_ms=…)` toggles static file vs live page.
