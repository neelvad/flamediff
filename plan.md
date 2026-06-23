# flamediff — plan

## What this is
A tool for tracking structural changes across raw model checkpoints (PyTorch /
safetensors) dropped over a training run (e.g. hourly). Reads weights off disk — no
forward pass, no hooks, no torch.compile coupling. Goal: surface *what* changed between
checkpoints and rank it by how unusual it is relative to the run's own trajectory.

## Layers (priority order)

### v1 — structural-what (the core, build first)
Detect and attribute *what* moved between consecutive checkpoints, from weights alone.

- **Dense weights:** tractable with the standard stats toolkit — per-tensor ‖Δ‖,
  spectral norm, effective rank, cosine of update direction, etc. This is the part
  everyone pictures, and it's the easy part.
- **Large embedding tables: the actual hard problem, and the defensible core.** Naive
  global matrix norms are useless. Two regimes, mirror images:

  | regime | identity across ckpts | map needed? | per-id attribution |
  |---|---|---|---|
  | A: static hash (`hash(id) mod N`) | stable | no (recompute hash) | contaminated (collisions) |
  | B: dynamic managed-collision (eviction) | unstable (slot reuse) | yes — and it exists (model unservable without it) | clean (collisions handled) |

  **v1 target = Regime B (dynamic managed-collision). Static (A) deferred.**

  Core diff primitive is an **id-keyed join, not a row-index subtract** (slot i ≠ slot i
  across checkpoints). Per managed-collision feature, between N and N−1:
  - join id→slot maps → split ids into **survivors / inserted / evicted / re-admitted**.
  - **churn (insert/evict/re-admit rates) is a first-class structural signal** — cheap set
    cardinalities, robust; a spike = a real event (vocab pressure, eviction thrash, id-dist shift).
  - survivor per-id Δ (‖Δ‖, cosine), **frequency-normalized** — and the eviction policy's own
    access counters (LRU/LFU state already in the checkpoint) double as the frequency normalizer.
  - **re-admission confound:** an evicted-then-readmitted id looks like a survivor but its
    embedding was reset on re-admission → huge spurious Δ ("reborn," not "learned"). Filter via
    generation/admission metadata; clean comparison set = "in both AND not re-admitted." Report
    re-admission rate as a contamination floor when generation metadata is absent.
  - table-geometry drift over the comparable set: anisotropy, covariance spectrum, effective
    rank, mean-norm — point-cloud summaries, not per-coordinate deltas.
  - report at two granularities: slot level (always clean) and id level (quantify contamination).
  - cost: the id-join is itself a large distributed/out-of-core join (maps can be billions of
    entries, co-sharded with the weights); exploit existing sharding, diff shard-local.

  **Adapter contract (system-specific, thin):** parse from a checkpoint (1) weight `[slots,dim]`
  [universal], (2) id→slot map [load-bearing, format-specific: TorchRec ZCH / internal / HKV],
  (3) per-id access/frequency count [nice-to-have, often in eviction metadata], (4) generation/
  admission stamp [nice-to-have, for re-admission filtering]. (3)/(4) degrade gracefully.

  **Confirmed serialization (TorchRec 1.0.0, `MCHManagedCollisionModule`), per table — validated
  against a generated fixture via `scripts/inspect_fixtures.py`:**
  - id→slot map = two parallel int64 arrays: `_mch_sorted_raw_ids` (raw ids, sorted asc, empty
    slots = `_delimiter` = INT64_MAX) and `_mch_remapped_ids_mapping` (slot per raw id). Clean
    ZCH — one slot per id, no collisions.
  - `_mch_counts` `[zch_size]` = LFU access count per entry == the frequency normalizer, for free.
  - `_current_iter_tensor`, `_mch_slots`, `_delimiter`, `_output_segments_tensor` = scalars/meta.
  - weights at state_dict key `_embedding_module.embeddings.{table}.weight` `[zch_size, dim]`.
  - **Eviction does NOT reset the embedding** — a reused slot keeps its vector, so a newly
    admitted / re-admitted id *inherits* the prior occupant's embedding. Per-id Δ across an
    eviction boundary is dominated by this slot-inheritance discontinuity, not training (fixture:
    re-admitted id ~155x the survivor-median ‖Δ‖). The diff must gather by id AND treat
    insert/evict/re-admit spans as comparability breaks, not learning signal.

### Cross-cutting technique (portable asset)
Noise-floor / null calibration: report drift in units of an instrument noise floor,
z-score against a resampled null, apply look-elsewhere corrections. Reused everywhere.

## Deferred / later

### Correlational-why via event-log join — DEFERRED
Attribute anomalies to run events (config/data/code/hardware changes). Out of scope for v1.
Why it's hard: the log often doesn't exist as a joinable stream; lives across 4–5
heterogeneous systems; step-vs-wallclock alignment + smeared causal lag; failure mode is
*confident-wrong* attribution (look-elsewhere — something always lines up, worse the better
the logging). Plan: **do NOT build ingestion.** Expose a simple join API — user supplies a
timestamped/step-stamped event stream; we align it to the anomaly timeline and attach a
"surprise" score (vs a permuted-timestamp null). Optional enrichment that degrades gracefully
when absent. Never claim causation — present as ranked coincident events.

### Behavioral-what via frozen probes — v2
Forward-pass-per-checkpoint over a fixed canary set; frozen difference-of-means / linear /
PCA probes. Reintroduces runnable model + harness + curated-input-set (specification variance
is its own research problem). Reach for it only when structural-what proves insufficient.

### Research ceiling (not on critical path)
Crosscoders (per-comparison SAE, activation space) and LLC / dev-interp (SGLD, expensive) are
the high-signal ceiling. Existence proofs, not bolt-ons.
