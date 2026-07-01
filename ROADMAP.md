# flamediff — roadmap

The product plan: where we are, where we're going, and why. (See [`plan.md`](plan.md) for the
design rationale and [`ARCHITECTURE.md`](ARCHITECTURE.md) for the built system.)

## Strategy

Go **deep, not broad**: become *the* tool for **recsys embedding-table drift monitoring** — a
niche we've confirmed is essentially empty — rather than a general checkpoint-diff utility. The
moat is the curated leading-indicator signals + calibrated detection + managed-collision
diffing, not format coverage.

Goal: a **real, usable product**. And deliberately dual: every milestone ships a short, honest
**writeup of its result** — because "built a real useful thing" and "got a clear positive/negative
result and communicated it concisely" are both first-class deliverables.

Principles carried through every version: the diff core stays **pure-measurement**; adapters
stay **thin**; **calibrate everything** (report drift in noise-floor units); ship a result note
per milestone.

## v0.1 — validated prototype (where we are)

The full pipeline exists and is validated end-to-end on generated TorchRec MCH checkpoints:
adapter → id-keyed diff (de-confounded frequency-residual + frozen scorers, geometry,
re-admission-aware) → trajectory series → robust-z / Page-Hinkley / PELT detection
(FPR-calibrated, comparably ranked) → TUI; with a mutation harness + calibration sweep proving
detection power.

**The risky part — does the signal / detection actually work? — is done.** What remains is
engineering and scope, not research.

Honest limits: in-memory scale (~10⁴ ids), one format (TorchRec MCH/ZCH), synthetic calibration.

## v1.0 — real-scale analyzer  · *the gate*

Make it run on actual production checkpoints. Nothing else matters until this is true.
- `ShardedTable` behind the existing `gather(ids)` Protocol; read **sharded TorchRec DMP**
  checkpoints (the real serialization, split across ranks).
- **Shard-local merge join** on the co-sharded id→slot maps; out-of-core weight access.
- Streaming / sampled geometry (partly done already).
- **Real calibration** on a genuinely-clean real run, replacing the synthetic defaults.

*Result to ship:* it diffs + detects on a real production-scale table, with perf numbers.
*Gating risk:* the real distributed-systems work — the architecture supports it, it's execution.

## v1.5 — correlational "why"

Turn "what moved" into "what it coincides with" — the actionability jump.
- A thin **event-log join**: the user supplies a timestamped event stream (config / data-shard /
  code / hardware changes); we align it to the anomaly timeline.
- A **surprise score** via a permuted-timestamp null (look-elsewhere aware); rank coincident
  events; **never claim causation** — present ranked coincidences.

*Result to ship:* "anomaly at step N coincides with event E (surprise p < x)."
*Dependency:* only as good as whether the run emits a joinable event log.

## v2.0 — monitoring product

From "a tool I run" to "a system that watches for me."
- A runner over each checkpoint drop; **persist trajectory history** (real detection runway +
  cross-run baselines); notifications (Slack / webhook); optional **CI / deploy gate**.
- The shareable surface (HTML report / dashboard) for teams.

*Result to ship:* a continuous monitor + precision/recall on a real run's known incidents.
This is where the low-false-positive / calibration work pays off — alert fatigue is the killer.

## Branches (interest- or demand-driven, off the critical path)

- **D — format breadth.** Static-hash (regime A, common in DLRMs) first, then Merlin/HKV, LLM
  vocab, dense models. *After* there's adoption to broaden for — breadth on toy data is just more
  toys.
- **E — behavioral "what" (research spike).** Frozen difference-of-means probes over a fixed
  canary set. The research question: *does cheap weight-space drift predict behavioral drift, and
  can a frozen probe-bank catch behavioral regressions that an hourly weight-diff misses?* Ship a
  **research note with the result, positive or negative.** The most interesting / most research-y
  branch; reintroduces forward passes + canary curation, so it's a heavier, different thing.
  **Decoupled** from A/B/C — it's a self-contained experiment that can run as a parallel,
  time-boxed spike whenever the research signal is wanted, not strictly after the product is built.
- **F — visualization web UI.** *TensorBoard / Weights & Biases, but for checkpoints.* **v1 built**
  (`flamediff report --html`): a self-contained static page — vanilla JS + inline SVG, no server /
  build / deps — with the per-(table, metric) trajectory sparklines + anomaly markers, the ranked
  event list, and drill-down into each event's attribution / churn. It renders the same `Report`
  JSON the CLI emits, so it's a pure presentation layer over a stable seam. **v2 built**
  (`flamediff serve`): a thin stdlib HTTP server whose page fetches `/data.json` on an interval and
  re-renders in place, so the dashboard live-refreshes as checkpoints land (verified end-to-end).
  The natural demo/portfolio surface — turns "CLI + JSON" into something you *browse*, live.

## Non-goals (for now)

- The activation-capture parity gate (a different, fragile tool).
- Crosscoders / SAE model-diffing (the research ceiling).
- LLM-first (recsys-first; LLM vocab is a later breadth item).
