# Research note — does weight-space drift predict *behavioral* drift?

*flamediff branch E (behavioral probes). Reproduce: `modal run scripts/behavioral_probe.py`.*

## The question

flamediff monitors recsys checkpoints by diffing **weights** — cheaply, with no forward pass. That is
only useful if weight-space drift actually tracks how the model **behaves**. So: does it? And branch
B's de-confounded attribution claims to find ids whose *meaning* changed — does that hold up against
real behavioral change?

## Method

- **Model:** a real recsys model — TorchRec managed-collision embeddings + a **dot-product
  interaction** (matrix-factorization / two-tower style). The dot product forces the interaction to
  live in the *embeddings* (a flexible MLP head instead just absorbs it — see the methodology notes).
- **A controlled, popularity-independent meaning change:** the target is a rank-`R` latent dot
  product `latent_a · latent_v`. Midway through training we **flip** (`latent → −latent`) a random,
  popularity-spanning subset of author ids. Those ids must *relearn* — a genuine meaning change that
  is, by construction, independent of how popular / how-much-trained an id is (the confound).
- **Behavioral measurement:** at each checkpoint, score a fixed **canary** set of ids against a
  **frozen panel**, so the change isolates *that id's own* embedding rather than the shared,
  moving panel.
- **The test:** does the weight-space signal — raw `‖Δ‖`, and flamediff's de-confounded **residual** —
  identify the ids that behaviorally changed (the known flipped set), by AUC over the immediate
  post-flip relearning window?
- **The sweep:** hold the task fixed, vary the embedding dimension `DIM ∈ {8, 16, 32, 64}` against the
  true latent rank `R = 4`.

## Result

*(Single seed — kept for the narrative; the **multi-seed replication** section below is the
authoritative version with error bars.)*

The **projected** columns are the follow-up experiment (see *Does subspace projection repair the
dilution?* below): per-id drift projected onto the table's dominant covariance eigenbasis before
scoring — at an automatic 90%-energy rank (`proj@e90`), and at a fixed `2×RANK = 8` oracle
(`proj@r8`). `rank90` is the automatic rank chosen (how concentrated the covariance actually was).

| `DIM` (rank = 4) | final train loss | `rank90` | AUC behavior | **AUC weight (raw ‖Δ‖)** | AUC residual | AUC proj@e90 | AUC proj@r8 |
|---|---|---|---|---|---|---|---|
| 8  | 0.36 | 5  | 0.562 | **0.591** | 0.591 | 0.584 | 0.591 |
| 16 | 0.33 | 9  | 0.586 | **0.604** | 0.596 | 0.606 | 0.607 |
| 32 | 0.42 | 18 | 0.567 | **0.593** | 0.598 | 0.595 | 0.594 |
| 64 | 0.76 | 42 | 0.536 | **0.548** | 0.554 | 0.546 | 0.552 |

## Findings (honest)

1. **Weight-space drift predicts behavioral change — above chance at every dim (~0.55–0.60).** The
   core premise of cheap weight-diff monitoring holds in this setup.
2. **It degrades with over-parameterization.** `DIM = 64` (16× the true rank) is clearly weakest
   (0.548) vs ~0.59–0.60 for the tighter dims — consistent with the hypothesis that when `DIM ≫ rank`,
   most weight movement is in the **behaviorally-irrelevant null space** and dilutes the signal. The
   slope is *gentle*, not dramatic; AUCs this close to chance carry ~±0.05 run-to-run noise.
3. **A convergence confound, stated plainly:** the `DIM = 64` model also *under-converged* in the same
   budget (loss 0.76 vs ~0.35). So its weaker signal is partly null-space, partly just noisier
   embeddings — the two are not cleanly separated here.
4. **The de-confounded residual ≈ raw `‖Δ‖`** throughout — no advantage in this *behavioral* regime,
   because the bottleneck is the null space, not popularity. (The residual's advantage *does* appear
   when popularity is the dominant confound: in branch B's injection test it lifts recovery AUC from
   0.21 to 0.999. Different regime, different bottleneck.)

**Bottom line:** a *weak-to-moderate positive with a real caveat*. Cheap weight-diff tracks behavior
when embeddings are reasonably tight; the link dilutes as they over-parameterize. Not a slam dunk —
which is exactly why it's worth shipping the result either way.

## Does subspace projection repair the dilution? (follow-up — a clean negative)

If finding 2's mechanism is right — at `DIM ≫ rank`, most movement is behaviorally-irrelevant
null-space motion — then projecting each id's Δ onto the table's *dominant covariance eigenbasis*
before scoring should discard the null-space component and recover the AUC at `DIM = 64`.
`flamediff.spectral.project_deltas` does exactly that; the sweep adds it at an automatic
90%-energy rank and at a fixed `2×RANK` oracle. (The sweep is seeded, but GPU training is not
bit-for-bit deterministic: across reruns the baseline AUCs drift by up to ~±0.013 — that is the
noise floor any per-run difference below should be read against.)

**It doesn't.** Projection is flat everywhere (±0.006 of raw ‖Δ‖), including at `DIM = 64`
(0.548 raw → 0.546/0.552 projected). The `rank90` column says why: the embedding covariance is
*not* concentrated near the true task rank — at `DIM = 64` its 90%-energy subspace is **42-dim**
(vs task rank 4). The variance the table carries is spread across most of the space (noise +
under-convergence), so "the table's top-variance subspace" is not "the behavioral subspace", and
covariance-based projection removes almost nothing that raw ‖Δ‖ didn't already keep. Even the
fixed `r = 8` oracle doesn't help — the behaviorally-relevant directions are evidently not the
top-*variance* directions of the table itself.

The sharpened hypothesis this left: behavioral relevance in a dot-product model is defined by
the *other tower* (movement matters where the co-embeddings have mass), so the right projection
basis is the interaction-weighted one — the covariance of the *opposing* table, not the table's
own. Tested next.

## Is the co-tower basis the right one? (second follow-up — also negative)

The interaction-weighted scorers (`flamediff.spectral`, both deployable at diff time since both
tables are in the same checkpoint): Δ projected onto the **video** tower's dominant eigenbasis
(`xproj`, auto 90%-energy rank and fixed `2×RANK`), and the untruncated covariance-weighted norm
`√(Δᵀ C_video Δ)` (`xweight`) — how much the id's *scores* against a typical co-embedding move,
computed purely from weights.

| `DIM` | loss | AUC behavior | **AUC raw ‖Δ‖** | proj@r8 (own) | xproj@e90 | xproj@r8 | xweight |
|---|---|---|---|---|---|---|---|
| 8  | 0.35 | 0.552 | **0.591** | 0.591 | 0.588 | 0.591 | 0.581 |
| 16 | 0.32 | 0.565 | **0.591** | 0.589 | 0.584 | 0.585 | 0.585 |
| 32 | 0.41 | 0.577 | **0.589** | 0.591 | 0.588 | 0.586 | 0.583 |
| 64 | 0.73 | 0.540 | **0.552** | 0.556 | 0.552 | 0.561 | 0.557 |

**Also flat.** At `DIM = 64` the best interaction-weighted variant (`xproj@r8`, 0.561) sits
+0.009 over raw — inside the ±0.013 cross-run noise — and nowhere near the tight-dim ~0.59.
Two observations make this a *coherent* negative rather than a puzzle:

1. **The behavioral score itself agrees.** The frozen-panel fingerprint delta *is* a
   panel-sampled readout-weighted norm (`‖Δ·P₀ᵀ‖`), and its AUC is *below* raw ‖Δ‖ at every dim.
   `xweight` is the population version of the same quantity (all 20k co-embeddings instead of a
   128-panel), and it lands in the same place. Readout-weighting is self-consistent — it just
   isn't a better identifier of the planted flip than raw movement.
2. **Why it can't win here:** the co-tower is trained on the same task with the same
   over-parameterization, so *its* covariance is nearly as diffuse as the author table's
   (`rank90 = 42/64` on both sides). Weighting by a near-isotropic covariance is a no-op. The
   dilution isn't "movement in a subspace nobody reads" — under Adam, movement *and* readout mass
   are both spread across the whole space.

**Bottom line for branch E, now settled:** no weight-space reweighting we tested — own-basis
projection, co-tower projection, full interaction-weighted norm, or the popularity-de-confounded
residual — beats raw ‖Δ‖ at identifying behaviorally-changed ids in the over-parameterized
regime. The dilution appears irreducible within linear reweightings of the weight diff. The
practical conclusions stand: (a) keep embeddings tight if you want weight diffs to track
behavior, and (b) when you need behavioral *precision*, measure behavior — a frozen-panel probe
tier on nominated ids, not a cleverer weight norm.

## Multi-seed replication (n = 5) — the numbers that stand

Everything above is a single seed. This section is **5 full independent replications** — new
latents, flip set, canaries, data stream, and init per seed, run as parallel containers
(`experiment.map`). Two meta-lessons first: cross-seed spread is ±0.02–0.03 AUC (about twice the
seeded-GPU rerun noise), and the original seed turns out to have been a *below-average draw* —
absolute AUCs run ~0.03–0.05 higher on average than the tables above. Levels move; none of the
conclusions do.

| `DIM` | loss | AUC behavior | **AUC raw ‖Δ‖** | residual | proj@r8 (own) | xproj@r8 (co) | xweight |
|---|---|---|---|---|---|---|---|
| 8  | 0.37±0.04 | 0.631±0.013 | **0.633±0.017** | 0.597±0.032 | 0.633±0.017 | 0.633±0.017 | 0.633±0.014 |
| 16 | 0.33±0.04 | 0.646±0.022 | **0.656±0.018** | 0.619±0.034 | 0.657±0.019 | 0.656±0.019 | 0.654±0.022 |
| 32 | 0.41±0.04 | 0.631±0.028 | **0.631±0.018** | 0.604±0.032 | 0.631±0.021 | 0.630±0.021 | 0.631±0.024 |
| 64 | 0.69±0.06 | 0.596±0.040 | **0.597±0.025** | 0.583±0.033 | 0.599±0.028 | 0.600±0.028 | 0.600±0.031 |

Paired **within-seed** contrasts — the honest test, since cross-seed task variance cancels:

| claim | contrast | result |
|---|---|---|
| **dilution is real** | raw@16 − raw@64 | **+0.059 ± 0.020 — 5/5 seeds > 0** |
| own-basis projection repairs it | proj@r8 − raw, @64 | +0.001 ± 0.008 (2/5) — null |
| co-tower projection repairs it | xproj@r8 − raw, @64 | +0.002 ± 0.007 (3/5) — null |
| interaction-weighted norm repairs it | xweight − raw, @64 | +0.003 ± 0.010 (3/5) — null |
| the residual helps in this regime | resid − raw, @64 | −0.014 ± 0.016 (1/5) — if anything it *hurts* |

With error bars, the story is: **weight diff tracks behavior at ~0.60–0.66 AUC; going from
`DIM=16` to `DIM=64` costs ~0.06 AUC in every seed; no linear reweighting of the weight diff gets
any of it back; and the popularity-residual slightly hurts when popularity isn't the confound.**
The `DIM=64` convergence confound also replicates in every seed (loss 0.69 vs ~0.33–0.41), so
"null-space dilution" and "under-convergence" remain entangled at the top dim.

## What the iteration taught us (methodology)

Getting a *valid* behavioral probe out of a toy model surfaced real subtleties, each a lesson:

- A flexible **MLP head absorbs** a relationship change without moving the embeddings → invisible to a
  weight diff. Use a dot-product interaction so the change must live in the embeddings.
- A learnable **global scale absorbs the magnitude**, keeping embeddings near-zero → no signal. Drop
  it; make the embeddings carry the interaction.
- Dot-product **MF with SGD stalls** (vanishing gradients at small init) → use Adam.
- A **shared, moving panel** swamps per-id behavior → freeze the panel to isolate each id.
- **Over-parameterization** (`DIM ≫ rank`) → null-space drift dilutes the weight↔behavior link.
- A table's **own covariance eigenbasis is not its behavioral basis** — projecting Δ onto the
  top-variance subspace doesn't recover the diluted signal.
- Neither is the **co-tower's**: under the same over-parameterized training, the co-table's
  covariance is just as diffuse, so interaction-weighting is a near-no-op. Movement and readout
  mass dilute *together* — the fix for precision is a behavioral probe tier, not a better norm.
- **Seeded ≠ deterministic on GPU**: baseline AUCs drift ~±0.013 across identical reruns. Any
  single-run difference below that is noise; multi-seed averaging is the real fix.

## Limitations

Toy scale; one model class (MF dot-product); the frozen-panel probe is one behavioral view among
many. Multi-seed replication is now done (n = 5, above) — the remaining upgrades are decoupling
convergence from dim (train each to a matched loss; the `DIM=64` under-convergence confound
replicates in every seed), a real downstream head, and a semi-real dataset (MovieLens/Criteo-class,
with a real distribution shift instead of a planted flip). The pipeline
(`scripts/behavioral_probe.py`) supports all of it.
