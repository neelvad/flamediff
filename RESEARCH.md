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
`flamediff.spectral.project_deltas` does exactly that; the sweep (fully seeded, so the baseline
columns reproduce bit-for-bit) adds it at an automatic 90%-energy rank and at a fixed `2×RANK`
oracle.

**It doesn't.** Projection is flat everywhere (±0.006 of raw ‖Δ‖), including at `DIM = 64`
(0.548 raw → 0.546/0.552 projected). The `rank90` column says why: the embedding covariance is
*not* concentrated near the true task rank — at `DIM = 64` its 90%-energy subspace is **42-dim**
(vs task rank 4). The variance the table carries is spread across most of the space (noise +
under-convergence), so "the table's top-variance subspace" is not "the behavioral subspace", and
covariance-based projection removes almost nothing that raw ‖Δ‖ didn't already keep. Even the
fixed `r = 8` oracle doesn't help — the behaviorally-relevant directions are evidently not the
top-*variance* directions of the table itself.

The sharpened hypothesis this leaves: behavioral relevance in a dot-product model is defined by
the *other tower* (movement matters where the co-embeddings have mass), so the right projection
basis is the interaction-weighted one — e.g. the covariance of the *opposing* table, or a
CCA-style joint basis — not the table's own. That needs the co-table at diff time (flamediff has
it: both tables are in the same checkpoint) and is the natural next experiment. Meanwhile the
practical reading of the dilution stands, unrepaired: keep embeddings tight if you want weight
diffs to track behavior.

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
  top-variance subspace doesn't recover the diluted signal; behavioral relevance lives in the
  interaction (the other tower), not in the table's variance structure.

## Limitations

Toy scale; a single seed per dim (effect sizes are small relative to run-to-run noise); one model
class (MF dot-product); the frozen-panel probe is one behavioral view among many. A stronger version
would average over seeds, decouple convergence from dim (train each to a matched loss), and add a real
downstream head. Left as future work — the pipeline (`scripts/behavioral_probe.py`) supports it.
