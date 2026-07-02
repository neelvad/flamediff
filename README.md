# flamediff

A tool for detecting and attributing structural drift across raw model checkpoints dropped over a
training run — reading weights off disk, no forward pass.
See [`plan.md`](plan.md) for the design, [`ROADMAP.md`](ROADMAP.md) for the product plan,
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the built system, and [`RESEARCH.md`](RESEARCH.md) for the
behavioral-probe research note (does cheap weight-diff predict behavioral drift?).

**▶ [Live example report](https://neelvad.github.io/flamediff/)** — the browsable HTML view
(`flamediff report --html`): trajectory sparklines with anomaly markers, ranked events, and
drill-down into each event's *why*. Hover a chart for the checkpoint step + value.

**Supported format (the one it reads today):** dynamic managed-collision embedding tables (recsys /
TorchRec MCH-ZCH), from single-device `state_dict`s and sharded DCP checkpoints — the diff is an
id-keyed join over checkpoints rather than a row-index subtract.

> **v1 scope (honest limits):**
> - **Scale.** Reads single-device and **sharded (DCP)** TorchRec checkpoints, locally. Large
>   weights are read **out-of-core, zero-copy** — mmapping the `.distcp` chunks directly (with a
>   mmap-scratch fallback), auto by size — and the diff gathers survivors in batches, so a
>   bigger-than-RAM checkpoint stays bounded. The remaining limit is the *diff result*: the per-id
>   summary arrays still materialize, so a genuine billion-*id* diff needs a streaming (t-digest)
>   reduction — noted, not built.
> - **Calibration.** The shipped `flamediff/calibration.json` is derived from **clean real-scale
>   runs** (10 stationary TorchRec MCH runs, dim=64, 2 tables; see its `provenance`). Regenerate
>   for your own data with `scripts/calibrate_real.py` (Modal) or `scripts/calibrate.py` (synthetic).
> - **One format.** Only the TorchRec MCH/ZCH adapter exists; static-hash (regime A) is deferred.

## Quickstart

```bash
git clone https://github.com/neelvad/flamediff
cd flamediff
uv sync                 # create .venv and install deps (managed with uv, Python 3.12)
uv run flamediff --help
```

The reference **fixtures are large and not checked in** (see [Reference fixtures](#reference-fixtures)
to generate them). The easiest first look needs nothing to run: the
**[live example report](https://neelvad.github.io/flamediff/)**, or generate your own HTML with
`flamediff report <run_dir> --html out.html`.

## Usage

The product entry point is `flamediff report` — it fuses calibrated anomaly detection with
attribution over a run's checkpoints, so each flagged event comes with *why* it drifted:

```bash
flamediff report <run_dir>                    # ranked anomalies, each with a 'why' line
flamediff report <run_dir> --min-severity 5   # focus on the strongest
flamediff report <run_dir> --json --md out.md # machine + shareable outputs
flamediff report <run_dir> --html out.html    # a browsable, self-contained web view
flamediff report <run_dir> --fail-on 5        # exit nonzero past a severity (CI gate)

flamediff watch <run_dir> --interval 600      # stream NEW anomalies as checkpoints drop
flamediff watch <run_dir> --fail-on 8         # guard a live run; exit nonzero on severe drift

flamediff serve <run_dir> --interval 600      # live browsable dashboard, auto-refreshing

flamediff rank <run_dir>                      # low-rank structure: rank-at-energy trajectories,
                                              # factorization advisory (how small, and safe when?)
```

Sample output — correlated signals are grouped into **incidents** (one underlying cause fires many
series at once), and every signal carries a *why* (a churn breakdown, or the drift attribution):

```text
flamediff report — run_1782312586  (24 ckpts, 2 tables)  cal=REAL (FPR 0.05)

INCIDENTS (calibrated severity ≥ 1, most severe first):
  ▌ steps 250–450  worst 17.6×  (27 signals, 2 tables)
    ● step 250  author_id_emb.inserted_rate  17.6×  [page_hinkley]
         why: churn down: 1195 inserted / 1195 evicted / 0 re-admitted / 4 slot-moved
         also: video_id_emb.inserted_rate 16.7×, video_id_emb.evicted_rate 4.8×, +24 more
  ▌ steps 550–700  worst 2.4×  (18 signals, 2 tables)
    ● step 600  author_id_emb.evicted_rate  2.4×  [pelt]
         why: churn down: 28 inserted / 28 evicted / 0 re-admitted / 0 slot-moved
         also: video_id_emb.evicted_rate 2.3×, author_id_emb.evicted_rate 2.1×, +15 more

SUMMARY: 2 incidents (45 signals) across 2 tables; worst step 250 (author_id_emb.inserted_rate 17.6×)
```

## Development

Managed with [uv](https://docs.astral.sh/uv/) (Python 3.12, pinned in `.python-version`):

```bash
uv sync                 # create .venv and install deps + dev group
uv run pytest -q        # unit tests (add nothing) / integration (needs fixtures/)
uv run ruff check .
uv run scripts/run_diff.py        # pairwise diff over a downloaded trajectory
uv run scripts/mutation_demo.py   # detection-power demo
uv run scripts/detect_demo.py     # ranked trajectory anomaly events
uv run scripts/attribution_demo.py # why drift happened (de-confounded) + injection proof
uv run scripts/calibrate.py       # calibration sweep -> power report + calibration.json
uv run flamediff-tui              # interactive TUI to browse events (tui extra / dev group)
```

## Reference fixtures

flamediff diffs *consecutive* checkpoints, so we develop against a generated
**trajectory** of TorchRec managed-collision checkpoints (with real insertion /
eviction / re-admission between them) plus a ground-truth sidecar.

The generator runs on [Modal](https://modal.com) (TorchRec is Linux/CUDA-first and
won't build on arm64 macOS):

```bash
modal run scripts/generate_checkpoints.py
# then download the trajectory the run prints, e.g.:
modal volume get flamediff-fixtures run_<ts> ./fixtures/
```

The first run also doubles as an API/serialization discovery run — it prints the
torch/torchrec/fbgemm versions and dumps the managed-collision module's buffer layout,
which is what the parser must read.
