# flamediff

A tool for detecting and attributing structural drift across raw model checkpoints dropped over a
training run — reading weights off disk, no forward pass.
See [`plan.md`](plan.md) for the design, [`ROADMAP.md`](ROADMAP.md) for the product plan, and
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the built system.

**▶ [Live example report](https://neelvad.github.io/flamediff/)** — the browsable HTML view
(`flamediff report --html`): trajectory sparklines with anomaly markers, ranked events, and
drill-down into each event's *why*. Hover a chart for the checkpoint step + value.

**Supported format (the one it reads today):** dynamic managed-collision embedding tables (recsys /
TorchRec MCH-ZCH), from single-device `state_dict`s and sharded DCP checkpoints — the diff is an
id-keyed join over checkpoints rather than a row-index subtract.

> **v1 scope (honest limits):**
> - **Scale.** Reads both single-device and **sharded (DCP)** TorchRec checkpoints, locally; a
>   large weight loads **out-of-core** (reassembled into an mmap scratch file, auto by size) so a
>   bigger-than-RAM checkpoint stays bounded. The remaining limit is the *diff result*: the id-join
>   and per-id arrays still materialize, so billion-*id* diffs aren't streamed yet (and Stage 2
>   copies the weight to scratch rather than mmapping the `.distcp` chunks directly).
> - **Calibration.** The shipped `flamediff/calibration.json` is derived from **clean real-scale
>   runs** (10 stationary TorchRec MCH runs, dim=64, 2 tables; see its `provenance`). Regenerate
>   for your own data with `scripts/calibrate_real.py` (Modal) or `scripts/calibrate.py` (synthetic).
> - **One format.** Only the TorchRec MCH/ZCH adapter exists; static-hash (regime A) is deferred.

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
```

Each anomaly reads as *"step N, `table.metric`, 3.1× over the calibrated bar — idiosyncratic
drift (global 2%, popularity 24%, residual 74%); movers …"* (or a churn breakdown for
insertion/eviction spikes).

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
