# flamediff

A tool for tracking structural changes across raw model checkpoints (PyTorch /
safetensors) dropped over a training run — reading weights off disk, no forward pass.
See [`plan.md`](plan.md) for the design and scope.

v1 target: **dynamic managed-collision embedding tables** (recsys / TorchRec MCH-ZCH),
where the diff is an id-keyed join over checkpoints rather than a row-index subtract.

## Development

Managed with [uv](https://docs.astral.sh/uv/) (Python 3.12, pinned in `.python-version`):

```bash
uv sync                 # create .venv and install deps + dev group
uv run pytest -q        # unit tests (add nothing) / integration (needs fixtures/)
uv run ruff check .
uv run scripts/run_diff.py        # pairwise diff over a downloaded trajectory
uv run scripts/mutation_demo.py   # detection-power demo
uv run scripts/detect_demo.py     # ranked trajectory anomaly events
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
