"""Spectral / low-rank structure over a run -- the factorization-advisory layer.

Per table: the eigenspectrum trajectory (from the same sampled row covariance the diff's geometry
uses), energy-at-rank curves, the rank needed at each energy target, and when in the run that rank
stabilized. This answers the compression-planning question directly: *how small could a low-rank
factorization of this table be, and is it safe to decide yet* -- factorizing before the rank
plateaus bakes in a dimensionality the table hasn't finished growing into.

Also home to ``project_deltas``: per-id drift projected onto a table's dominant subspace, removing
the behaviorally-irrelevant null-space component (see RESEARCH.md -- null-space movement is why raw
||delta|| dilutes as DIM >> true rank).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import torch

from flamediff import stats
from flamediff.types import Checkpoint, EmbeddingTable

_SAMPLE = 8192  # same row-sample cap as diff._geom
_ENERGIES = (0.90, 0.95, 0.99)


def table_spectrum(table: EmbeddingTable, sample: int = _SAMPLE) -> torch.Tensor:
    """Descending covariance eigenspectrum of a (sampled) table -- O(sample x dim + dim^3)."""
    ids = table.ids()
    n = int(ids.size)
    if n == 0:
        return torch.zeros(table.dim)
    if n > sample:
        ids = ids[np.random.default_rng(0).choice(n, sample, replace=False)]
    vals, _ = stats.row_covariance_eig(table.gather(ids).float())
    return vals


def project_deltas(
    prev: EmbeddingTable, cur: EmbeddingTable, ids: np.ndarray, *,
    energy: float = 0.90, sample: int = _SAMPLE,
) -> np.ndarray:
    """Per-id ||delta|| projected onto cur's dominant (energy-rank) covariance eigenbasis.

    Movement inside the subspace the table actually uses; the null-space component -- weight
    motion with no behavioral surface -- is removed.
    """
    all_ids = cur.ids()
    n = int(all_ids.size)
    basis_ids = (all_ids if n <= sample
                 else all_ids[np.random.default_rng(0).choice(n, sample, replace=False)])
    vals, vecs = stats.row_covariance_eig(cur.gather(basis_ids).float())
    r = max(1, stats.rank_at_energy(vals, energy))
    V = vecs[:, :r]  # [dim, r]
    delta = cur.gather(ids).float() - prev.gather(ids).float()
    return (delta @ V).norm(dim=1).cpu().numpy()


@dataclass
class TableSpectral:
    name: str
    dim: int
    n: int                      # resident ids at the last checkpoint
    steps: list[int]
    rank_series: dict           # energy -> [rank per checkpoint]
    effective_rank: list[float]
    energy_curve: list[float]   # cumulative energy fraction at rank 1..dim (last checkpoint)
    stable_since: int | None    # first step from which rank95 stays within +-1 of its final value
    meta: dict = field(default_factory=dict)

    def final_rank(self, energy: float) -> int:
        return self.rank_series[energy][-1]

    def to_dict(self) -> dict:
        return {
            "table": self.name, "dim": self.dim, "n": self.n, "steps": self.steps,
            "rank_at_energy": {f"{e:g}": r for e, r in self.rank_series.items()},
            "effective_rank": [round(x, 2) for x in self.effective_rank],
            "energy_curve": [round(x, 4) for x in self.energy_curve],
            "stable_since": self.stable_since,
        }

    def to_lines(self) -> list[str]:
        r90, r95, r99 = (self.final_rank(e) for e in _ENERGIES)
        er = self.effective_rank
        lines = [
            f"{self.name}  dim={self.dim}  n={self.n}",
            f"  rank needed: 90% -> {r90}, 95% -> {r95}, 99% -> {r99}   "
            f"(effective rank {er[0]:.1f} -> {er[-1]:.1f} over the run)",
        ]
        if self.stable_since is not None:
            lines.append(f"  rank95 stable since step {self.stable_since} "
                         f"— safe to size a factorization now")
        else:
            lines.append("  rank95 still moving — factorizing now would bake in a "
                         "dimensionality the table hasn't settled into")
        if r95 <= self.dim // 2:
            lines.append(f"  advisory: top-{r95} factors keep 95% of the variance "
                         f"({r95}/{self.dim} = {r95 / self.dim:.0%} of the parameters)")
        marks = " ".join(f"r={r}:{self.energy_curve[r - 1]:.0%}"
                         for r in (1, 2, 4, 8, 16, 32, 64, 128) if r <= self.dim)
        lines.append(f"  energy at rank: {marks}")
        return lines


def _stable_since(ranks: list[int], steps: list[int]) -> int | None:
    """First step from which the rank stays within +-1 of its final value; None if that window is
    shorter than 3 checkpoints (still moving / not enough evidence)."""
    final = ranks[-1]
    start = len(ranks)
    for i in range(len(ranks) - 1, -1, -1):
        if abs(ranks[i] - final) <= 1:
            start = i
        else:
            break
    if len(ranks) - start < 3:
        return None
    return steps[start]


def spectral_report(checkpoints: list[Checkpoint], *, table: str | None = None) -> list:
    """Per-table spectral trajectory over a run's checkpoints -> list[TableSpectral]."""
    out = []
    names = sorted(checkpoints[0].embedding_tables)
    steps = [c.step if c.step is not None else i for i, c in enumerate(checkpoints)]
    for name in names:
        if table is not None and name != table:
            continue
        spectra = [table_spectrum(c.embedding_tables[name]) for c in checkpoints]
        rank_series = {e: [stats.rank_at_energy(s, e) for s in spectra] for e in _ENERGIES}
        last = spectra[-1]
        total = float(last.sum())
        curve = ((torch.cumsum(last, 0) / total).tolist() if total > 0
                 else [0.0] * int(last.numel()))
        tbl = checkpoints[-1].embedding_tables[name]
        out.append(TableSpectral(
            name=name, dim=tbl.dim, n=int(tbl.ids().size), steps=steps,
            rank_series=rank_series,
            effective_rank=[stats.effective_rank_from_spectrum(s) for s in spectra],
            energy_curve=curve,
            stable_since=_stable_since(rank_series[0.95], steps),
        ))
    return out


def render_text(run: str, tables: list) -> str:
    lines = [f"flamediff rank — {run}  ({len(tables)} tables)", ""]
    for t in tables:
        lines += t.to_lines() + [""]
    return "\n".join(lines).rstrip("\n")


def render_json(run: str, tables: list) -> str:
    return json.dumps({"run": run, "tables": [t.to_dict() for t in tables]}, indent=2)
