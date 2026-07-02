import glob

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from flamediff import stats
from flamediff.cli import app
from flamediff.diff import diff_table
from flamediff.spectral import (
    _stable_since,
    covariance_weighted_deltas,
    project_deltas,
    spectral_report,
)
from flamediff.types import Checkpoint

RUN = "fixtures/run_1782312586"


def _lowrank_weights(num_slots, dim, rank, seed=0, noise=1e-3):
    """Rows living (up to tiny noise) in the span of the first `rank` coordinate axes."""
    g = torch.Generator().manual_seed(seed)
    W = torch.zeros(num_slots, dim)
    W[:, :rank] = torch.randn(num_slots, rank, generator=g)
    return W + noise * torch.randn(num_slots, dim, generator=g)


def test_rank_at_energy():
    spec = torch.tensor([10.0, 5.0, 1.0, 0.0])
    assert stats.rank_at_energy(spec, 0.60) == 1   # 10/16 = 0.625
    assert stats.rank_at_energy(spec, 0.90) == 2   # 15/16 = 0.9375
    assert stats.rank_at_energy(spec, 0.99) == 3
    assert stats.rank_at_energy(spec, 1.0) == 3    # the zero tail never helps
    assert stats.rank_at_energy(torch.zeros(4)) == 0


def test_subspace_overlap_identity_and_rotation():
    vals, vecs = stats.row_covariance_eig(_lowrank_weights(500, 16, rank=4))
    assert stats.subspace_overlap(vecs, vals, vecs, 4) == pytest.approx(1.0)
    # rotate the dominant plane out of its span: swap the top-4 axes with 4 null-space axes
    perm = list(range(16))
    perm[:4], perm[4:8] = perm[4:8], perm[:4]
    rot = vecs[:, perm]
    assert stats.subspace_overlap(rot, vals, vecs, 4) < 0.05


def test_diff_table_flags_basis_rotation(make_table):
    ids = np.arange(600)
    a = make_table(ids=ids, dim=16, weights=_lowrank_weights(600, 16, rank=4))
    same = diff_table(a, a)
    assert same.subspace_overlap == pytest.approx(1.0)
    assert same.geom_cur.rank95 <= 5                       # near the true rank, not the dim
    # move every row's energy into fresh axes: same per-row norms, rotated basis
    W = a.copy_weights()
    W = torch.cat([W[:, 8:], W[:, :8]], dim=1)
    rotated = diff_table(a, a.with_weights(W))
    assert rotated.subspace_overlap < 0.05


def test_project_deltas_removes_nullspace_motion(make_table):
    ids = np.arange(400)
    W = _lowrank_weights(400, 16, rank=4, noise=0.0)
    a = make_table(ids=ids, dim=16, weights=W)
    # all movement in the null space of the used basis -> projected drift ~0, raw drift large
    Wn = W.clone()
    Wn[:, 8:] += 1.0
    proj = project_deltas(a, a.with_weights(Wn), ids)
    raw = (Wn - W).norm(dim=1).numpy()
    assert raw.min() > 1.0 and proj.max() < 1e-4
    # the same magnitude of movement inside the used subspace survives projection
    Ws = W.clone()
    Ws[:, :4] += 1.0
    proj_in = project_deltas(a, a.with_weights(Ws), ids)
    assert proj_in.min() > 0.5


def test_project_deltas_with_co_tower_basis(make_table):
    ids = np.arange(400)
    W = _lowrank_weights(400, 16, rank=4, noise=0.0)
    a = make_table(ids=ids, dim=16, weights=W)
    # the co-table's mass lives in axes 8..12, NOT where table a's own variance lives (axes 0..4)
    co_W = torch.zeros(300, 16)
    co_W[:, 8:12] = torch.randn(300, 4, generator=torch.Generator().manual_seed(1))
    co = make_table("co", ids=np.arange(300), dim=16, weights=co_W)
    # movement in a's own top-variance axes but invisible to the co-table -> removed by the
    # co-tower basis (which a's own basis would have kept)
    Wm = W.clone()
    Wm[:, :4] += 1.0
    proj_co = project_deltas(a, a.with_weights(Wm), ids, basis_table=co, rank=4)
    proj_own = project_deltas(a, a.with_weights(Wm), ids, rank=4)
    assert proj_co.max() < 1e-4 and proj_own.min() > 0.5
    # movement where the co-table has mass survives the co-tower basis
    Wv = W.clone()
    Wv[:, 8:12] += 1.0
    assert project_deltas(a, a.with_weights(Wv), ids, basis_table=co, rank=4).min() > 0.5


def test_covariance_weighted_deltas_matches_definition(make_table):
    g = torch.Generator().manual_seed(2)
    W, Wn = torch.randn(200, 8, generator=g), torch.randn(200, 8, generator=g)
    co_W = torch.randn(500, 8, generator=g)
    a = make_table(ids=np.arange(200), dim=8, weights=W)
    co = make_table("co", ids=np.arange(500), dim=8, weights=co_W)
    got = covariance_weighted_deltas(a, a.with_weights(Wn), np.arange(200), co)
    C = torch.from_numpy(np.cov(co_W.numpy(), rowvar=False)).float()
    want = torch.einsum("nd,de,ne->n", Wn - W, C, Wn - W).clamp_min(0).sqrt().numpy()
    np.testing.assert_allclose(got, want, rtol=1e-4)


def test_stable_since():
    steps = [100, 200, 300, 400, 500]
    assert _stable_since([9, 5, 5, 6, 5], steps) == 200    # within +-1 of final from index 1
    assert _stable_since([3, 9, 5, 9, 5], steps) is None   # still moving at the tail
    assert _stable_since([5, 5, 5, 5, 5], steps) == 100


def test_spectral_report_tracks_rank_growth(make_table):
    ids = np.arange(500)
    cks = []
    for i, r in enumerate((2, 4, 4, 4)):                   # rank grows, then plateaus
        t = make_table(ids=ids, dim=16, weights=_lowrank_weights(500, 16, rank=r, seed=i))
        cks.append(Checkpoint(path=f"ck{i}", step=i * 100, embedding_tables={"t": t}))
    (ts,) = spectral_report(cks)
    r95 = ts.rank_series[0.95]
    assert r95[0] < r95[-1] <= 6                           # growth visible, plateau near true rank
    assert ts.stable_since == 100                          # settled from the second checkpoint
    assert ts.energy_curve[-1] == pytest.approx(1.0)
    assert 0 < ts.final_rank(0.90) <= ts.final_rank(0.99)
    d = ts.to_dict()
    assert d["rank_at_energy"]["0.95"] == r95


def test_cli_rank_html_is_self_contained(write_ckpt, tmp_path):
    import json
    import re

    g = torch.Generator().manual_seed(0)
    base = torch.zeros(100, 16)
    base[:, :3] = torch.randn(100, 3, generator=g)
    for i in range(3):
        write_ckpt(f"ckpt_{i:03d}",
                   {"t": {"ids": np.arange(80), "slots": np.arange(80),
                          "weight": base + 0.01 * i * torch.randn(100, 16, generator=g)}},
                   step=i * 100)
    out = tmp_path / "rank.html"
    res = CliRunner().invoke(app, ["rank", str(tmp_path), "--html", str(out)])
    assert res.exit_code == 0
    html = out.read_text()
    assert html.startswith("<!doctype html>")
    assert "http://" not in html and "https://" not in html  # no external deps -> works offline
    data = json.loads(re.search(r'application/json">(.*?)</script>', html, re.S)
                      .group(1).replace("<\\/", "</"))
    (t,) = data["tables"]
    assert t["dim"] == 16 and len(t["rank_at_energy"]["0.95"]) == 3
    assert t["rank_at_energy"]["0.95"][-1] <= 5              # the planted rank-3 structure


@pytest.mark.integration
def test_cli_rank_on_fixture():
    if len(glob.glob(f"{RUN}/ckpt_*")) < 2:
        pytest.skip("no trajectory fixture under fixtures/run_*")
    res = CliRunner().invoke(app, ["rank", RUN])
    assert res.exit_code == 0
    assert "rank needed" in res.stdout and "energy at rank" in res.stdout
    js = CliRunner().invoke(app, ["rank", RUN, "--json"])
    assert js.exit_code == 0 and '"rank_at_energy"' in js.stdout
