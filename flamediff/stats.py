"""Owned commodity statistics: norms, geometry, and the frequency-residual scorer.

torch for tensor math (GPU-able later), numpy for the small regression in the scorer.
"""
from __future__ import annotations

import numpy as np
import torch

_EPS = 1e-12


# --- per-row (per-id) quantities --------------------------------------------------------
def row_delta_norm(prev: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
    return (cur - prev).norm(dim=1)


def row_cosine(prev: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
    num = (prev * cur).sum(dim=1)
    den = prev.norm(dim=1) * cur.norm(dim=1) + _EPS
    return num / den


# --- table geometry (from the dim x dim row covariance; cheap regardless of #rows) ------
def row_covariance_eigvals(W: torch.Tensor) -> torch.Tensor:
    if W.shape[0] < 2:
        return torch.zeros(W.shape[1])
    Wc = W - W.mean(dim=0, keepdim=True)
    cov = (Wc.T @ Wc) / (W.shape[0] - 1)
    return torch.linalg.eigvalsh(cov).clamp_min(0.0)


def row_covariance_eig(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigenvalues (descending) and matching eigenvectors (columns) of the row covariance."""
    dim = W.shape[1]
    if W.shape[0] < 2:
        return torch.zeros(dim), torch.eye(dim)
    Wc = W - W.mean(dim=0, keepdim=True)
    cov = (Wc.T @ Wc) / (W.shape[0] - 1)
    vals, vecs = torch.linalg.eigh(cov)  # ascending
    return vals.flip(0).clamp_min(0.0), vecs.flip(1)


def rank_at_energy(spectrum_desc: torch.Tensor, energy: float = 0.95) -> int:
    """Smallest r whose top-r eigenvalues capture >= `energy` of the total (0 for a zero spectrum).

    The factorization-rank question directly: how small can a low-rank approximation be while
    keeping `energy` of the table's variance."""
    total = float(spectrum_desc.sum())
    if total <= _EPS:
        return 0
    cum = torch.cumsum(spectrum_desc, 0) / total
    return int(torch.searchsorted(cum, torch.tensor(energy - 1e-9)).item()) + 1


def subspace_overlap(
    eigvecs_prev: torch.Tensor, eigvals_cur: torch.Tensor, eigvecs_cur: torch.Tensor, r: int
) -> float:
    """Fraction of cur's top-r variance energy captured by prev's top-r eigenbasis, in [0, 1]
    (1 = the dominant subspace did not move).

    Energy-weighted, so it is robust to within-subspace rotation and eigenvalue ties, unlike raw
    principal angles: v' C_cur v over prev's top-r directions equals sum_k lambda_k (v.u_k)^2,
    so only the eigendecompositions are needed, never C_cur itself."""
    r = max(1, min(int(r), eigvecs_prev.shape[1]))
    den = float(eigvals_cur[:r].sum())
    if den <= _EPS:
        return 1.0
    M = eigvecs_prev[:, :r].T @ eigvecs_cur  # [r, dim] cosines between the two bases
    num = float(((M**2) * eigvals_cur.unsqueeze(0)).sum())
    return min(1.0, num / den)


def mean_row_norm(W: torch.Tensor) -> float:
    if W.shape[0] == 0:
        return 0.0
    return float(W.norm(dim=1).mean())


def effective_rank_from_spectrum(spectrum: torch.Tensor) -> float:
    """exp(entropy of the normalized spectrum) -- the effective dimensionality."""
    total = float(spectrum.sum())
    if total <= _EPS:
        return 0.0
    p = spectrum / total
    p = p[p > _EPS]
    return float(torch.exp(-(p * p.log()).sum()))


def anisotropy_from_spectrum(spectrum: torch.Tensor) -> float:
    """Top eigenvalue over the mean -- how stretched the cloud is (1.0 = isotropic)."""
    mean = float(spectrum.mean())
    if mean <= _EPS:
        return 0.0
    return float(spectrum.max()) / mean


# --- whole-tensor quantities (dense weights) --------------------------------------------
def _as_2d(W: torch.Tensor) -> torch.Tensor:
    return W if W.ndim == 2 else W.reshape(W.shape[0], -1)


def tensor_delta_norm(prev: torch.Tensor, cur: torch.Tensor) -> float:
    return float((cur - prev).norm())


def tensor_cosine(prev: torch.Tensor, cur: torch.Tensor) -> float:
    a, b = prev.flatten(), cur.flatten()
    return float((a @ b) / (a.norm() * b.norm() + _EPS))


def spectral_norm(W: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(_as_2d(W), ord=2))


def matrix_effective_rank(W: torch.Tensor) -> float:
    s = torch.linalg.svdvals(_as_2d(W))
    return effective_rank_from_spectrum(s)


# --- the frequency-residual score (the differentiated signal) ---------------------------
def freq_residual(delta_norm: np.ndarray, dcount: np.ndarray) -> np.ndarray:
    """How much each id moved relative to what its training frequency predicts.

    Fit log(||delta||) ~ a + b*log1p(dcount) over the ids that actually moved and return
    the MAD-z-scored, clipped residual. Positive = moved more than its update count predicts;
    negative = moved less. The fit/scale use movers only: the realistic zipf tail of ids that
    never move (||delta|| == 0) is a separate population that would otherwise dominate the
    regression and collapse the scale -- those score 0 here ("trained but frozen" is left to a
    dedicated metric).
    """
    delta_norm = np.asarray(delta_norm, dtype=np.float64)
    dcount = np.asarray(dcount, dtype=np.float64)
    out = np.zeros(delta_norm.size, dtype=np.float64)
    movers = delta_norm > 0.0
    if int(movers.sum()) < 3:
        return out
    y = np.log(delta_norm[movers])
    x = np.log1p(np.maximum(dcount[movers], 0.0))
    if x.std() < _EPS:
        resid = y - np.median(y)
    else:
        coef = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y, rcond=None)[0]
        resid = y - (coef[0] + coef[1] * x)
    med = np.median(resid)
    scale = 1.4826 * np.median(np.abs(resid - med))
    if scale < _EPS:  # movers all fit perfectly -> fall back to their std
        scale = float(resid.std())
    if scale < _EPS:
        return out
    out[movers] = np.clip((resid - med) / scale, -25.0, 25.0)
    return out


def _avg_rank(a: np.ndarray) -> np.ndarray:
    """1-based average ranks (ties share the mean rank); like scipy.stats.rankdata('average')."""
    a = np.asarray(a, dtype=np.float64)
    sorter = np.argsort(a, kind="stable")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size)
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    counts = np.r_[np.flatnonzero(obs), a.size]
    return 0.5 * (counts[dense] + counts[dense - 1] + 1)


def frozen_score(delta_norm: np.ndarray, dcount: np.ndarray) -> np.ndarray:
    """Trained-but-didn't-move signal: pctrank(dcount) - pctrank(||delta||), in [-1, 1].

    High = updated a lot relative to how little it moved (saturated/frozen/dead). Rank-based,
    so it is robust to the zero-inflated movement distribution and needs no threshold, and -
    unlike the residual scorer - it meaningfully scores the non-movers too.
    """
    delta_norm = np.asarray(delta_norm, dtype=np.float64)
    dcount = np.asarray(dcount, dtype=np.float64)
    n = delta_norm.size
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    pr_train = (_avg_rank(dcount) - 1.0) / (n - 1)
    pr_move = (_avg_rank(delta_norm) - 1.0) / (n - 1)
    return pr_train - pr_move


# --- attribution: separate global basis drift / popularity / idiosyncratic change ------------
def procrustes_align(prev: torch.Tensor, cur: torch.Tensor) -> dict:
    """Orthogonal-Procrustes align `cur` onto `prev` (rotation + translation) over shared rows.

    Returns the per-row *aligned* drift norm (movement after removing the table-wide rotation
    and mean shift) plus an energy decomposition of the total drift -- translation / rotation /
    aligned-residual fractions that sum to 1. The SVD is dim x dim, so it is cheap at any #rows.
    """
    n, dim = prev.shape
    if n < 2:
        return {"aligned_delta_norm": np.zeros(n), "mean_shift_norm": 0.0,
                "rotation_magnitude": 0.0, "frac_translation": 0.0,
                "frac_rotation": 0.0, "frac_aligned": 1.0}
    mean_shift = cur.mean(0) - prev.mean(0)
    ap = prev - prev.mean(0)
    ac = cur - cur.mean(0)
    ss_total = float(((cur - prev) ** 2).sum()) + _EPS
    ss_centered = float(((ap - ac) ** 2).sum())
    # R minimizes ||ap - ac @ R|| over orthogonal R  (M = ac^T ap = U S Vt -> R = U Vt)
    u, _s, vt = torch.linalg.svd(ac.T @ ap)
    R = u @ vt
    aligned_delta = ap - ac @ R
    ss_aligned = float((aligned_delta ** 2).sum())
    return {
        "aligned_delta_norm": aligned_delta.norm(dim=1).cpu().numpy(),
        "mean_shift_norm": float(mean_shift.norm()),
        "rotation_magnitude": float((R - torch.eye(dim, dtype=R.dtype)).norm()),
        "frac_translation": n * float((mean_shift ** 2).sum()) / ss_total,
        "frac_rotation": (ss_centered - ss_aligned) / ss_total,
        "frac_aligned": ss_aligned / ss_total,
    }


def loglog_residual(y_norm: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Multi-covariate generalization of `freq_residual`: fit log(y) ~ [1, X] over movers (y>0)
    and return the MAD-z-scored, clipped residual per row (0 for non-movers) plus the fit R^2.

    With X = log1p(dcount, count), the residual is drift *not* explained by popularity churn --
    the idiosyncratic "meaning changed" signal. R^2 is how much popularity explains.
    """
    y_norm = np.asarray(y_norm, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    out = np.zeros(y_norm.size, dtype=np.float64)
    movers = y_norm > 0.0
    if int(movers.sum()) < X.shape[1] + 3:
        return out, 0.0
    yv = np.log(y_norm[movers])
    A = np.column_stack([np.ones(int(movers.sum())), X[movers]])
    coef = np.linalg.lstsq(A, yv, rcond=None)[0]
    resid = yv - A @ coef
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > _EPS else 0.0
    med = np.median(resid)
    scale = 1.4826 * np.median(np.abs(resid - med))
    if scale < _EPS:
        scale = float(resid.std())
    if scale < _EPS:
        return out, r2
    out[movers] = np.clip((resid - med) / scale, -25.0, 25.0)
    return out, r2
