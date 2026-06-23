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
