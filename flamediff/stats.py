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

    Fit log(||delta||) ~ a + b*log1p(dcount) and return the MAD-z-scored residual.
    Positive = moved more than its update count predicts; negative = trained but barely moved.
    """
    delta_norm = np.asarray(delta_norm, dtype=np.float64)
    dcount = np.asarray(dcount, dtype=np.float64)
    if delta_norm.size == 0:
        return np.zeros(0, dtype=np.float64)
    y = np.log(np.maximum(delta_norm, _EPS))
    x = np.log1p(np.maximum(dcount, 0.0))
    if y.size < 3 or x.std() < _EPS:
        resid = y - np.median(y)
    else:
        coef = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y, rcond=None)[0]
        resid = y - (coef[0] + coef[1] * x)
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    return (resid - med) / (1.4826 * mad + _EPS)
