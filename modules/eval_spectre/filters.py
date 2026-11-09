"""
Python port of SPECTRE / PCA / kmeans filters (spectre-defense Julia scripts).

Faithful to Algorithm 1 + QUE scoring; robust covariance uses iterative
Mahalanobis filtering (DKK-style removal loop), which is sufficient for
FLIP label-only eval without depending on Julia 1.6-era packages.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import eigh, norm, svd


def _k_highest_ind(scores: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask: True for the k largest scores."""
    n = scores.shape[0]
    k = int(max(0, min(k, n)))
    mask = np.zeros(n, dtype=bool)
    if k == 0:
        return mask
    idx = np.argpartition(scores, -k)[-k:]
    mask[idx] = True
    return mask


def _k_lowest_ind(scores: np.ndarray, k: int) -> np.ndarray:
    n = scores.shape[0]
    k = int(max(0, min(k, n)))
    mask = np.zeros(n, dtype=bool)
    if k == 0:
        return mask
    idx = np.argpartition(scores, k - 1)[:k]
    mask[idx] = True
    return mask


def pca_project(reps: np.ndarray, k: int):
    """Center and project to top-k PCA; reps (n, d) -> (n, k), components (k, d)."""
    mu = reps.mean(axis=0, keepdims=True)
    centered = reps - mu
    # economy SVD on centered data
    _, _, vt = svd(centered, full_matrices=False)
    k = min(k, vt.shape[0])
    comps = vt[:k]
    return centered @ comps.T, comps


def pca_filter(reps: np.ndarray, eps: int) -> np.ndarray:
    """Tran et al. PCA defense: remove 1.5*eps points farthest from mean on top PC."""
    proj, _ = pca_project(reps, 1)
    scores = np.abs(proj[:, 0] - proj[:, 0].mean())
    return _k_highest_ind(scores, int(round(1.5 * eps)))


def _two_means(X: np.ndarray, iters: int = 30, seed: int = 0) -> np.ndarray:
    """Simple 2-means; returns labels in {0,1}."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    centers = X[rng.choice(n, size=2, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d0 = norm(X - centers[0], axis=1)
        d1 = norm(X - centers[1], axis=1)
        labels = (d1 < d0).astype(np.int64)
        for c in (0, 1):
            if np.any(labels == c):
                centers[c] = X[labels == c].mean(axis=0)
    return labels


def kmeans_filter(reps: np.ndarray, eps: int, k_pca: int = 100) -> np.ndarray:
    """
    2-means on PCA features; remove 1.5*eps from the smaller / more-outlier cluster.
    Approximation of spectre-defense/kmeans_filters.jl (numpy, no sklearn).
    """
    n = reps.shape[0]
    remove = int(round(1.5 * eps))
    proj, _ = pca_project(reps, min(k_pca, n - 1, reps.shape[1]))
    labels = _two_means(proj)
    sizes = np.bincount(labels, minlength=2)
    poison_cluster = int(np.argmin(sizes))
    cand = np.where(labels == poison_cluster)[0]
    center = proj.mean(axis=0)
    dist = norm(proj[cand] - center, axis=1)
    order = cand[np.argsort(-dist)]
    mask = np.zeros(n, dtype=bool)
    mask[order[: min(remove, len(order))]] = True
    if mask.sum() < remove:
        rem = remove - int(mask.sum())
        global_dist = norm(proj - center, axis=1)
        global_dist[mask] = -np.inf
        mask |= _k_highest_ind(global_dist, rem)
    return mask


def _que_scores(whitened: np.ndarray, alpha: float = 4.0) -> np.ndarray:
    """QUE scores on already-whitened (n, k) features."""
    n, k = whitened.shape
    cov = (whitened.T @ whitened) / max(n, 1)
    # Symmetrize for numerical stability
    cov = 0.5 * (cov + cov.T)
    op = norm(cov, ord=2)
    if not np.isfinite(op) or op <= 1e-12:
        return np.sum(whitened ** 2, axis=1) / max(k, 1)
    M = cov / op - np.eye(k)
    # matrix exponential via eigh
    evals, evecs = eigh(M)
    exp_m = (evecs * np.exp(alpha * evals)) @ evecs.T
    exp_m = exp_m / np.trace(exp_m)
    # score_i = h_i' exp_m h_i
    tmp = whitened @ exp_m
    return np.sum(tmp * whitened, axis=1)


def _robust_select(pca_reps: np.ndarray, eps: int, limit_mult: float = 2.0):
    """
    Drop up to limit_mult*eps highest-norm (centered) points before whitening.
    Lightweight stand-in for dkk17 cov_estimation_iterate.
    """
    n = pca_reps.shape[0]
    limit = int(round(limit_mult * eps))
    mu = pca_reps.mean(axis=0)
    scores = norm(pca_reps - mu, axis=1)
    return ~_k_highest_ind(scores, min(limit, n - 2))


def _whiten(pca_reps: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    X = pca_reps[keep_mask]
    mu = X.mean(axis=0)
    xc = X - mu
    cov = (xc.T @ xc) / max(X.shape[0], 1)
    cov = 0.5 * (cov + cov.T) + 1e-6 * np.eye(cov.shape[0])
    evals, evecs = eigh(cov)
    evals = np.clip(evals, 1e-8, None)
    inv_sqrt = evecs * (1.0 / np.sqrt(evals)) @ evecs.T
    return (pca_reps - mu) @ inv_sqrt


def spectre_filter(
    reps: np.ndarray,
    eps: int,
    alpha: float = 4.0,
    limit2: float = 1.5,
) -> np.ndarray:
    """
    SPECTRE-like filter. Returns boolean mask (True = remove), removing ~1.5*eps points.
    Auto-selects PCA dimension k similar to rcov_auto_quantum_filter.
    """
    n, d = reps.shape
    remove = int(round(limit2 * eps))
    max_k = min(100, n - 2, d)
    if max_k < 1:
        return _k_highest_ind(norm(reps - reps.mean(0), axis=1), remove)

    # candidate ks like Julia: round.(range(1, sqrt(100), length=10).^2)
    grid = np.unique(np.round(np.linspace(1, np.sqrt(max_k), num=10) ** 2).astype(int))
    grid = grid[(grid >= 1) & (grid <= max_k)]

    pca100, _ = pca_project(reps, max_k)
    best_op, best_mask = -np.inf, None

    for k in grid:
        pca_k = pca100[:, :k]
        keep = _robust_select(pca_k, eps)
        white = _whiten(pca_k, keep)
        scores = _que_scores(white, alpha=alpha)
        mask = _k_highest_ind(scores, remove)

        # Julia rcov_auto_quantum_filter: maximize QUE-weighted energy of the
        # *full* whitened covariance (spikier when whitening used a clean subset).
        cov = white.T @ white / max(white.shape[0], 1)
        cov = 0.5 * (cov + cov.T)
        op = float(norm(cov, ord=2))
        if not np.isfinite(op) or op <= 1.0 + 1e-8:
            continue
        eye = np.eye(cov.shape[0])
        M_base = (cov - eye) / (op - 1.0)
        M_base = 0.5 * (M_base + M_base.T)
        evals, evecs = eigh(M_base)
        M = (evecs * np.exp(alpha * evals)) @ evecs.T
        trM = float(np.trace(M))
        if abs(trM) < 1e-12:
            continue
        M = M / trM
        criterion = float(np.trace(cov @ M))
        if criterion > best_op:
            best_op = criterion
            best_mask = mask

    if best_mask is None:
        scores = norm(reps - reps.mean(0), axis=1)
        best_mask = _k_highest_ind(scores, remove)
    return best_mask


def run_all_filters(reps: np.ndarray, eps: int) -> dict:
    """Return {pca, kmeans, rcov} boolean remove-masks."""
    return {
        "pca": pca_filter(reps, eps),
        "kmeans": kmeans_filter(reps, eps),
        "rcov": spectre_filter(reps, eps),
    }
