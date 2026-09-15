"""Minimal frozen reusable-ranking primitives used by GCRSS.

This module intentionally contains no benchmark runner, baseline, evaluator,
dataset, or result code.  It exposes only the train-matrix operations needed
to build the reusable ranking consumed by GCRSS.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


EPS = 1e-12


def l2_rows(x: np.ndarray) -> np.ndarray:
    return normalize(np.asarray(x, dtype=float), norm="l2", axis=1)


def _knn(z: np.ndarray, k: int):
    k = min(max(1, int(k)), len(z) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
    dist, ind = nn.kneighbors(z)
    return dist[:, 1:], ind[:, 1:]


def graph_euclidean(z: np.ndarray, k: int):
    from scipy import sparse

    dist, ind = _knn(z, k)
    n, kk = ind.shape
    sigma = np.maximum(dist[:, -1], EPS)
    rows = np.repeat(np.arange(n), kk)
    cols = ind.ravel()
    ds = dist.ravel()
    vals = np.exp(-(ds**2) / (sigma[rows] * sigma[cols] + EPS))
    directed = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    weight = 0.5 * (directed + directed.T)
    weight.setdiag(0)
    weight.eliminate_zeros()
    return weight.tocsr()


def lap_score(x: np.ndarray, weight) -> np.ndarray:
    degree = np.asarray(weight.sum(1)).ravel()
    degree_sum = degree.sum()
    mu = (degree[:, None] * x).sum(0) / max(degree_sum, EPS)
    centered = x - mu
    denominator = (degree[:, None] * centered * centered).sum(0)
    numerator = denominator - (centered * (weight @ centered)).sum(0)
    score = np.full(x.shape[1], np.inf)
    valid = denominator > EPS
    score[valid] = numerator[valid] / denominator[valid]
    return score


def percentile_rank(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    ranked = np.empty_like(scores)
    dimension = scores.shape[0]
    for q in range(scores.shape[1]):
        ranked[:, q] = (rankdata(scores[:, q], method="average") - 1) / max(
            dimension - 1, 1
        )
    return ranked


def corr_abs(x: np.ndarray) -> np.ndarray:
    centered = np.asarray(x, dtype=float) - np.asarray(x, dtype=float).mean(0)
    centered /= np.linalg.norm(centered, axis=0, keepdims=True) + EPS
    correlation = np.abs(centered.T @ centered)
    np.fill_diagonal(correlation, 0)
    return np.clip(correlation, 0, 1).astype(np.float32)


def entropic(ranked_scores: np.ndarray, tau: float) -> np.ndarray:
    return tau * (
        logsumexp(ranked_scores / tau, axis=1)
        - np.log(ranked_scores.shape[1])
    )


def base_score(ranked_scores: np.ndarray, alpha: float, tau: float) -> np.ndarray:
    return (1 - alpha) * ranked_scores[:, 1] + alpha * entropic(ranked_scores, tau)


def diversity_rank(
    score: np.ndarray,
    correlation: np.ndarray,
    lambda_div: float,
    max_m: int = 50,
    pool: int = 500,
) -> np.ndarray:
    """Return a complete feature permutation using frozen diversity tie-breaking."""

    relative = (rankdata(score) - 1) / max(len(score) - 1, 1)
    base = np.lexsort((np.arange(relative.size, dtype=np.int64), relative))
    pool_idx = base[: min(pool, len(base))]
    selected = [int(pool_idx[0])]
    remaining = set(map(int, pool_idx[1:]))
    while len(selected) < min(max_m, len(pool_idx)):
        candidates = np.fromiter(remaining, dtype=int)
        penalty = correlation[candidates][:, selected].mean(1)
        objective = relative[candidates] + lambda_div * penalty
        chosen = int(candidates[np.lexsort((candidates, objective))[0]])
        selected.append(chosen)
        remaining.remove(chosen)
    selected_set = set(selected)
    rest = [int(j) for j in base if j not in selected_set]
    return np.asarray(selected + rest, dtype=np.int64)


class _Core:
    """Namespace matching the frozen frontend's minimal core API."""

    l2_rows = staticmethod(l2_rows)
    PCA = staticmethod(PCA)
    graph_euclidean = staticmethod(graph_euclidean)
    lap_score = staticmethod(lap_score)
    pct = staticmethod(percentile_rank)
    corr_abs = staticmethod(corr_abs)
    base_score = staticmethod(base_score)
    diversity_rank = staticmethod(diversity_rank)


CORE = _Core()
