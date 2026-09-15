"""Adapter exposing shared-ranking data to budget-conditioned realization."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reusable_ranking import ALPHA, K_SET, LAMBDA_DIV, TAU, CORE


@dataclass(frozen=True)
class FrozenRankingFrontend:
    X_l2: np.ndarray
    A: np.ndarray
    quality: np.ndarray
    reusable_ranking: np.ndarray


def build_ranking_frontend(X_fs_train) -> FrozenRankingFrontend:
    X_l2 = CORE.l2_rows(np.asarray(X_fs_train, dtype=float))
    rank = min(64, X_l2.shape[1], X_l2.shape[0] - 1)
    Z = CORE.l2_rows(CORE.PCA(rank, svd_solver="randomized", random_state=0).fit_transform(X_l2))
    scores = np.column_stack([CORE.lap_score(X_l2, CORE.graph_euclidean(Z, k)) for k in K_SET])
    A = CORE.pct(scores)
    C = CORE.corr_abs(X_l2)
    quality = CORE.base_score(A, ALPHA, TAU)
    ranking = CORE.diversity_rank(quality, C, LAMBDA_DIV, max_m=50, pool=500)
    return FrozenRankingFrontend(X_l2, A, quality, np.asarray(ranking, dtype=np.int64))


def _corr_abs_local(X):
    return CORE.corr_abs(np.asarray(X, dtype=float))


def _jaccard(a, b):
    left, right = set(map(int, a)), set(map(int, b))
    return len(left & right) / len(left | right)
