"""Frozen reusable-ranking Stage-I frontend used by GCRSS.

Only the ranking constants and the minimal frontend wrapper are included in
this code release.  No baseline, evaluator, dataset, or benchmark runner is
part of this module.
"""

from __future__ import annotations

import numpy as np

from .ranking_core import CORE


ALPHA = 0.2
TAU = 0.5
LAMBDA_DIV = 1.2
K_SET = (8, 12, 16)
REFERENCE_K = 12


class ReusableRanker:
    """Train-only reusable ranking frontend."""

    def __init__(self):
        self.rank_calls = 0
        self.last_diagnostics = None

    def rank(self, X_fs_train, *, n_clusters=None, random_state=None) -> np.ndarray:
        if n_clusters is not None:
            raise ValueError("ReusableRanker must not receive n_clusters")
        x_std = np.asarray(X_fs_train, dtype=float)
        if (
            x_std.ndim != 2
            or x_std.shape[0] < 2
            or x_std.shape[1] < 1
            or not np.isfinite(x_std).all()
        ):
            raise ValueError("ReusableRanker input must be a finite non-empty training matrix")
        self.rank_calls += 1
        x_l2 = CORE.l2_rows(x_std)
        rank = min(64, x_l2.shape[1], x_l2.shape[0] - 1)
        z = CORE.l2_rows(
            CORE.PCA(rank, svd_solver="randomized", random_state=0).fit_transform(x_l2)
        )
        scores = np.column_stack(
            [CORE.lap_score(x_l2, CORE.graph_euclidean(z, k)) for k in K_SET]
        )
        a = CORE.pct(scores)
        c = CORE.corr_abs(x_l2)
        quality = CORE.base_score(a, ALPHA, TAU)
        ranking = CORE.diversity_rank(quality, c, LAMBDA_DIV, max_m=50, pool=500)
        self.last_diagnostics = {
            "n_samples": x_std.shape[0],
            "n_features": x_std.shape[1],
            "pca_rank": rank,
            "graph_scales": list(K_SET),
            "reference_scale": REFERENCE_K,
            "l2_before_pca": True,
            "feature_score_space": "X_l2_original_features",
            "redundancy_space": "X_l2_original_features",
            "graph_space": "PCA_X_l2",
        }
        return np.asarray(ranking, dtype=np.int64)
