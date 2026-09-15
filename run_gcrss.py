"""Canonical GCRSS selector entry point.

This module combines shared ranking construction with budget-conditioned subset
realization. It does not duplicate or modify the feature-selection algorithm.
"""
from __future__ import annotations

import numpy as np


EPSILON_REL = 0.01
MAX_SWAPS = 2
CANDIDATE_POOL_SIZE = 500
GRAPH_SCALES = (8, 12, 16)
REFERENCE_SCALE = 12
FROZEN_CONFIG = {
    "method": "GCRSS",
    "initialization": "frozen reusable ranking top-m",
    "candidate_pool": "frozen reusable ranking top-500",
    "epsilon_rel": EPSILON_REL,
    "max_swaps": MAX_SWAPS,
    "graph_scales": GRAPH_SCALES,
    "reference_scale": REFERENCE_SCALE,
    "alpha": 0.2,
    "tau": 0.5,
    "lambda_div": 1.2,
}


def assert_frozen_config() -> dict:
    """Validate and return the in-code final method contract."""

    from methods.reusable_ranking import ALPHA, K_SET, LAMBDA_DIV, REFERENCE_K, TAU

    assert FROZEN_CONFIG["initialization"] == "frozen reusable ranking top-m"
    assert FROZEN_CONFIG["candidate_pool"] == "frozen reusable ranking top-500"
    assert FROZEN_CONFIG["epsilon_rel"] == EPSILON_REL
    assert FROZEN_CONFIG["max_swaps"] == MAX_SWAPS
    assert FROZEN_CONFIG["graph_scales"] == K_SET == GRAPH_SCALES
    assert FROZEN_CONFIG["reference_scale"] == REFERENCE_K == REFERENCE_SCALE
    assert FROZEN_CONFIG["alpha"] == ALPHA == 0.2
    assert FROZEN_CONFIG["tau"] == TAU == 0.5
    assert FROZEN_CONFIG["lambda_div"] == LAMBDA_DIV == 1.2
    return dict(FROZEN_CONFIG)


def select_from_frozen_ranking(frontend, m: int):
    """Select features from one frozen reusable-ranking frontend.

    The guard ensures that the paper's relative structural-consistency constraint
    is well-defined; unsupported values fail instead of silently changing it.
    """
    assert_frozen_config()
    ranking = np.asarray(frontend.reusable_ranking, dtype=np.int64)
    pool = ranking[: min(CANDIDATE_POOL_SIZE, ranking.size)]
    if not 1 <= m <= pool.size:
        raise ValueError(f"m={m} is not executable for pool size {pool.size}")

    from methods.gcrss import _corr_abs_local
    from methods.gcrss_stage2 import structural_profiles, trust_swap

    ap = frontend.A[pool]
    cp = _corr_abs_local(frontend.X_l2[:, pool])
    initial = np.arange(m, dtype=np.int64)
    base = structural_profiles(ap, initial)
    if not np.isfinite(base) or base <= 0:
        raise ValueError("relative constraint requires finite positive g(S0)")
    epsilon = EPSILON_REL * base
    selected, swaps, _, limit = trust_swap(
        ap, cp, initial, epsilon, max_iter=MAX_SWAPS, feature_ids=pool
    )
    return pool[selected], {
        "pool": pool,
        "initial": pool[initial],
        "epsilon": epsilon,
        "base_structural": base,
        "limit": limit,
        "accepted_swaps": swaps,
    }


def select_gcrss(X_fs_train, m: int):
    """Canonical label-free selector used by smoke and formal runners."""
    from methods.gcrss import build_ranking_frontend

    x_train = np.asarray(X_fs_train, dtype=float)
    feature_ids = np.arange(x_train.shape[1], dtype=np.int64)
    scale = x_train.std(axis=0)
    keep = scale > 0
    x_train = x_train[:, keep]
    mean = x_train.mean(axis=0)
    scale = scale[keep]
    x_train = (x_train - mean) / scale
    selected, diagnostics = select_from_frozen_ranking(
        build_ranking_frontend(x_train), m
    )
    diagnostics["pool"] = feature_ids[diagnostics["pool"]]
    diagnostics["initial"] = feature_ids[diagnostics["initial"]]
    return feature_ids[selected], diagnostics
