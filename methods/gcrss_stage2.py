"""Budget-conditioned subset realization for GCRSS.

The selector starts from the reusable-ranking top-m prefix. It then evaluates exact
one-for-one exchanges inside the frozen candidate pool, accepting only the
best deterministic redundancy improvement that satisfies the relative
structural-consistency constraint. At most two swaps are accepted.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .reusable_ranking import ALPHA, K_SET, REFERENCE_K, TAU


Q0 = K_SET.index(REFERENCE_K)


def structural_profiles(ap, subset, alpha=ALPHA, tau=TAU):
    """Evaluate the structural-consistency cost g(S) of one subset."""

    p = ap[np.asarray(subset, dtype=int)].mean(0)
    robust = float(tau * (logsumexp(p / tau) - np.log(p.size)))
    return float((1 - alpha) * p[Q0] + alpha * robust)


def diversity_local(cp, subset):
    """Mean absolute pairwise correlation within a subset."""

    s = np.asarray(subset, dtype=int)
    m = len(s)
    return 0.0 if m <= 1 else float(2 * np.triu(cp[np.ix_(s, s)], 1).sum() / (m * (m - 1)))


def trust_swap(
    ap,
    cp,
    s0,
    epsilon,
    alpha=ALPHA,
    tau=TAU,
    max_iter=2,
    tol=1e-12,
    feature_ids=None,
):
    """Perform deterministic best-improvement 1-for-1 exchanges under Eq. (8).

    Parameters are label-free: ``ap`` is the candidate structural profile
    matrix, ``cp`` is the candidate correlation matrix, and ``s0`` is the
    initial top-m subset in candidate-pool coordinates.
    """

    s = np.asarray(s0, dtype=np.int64).copy()
    m = len(s)
    q = ap.shape[1]
    feature_ids = (
        np.arange(ap.shape[0], dtype=np.int64)
        if feature_ids is None
        else np.asarray(feature_ids, dtype=np.int64)
    )
    base_struct = structural_profiles(ap, s, alpha, tau)
    limit = base_struct + float(epsilon)
    accepted = 0

    for _ in range(max_iter):
        in_subset = np.zeros(ap.shape[0], dtype=bool)
        in_subset[s] = True
        outside = np.flatnonzero(~in_subset)
        if outside.size == 0:
            break

        sum_a = ap[s].sum(0)
        cs = cp[np.ix_(s, s)]
        selected_corr = cs.sum(1)
        pairs = float(np.triu(cs, 1).sum())
        cos = cp[np.ix_(outside, s)]
        outside_corr = cos.sum(1)

        profiles = (sum_a[None, None, :] - ap[s][None, :, :] + ap[outside][:, None, :]) / m
        nominal = profiles[:, :, Q0]
        robust = tau * (logsumexp(profiles / tau, axis=2) - np.log(q))
        structural = (1 - alpha) * nominal + alpha * robust

        new_pairs = pairs - selected_corr[None, :] + outside_corr[:, None] - cos
        diversity = 0 * structural if m <= 1 else 2 * new_pairs / (m * (m - 1))
        current_diversity = diversity_local(cp, s)
        feasible = (structural <= limit + tol) & (diversity < current_diversity - tol)
        if not feasible.any():
            break

        masked = np.where(feasible, diversity, np.inf)
        outside_key = np.broadcast_to(
            feature_ids[outside, None], masked.shape
        ).ravel()
        selected_key = np.broadcast_to(
            feature_ids[s][None, :], masked.shape
        ).ravel()
        best = np.lexsort((selected_key, outside_key, masked.ravel()))[0]
        outside_i, selected_i = np.unravel_index(int(best), masked.shape)
        s[selected_i] = outside[outside_i]
        accepted += 1

    return s, accepted, base_struct, limit
