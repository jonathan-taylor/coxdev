"""
Tests to verify that zero-weight observations don't affect Cox model results.

When some observations have zero weights, the deviance, gradient, and
information matrix should match the results computed using only the
non-zero weight observations.

Implementation notes:
    Both Efron and Breslow tie-breaking methods correctly handle zero
    weights. The implementation uses three key corrections:

    1. Weighted eta centering: The linear predictor eta is centered using
       a weighted mean (np.average with weights=sample_weight), so zero-weight
       observations don't affect the centering.

    2. Corrected scaling (Efron only): The scaling factor for each observation
       in a tie group is computed as effective_rank / effective_cluster_size,
       where both numerator and denominator only count non-zero weight
       observations.

    3. Corrected w_avg denominator (Efron only): The average weight w_avg
       for a tie group is computed as:
           w_avg = sum(weights in tie group) / effective_cluster_size
       where effective_cluster_size is the count of non-zero weight observations,
       not the total count.
"""

import numpy as np
import pytest
from coxdev import CoxDeviance, StratifiedCoxDeviance
from simulate import simulate_df, all_combos


def generate_data_with_ties(n_zero, have_start_times, rng, tie_types=None):
    """
    Generate survival data with ties in both start and event times.

    Uses the simulate_df function which generates data with various tie patterns
    that are known to work with the library.

    Parameters
    ----------
    n_zero : int
        Number of observations to assign zero weight
    have_start_times : bool
        Whether to include start times (left-truncated data)
    rng : np.random.Generator
        Random number generator
    tie_types : tuple, optional
        Tie types to use from simulate.py. Default uses a mix.

    Returns
    -------
    dict
        Dictionary with event, status, start, weights, eta, and index arrays
    """
    # Use simulate_df to generate data with ties
    # Use a subset of tie_types that create various tie patterns
    if tie_types is None:
        tie_types = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (2, 2))

    df = simulate_df(tie_types, nrep=3, size=5, rng=rng, noinfo=True)

    event = df['event'].values
    status = df['status'].values
    n = len(event)

    if have_start_times:
        start = df['start'].values
    else:
        start = None

    # Adjust n_zero if larger than n
    n_zero = min(n_zero, n // 3)

    # Generate weights with some zeros
    weights = rng.uniform(0.5, 2.0, size=n)
    zero_idx = rng.choice(n, size=n_zero, replace=False)
    weights[zero_idx] = 0.0

    nonzero_idx = weights > 0

    # Linear predictor
    eta = rng.standard_normal(n) * 0.5

    return {
        'event': event,
        'status': status,
        'start': start,
        'weights': weights,
        'eta': eta,
        'nonzero_idx': nonzero_idx,
        'n': n,
    }


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('have_start_times', [True, False])
def test_zero_weights_match_subset(tie_breaking, have_start_times, request):
    """
    Test that zero-weight observations don't affect CoxDeviance results.

    Verifies that results with zero-weight observations match results
    computed using only the non-zero weight subset.
    """
    rng = np.random.default_rng(123)
    n_zero = 30

    data = generate_data_with_ties(n_zero, have_start_times, rng)

    event = data['event']
    status = data['status']
    start = data['start']
    weights = data['weights']
    eta = data['eta']
    nonzero_idx = data['nonzero_idx']

    # Create CoxDeviance with full data
    coxdev_full = CoxDeviance(
        event=event,
        status=status,
        start=start,
        tie_breaking=tie_breaking
    )

    # Create CoxDeviance with subset (non-zero weights only)
    coxdev_subset = CoxDeviance(
        event=event[nonzero_idx],
        status=status[nonzero_idx],
        start=start[nonzero_idx] if start is not None else None,
        tie_breaking=tie_breaking
    )

    # Compute results
    result_full = coxdev_full(eta, weights)
    result_subset = coxdev_subset(eta[nonzero_idx], weights[nonzero_idx])

    # Compare deviance
    assert np.allclose(result_full.deviance, result_subset.deviance, rtol=1e-10), \
        f"Deviance mismatch: {result_full.deviance} vs {result_subset.deviance}"

    # Compare saturated log-likelihood
    assert np.allclose(result_full.loglik_sat, result_subset.loglik_sat, rtol=1e-10), \
        f"Loglik_sat mismatch: {result_full.loglik_sat} vs {result_subset.loglik_sat}"

    # Compare gradient for non-zero weight observations
    assert np.allclose(result_full.gradient[nonzero_idx], result_subset.gradient, rtol=1e-10), \
        "Gradient mismatch for non-zero weight observations"

    # Gradient for zero-weight observations should be zero
    assert np.allclose(result_full.gradient[~nonzero_idx], 0.0, atol=1e-10), \
        "Gradient should be zero for zero-weight observations"

    # Compare diagonal Hessian for non-zero weight observations
    assert np.allclose(result_full.diag_hessian[nonzero_idx], result_subset.diag_hessian, rtol=1e-10), \
        "Diagonal Hessian mismatch for non-zero weight observations"

    # Compare information matrix action on a test vector
    info_full = coxdev_full.information(eta, weights)
    info_subset = coxdev_subset.information(eta[nonzero_idx], weights[nonzero_idx])

    # Test vector
    n = len(eta)
    v_full = rng.standard_normal(n)
    v_subset = v_full[nonzero_idx]

    # Information matrix-vector product
    Iv_full = info_full @ v_full
    Iv_subset = info_subset @ v_subset

    # The result for non-zero weight observations should match
    assert np.allclose(Iv_full[nonzero_idx], Iv_subset, rtol=1e-10), \
        "Information matrix-vector product mismatch"


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('have_start_times', [True, False])
def test_zero_weights_stratified(tie_breaking, have_start_times, request):
    """
    Test that zero-weight observations don't affect StratifiedCoxDeviance results.
    """
    rng = np.random.default_rng(456)
    n_strata = 3
    n_zero = 30

    data = generate_data_with_ties(n_zero, have_start_times, rng)

    event = data['event']
    status = data['status']
    start = data['start']
    weights = data['weights']
    eta = data['eta']
    nonzero_idx = data['nonzero_idx']
    n = data['n']

    # Generate strata
    strata = rng.choice(n_strata, size=n)

    # Full model
    stratdev_full = StratifiedCoxDeviance(
        event=event,
        status=status,
        strata=strata,
        start=start,
        tie_breaking=tie_breaking
    )

    # Subset model
    stratdev_subset = StratifiedCoxDeviance(
        event=event[nonzero_idx],
        status=status[nonzero_idx],
        strata=strata[nonzero_idx],
        start=start[nonzero_idx] if start is not None else None,
        tie_breaking=tie_breaking
    )

    result_full = stratdev_full(eta, weights)
    result_subset = stratdev_subset(eta[nonzero_idx], weights[nonzero_idx])

    # Compare deviance
    assert np.allclose(result_full.deviance, result_subset.deviance, rtol=1e-10), \
        f"Stratified deviance mismatch: {result_full.deviance} vs {result_subset.deviance}"

    # Compare saturated log-likelihood
    assert np.allclose(result_full.loglik_sat, result_subset.loglik_sat, rtol=1e-10), \
        f"Stratified loglik_sat mismatch: {result_full.loglik_sat} vs {result_subset.loglik_sat}"

    # Compare gradient for non-zero weight observations
    assert np.allclose(result_full.gradient[nonzero_idx], result_subset.gradient, rtol=1e-10), \
        "Stratified gradient mismatch"

    # Gradient for zero-weight observations should be zero
    assert np.allclose(result_full.gradient[~nonzero_idx], 0.0, atol=1e-10), \
        "Stratified gradient should be zero for zero-weight observations"

    # Compare diagonal Hessian for non-zero weight observations
    assert np.allclose(result_full.diag_hessian[nonzero_idx], result_subset.diag_hessian, rtol=1e-10), \
        "Stratified diagonal Hessian mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
