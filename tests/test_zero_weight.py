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

# R interface for comparison tests
try:
    import rpy2.robjects as rpy
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    from rpy2.robjects import default_converter
    has_rpy2 = True
    np_cv_rules = default_converter + numpy2ri.converter
    survivalR = importr('survival')
except ImportError:
    has_rpy2 = False


def compute_sat_loglik_R(event, status, weight, tie_breaking='breslow'):
    """
    Compute saturated log-likelihood using R.

    For Breslow (default):
        LL_sat = sum over unique event times: -W_C * log(W_C)

    For Efron:
        LL_sat = sum over unique event times:
            -W_C * [log(W_C) + (1/K_C+) * (log(K_C+!) - K_C+ * log(K_C+))]

    where:
        W_C = sum of weights for events at time t
        K_C+ = count of events with positive weight at time t
    """
    with np_cv_rules.context():
        rpy.r.assign('event', np.asarray(event))
        rpy.r.assign('status', np.asarray(status).astype(int))
        rpy.r.assign('weight', np.asarray(weight))
        rpy.r.assign('tie_breaking', tie_breaking)

        rpy.r('''
        compute_sat_loglik <- function(event, status, weight, tie_breaking) {
            event_times <- event[status == 1]
            event_weights <- weight[status == 1]
            unique_times <- unique(event_times)
            loglik_sat <- 0
            for (t in unique_times) {
                w_c <- sum(event_weights[event_times == t])
                if (w_c > 0) {
                    # Breslow term
                    loglik_sat <- loglik_sat - w_c * log(w_c)

                    # Efron penalty term
                    if (tie_breaking == "efron") {
                        k_c_plus <- sum(event_weights[event_times == t] > 0)
                        if (k_c_plus > 0) {
                            efron_penalty <- (w_c / k_c_plus) * (lgamma(k_c_plus + 1) - k_c_plus * log(k_c_plus))
                            loglik_sat <- loglik_sat - efron_penalty
                        }
                    }
                }
            }
            return(loglik_sat)
        }
        ''')

        loglik_sat_R = rpy.r('compute_sat_loglik(event, status, weight, tie_breaking)')[0]

    return loglik_sat_R


def get_coxph_result(event, status, X, beta, weight, start=None, ties='efron'):
    """
    Get Cox model results from R's coxph for comparison.

    Returns deviance (-2*loglik), gradient, and log-likelihood.
    """
    with np_cv_rules.context():
        rpy.r.assign('event', np.asarray(event))
        rpy.r.assign('status', np.asarray(status).astype(int))
        rpy.r.assign('X', X)
        rpy.r.assign('beta', beta)
        rpy.r.assign('sample_weight', np.asarray(weight))
        rpy.r.assign('ties', ties)
        rpy.r('sample_weight = as.numeric(sample_weight)')

        if start is not None:
            rpy.r.assign('start', np.asarray(start))
            rpy.r('y <- Surv(start, event, status)')
        else:
            rpy.r('y <- Surv(event, status)')

        rpy.r('fit <- coxph(y ~ X, init=beta, weights=sample_weight, '
              'control=coxph.control(iter.max=0), ties=ties, robust=FALSE)')

        loglik = rpy.r('fit$loglik')
        rpy.r('score <- colSums(coxph.detail(fit)$scor)')
        gradient = rpy.r('score')

    return -2 * gradient, -2 * loglik[1], loglik[1]


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

    # Create CoxDeviance with full data (weights at initialization)
    coxdev_full = CoxDeviance(
        event=event,
        status=status,
        start=start,
        sample_weight=weights,
        tie_breaking=tie_breaking
    )

    # Create CoxDeviance with subset (non-zero weights only)
    coxdev_subset = CoxDeviance(
        event=event[nonzero_idx],
        status=status[nonzero_idx],
        start=start[nonzero_idx] if start is not None else None,
        sample_weight=weights[nonzero_idx],
        tie_breaking=tie_breaking
    )

    # Compute results
    result_full = coxdev_full(eta)
    result_subset = coxdev_subset(eta[nonzero_idx])

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
    info_full = coxdev_full.information(eta)
    info_subset = coxdev_subset.information(eta[nonzero_idx])

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

    # Full model (weights at initialization)
    stratdev_full = StratifiedCoxDeviance(
        event=event,
        status=status,
        strata=strata,
        start=start,
        sample_weight=weights,
        tie_breaking=tie_breaking
    )

    # Subset model
    stratdev_subset = StratifiedCoxDeviance(
        event=event[nonzero_idx],
        status=status[nonzero_idx],
        strata=strata[nonzero_idx],
        start=start[nonzero_idx] if start is not None else None,
        sample_weight=weights[nonzero_idx],
        tie_breaking=tie_breaking
    )

    result_full = stratdev_full(eta)
    result_subset = stratdev_subset(eta[nonzero_idx])

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


@pytest.mark.skipif(not has_rpy2, reason="rpy2 not available")
class TestZeroWeightsCompareR:
    """Tests comparing zero-weight results against R's survival package."""

    @pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
    def test_sat_loglik_zero_weights_no_ties(self, tie_breaking):
        """Compare saturated log-likelihood with R when some weights are zero (no ties)."""
        event = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        status = np.array([1, 1, 1, 1, 1])
        weight = np.array([0.0, 2.0, 0.0, 3.0, 1.5])

        cox = CoxDeviance(event=event, status=status, sample_weight=weight, tie_breaking=tie_breaking)
        result = cox(np.zeros(5))

        loglik_sat_R = compute_sat_loglik_R(event, status, weight, tie_breaking=tie_breaking)

        assert np.isclose(result.loglik_sat, loglik_sat_R, rtol=1e-10), \
            f"Python: {result.loglik_sat}, R: {loglik_sat_R}"

    @pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
    def test_sat_loglik_zero_weights_with_ties(self, tie_breaking):
        """Compare saturated log-likelihood with R when zero weights occur at tied events."""
        event = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
        status = np.array([1, 1, 1, 1, 1])
        weight = np.array([0.0, 2.0, 1.5, 0.0, 3.0])

        cox = CoxDeviance(event=event, status=status, sample_weight=weight, tie_breaking=tie_breaking)
        result = cox(np.zeros(5))

        loglik_sat_R = compute_sat_loglik_R(event, status, weight, tie_breaking=tie_breaking)

        assert np.isclose(result.loglik_sat, loglik_sat_R, rtol=1e-10), \
            f"Python: {result.loglik_sat}, R: {loglik_sat_R}"

    @pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
    def test_sat_loglik_all_zero_at_one_time(self, tie_breaking):
        """Compare with R when all events at one time have zero weight."""
        event = np.array([1.0, 1.0, 2.0, 3.0])
        status = np.array([1, 1, 1, 1])
        weight = np.array([0.0, 0.0, 2.0, 3.0])

        cox = CoxDeviance(event=event, status=status, sample_weight=weight, tie_breaking=tie_breaking)
        result = cox(np.zeros(4))

        loglik_sat_R = compute_sat_loglik_R(event, status, weight, tie_breaking=tie_breaking)

        assert np.isclose(result.loglik_sat, loglik_sat_R, rtol=1e-10), \
            f"Python: {result.loglik_sat}, R: {loglik_sat_R}"

    @pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
    @pytest.mark.parametrize('have_start_times', [True, False])
    def test_sat_loglik_zero_weights_simulated(self, tie_breaking, have_start_times):
        """Compare saturated log-likelihood with R on simulated data with zero weights."""
        rng = np.random.default_rng(789)
        data = generate_data_with_ties(n_zero=20, have_start_times=have_start_times, rng=rng)

        cox = CoxDeviance(
            event=data['event'],
            status=data['status'],
            start=data['start'],
            sample_weight=data['weights'],
            tie_breaking=tie_breaking
        )
        result = cox(data['eta'])

        loglik_sat_R = compute_sat_loglik_R(data['event'], data['status'], data['weights'],
                                            tie_breaking=tie_breaking)

        assert np.isclose(result.loglik_sat, loglik_sat_R, rtol=1e-10), \
            f"Python: {result.loglik_sat}, R: {loglik_sat_R}"

    @pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
    @pytest.mark.parametrize('have_start_times', [True, False])
    def test_deviance_consistency_zero_weights(self, tie_breaking, have_start_times):
        """
        Test deviance consistency with R's coxph when zero weights present.

        Verifies: deviance - 2*loglik_sat = -2*loglik_R
        """
        rng = np.random.default_rng(101)
        data = generate_data_with_ties(n_zero=15, have_start_times=have_start_times, rng=rng)

        # Only use non-zero weight subset for R comparison (R's coxph with zero weights)
        nonzero_idx = data['nonzero_idx']
        event_sub = data['event'][nonzero_idx]
        status_sub = data['status'][nonzero_idx]
        weights_sub = data['weights'][nonzero_idx]
        start_sub = data['start'][nonzero_idx] if data['start'] is not None else None

        n_sub = len(event_sub)
        p = max(2, n_sub // 10)
        X = rng.standard_normal((n_sub, p))
        beta = rng.standard_normal(p) / np.sqrt(n_sub)
        eta_sub = X @ beta

        cox = CoxDeviance(
            event=event_sub,
            status=status_sub,
            start=start_sub,
            sample_weight=weights_sub,
            tie_breaking=tie_breaking
        )
        result = cox(eta_sub)

        _, _, loglik_R = get_coxph_result(
            event_sub, status_sub, X, beta, weights_sub,
            start=start_sub, ties=tie_breaking
        )

        # deviance - 2*loglik_sat should equal -2*loglik_R
        lhs = result.deviance - 2 * result.loglik_sat
        rhs = -2 * loglik_R

        assert np.isclose(lhs, rhs, rtol=1e-9), \
            f"deviance - 2*loglik_sat = {lhs}, -2*loglik_R = {rhs}"


def test_zero_weight_at_first_event_ordered_position():
    """
    Regression test for bug where first observation in event order has zero weight.

    This tests that the stratified C++ implementation correctly handles the case
    where effective_cluster_sizes[0] = 0 because the first event-ordered observation
    has zero weight. Previously, the code checked effective_cluster_sizes[0] > 0
    to decide whether to use zero-weight handling, which failed in this case.

    The fix was to use an explicit flag (use_zero_weight_handling) instead of
    checking the first element of effective_cluster_sizes.
    """
    from coxdev.stratified_cpp import StratifiedCoxDevianceCpp

    rng = np.random.default_rng(42)

    # Create data where first observation in event order will have zero weight
    # Use ties to ensure Efron correction is needed
    n = 20
    event = np.array([1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype=float)
    start = event - rng.uniform(0.1, 0.5, n)
    status = np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=int)

    # Use the unstratified CoxDeviance to get the event order (via preprocessing)
    cox_temp = CoxDeviance(event=event, start=start, status=status, tie_breaking='efron')
    event_order = cox_temp._event_order

    # Create weights where the first observation in event order has zero weight
    weights = np.ones(n)
    weights[event_order[0]] = 0.0

    # Verify our setup: first event-ordered observation should have zero weight
    w_event = weights[event_order]
    assert w_event[0] == 0.0, "First event-ordered observation should have zero weight"

    eta = rng.standard_normal(n) * 0.5

    # Test with stratified implementation (single stratum)
    strata = np.ones(n, dtype=np.int32)
    cox_strat = StratifiedCoxDevianceCpp(
        event=event, start=start, status=status,
        strata=strata, sample_weight=weights, tie_breaking='efron'
    )
    result_strat = cox_strat(eta)

    # Test with unstratified implementation
    cox_unstrat = CoxDeviance(event=event, start=start, status=status,
                               sample_weight=weights, tie_breaking='efron')
    result_unstrat = cox_unstrat(eta)

    # The key test: stratified should match unstratified
    assert np.isclose(result_strat.deviance, result_unstrat.deviance, rtol=1e-10), \
        f"Deviance mismatch: stratified={result_strat.deviance}, unstratified={result_unstrat.deviance}"

    assert np.isclose(result_strat.loglik_sat, result_unstrat.loglik_sat, rtol=1e-10), \
        "Loglik_sat mismatch"

    assert np.allclose(result_strat.gradient, result_unstrat.gradient, rtol=1e-10), \
        "Gradient mismatch"

    # Also verify against subset (non-zero weights only)
    nonzero_idx = weights > 0
    cox_subset = CoxDeviance(
        event=event[nonzero_idx],
        start=start[nonzero_idx],
        status=status[nonzero_idx],
        sample_weight=weights[nonzero_idx],
        tie_breaking='efron'
    )
    result_subset = cox_subset(eta[nonzero_idx])

    assert np.isclose(result_strat.deviance, result_subset.deviance, rtol=1e-10), \
        f"Deviance mismatch with subset: stratified={result_strat.deviance}, subset={result_subset.deviance}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
