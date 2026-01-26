"""
Comprehensive comparison of coxdev vs adelie for Cox proportional hazards.

Tests both Breslow and Efron tie-breaking methods across various data scenarios.

Key relationships between coxdev and adelie:
- adelie normalizes weights to sum to 1 and uses w̄_i (average weight within ties)
- coxdev uses raw weights directly

The exact mathematical relationships are:
1. Gradient: grad_coxdev = -2 * sum(weights) * grad_adelie
2. Hessian: hess_coxdev = 2 * sum(weights) * hess_adelie
3. Deviance: Different conventions due to w̄_i term in adelie
   - With uniform weights: deviance_coxdev = -2 * (weight_sum * (-loss_adelie) - log(weight_sum) * weighted_events)
   - With non-uniform weights: more complex relationship due to w̄_i adjustment

For optimization purposes, the gradient and Hessian relationships are what matter.
"""

import numpy as np
import pytest

# Import coxdev
from coxdev import CoxDeviance, StratifiedCoxDeviance

# Import adelie (requires pip install adelie)
try:
    import adelie.glm as adelie_glm
    HAS_ADELIE = True
except ImportError:
    HAS_ADELIE = False

# Import simulation utilities
from simulate import simulate_df, rng, sample_weights

# Skip all tests if adelie not available
pytestmark = pytest.mark.skipif(not HAS_ADELIE, reason="adelie not available")


def generate_test_data(n, seed=42, with_ties=True, tie_fraction=0.3):
    """
    Generate synthetic survival data for testing.

    Parameters
    ----------
    n : int
        Number of observations
    seed : int
        Random seed
    with_ties : bool
        Whether to create tied event times
    tie_fraction : float
        Fraction of observations with tied times

    Returns
    -------
    dict with keys: start, stop, status, eta, weights
    """
    np.random.seed(seed)

    # Generate start times (left truncation)
    start = np.random.exponential(1, n)

    # Generate stop times
    if with_ties:
        # Create some tied event times
        n_unique = max(2, int(n * (1 - tie_fraction)))
        unique_times = np.sort(np.random.exponential(2, n_unique)) + start.min()
        stop = np.random.choice(unique_times, n) + start
    else:
        stop = start + np.random.exponential(2, n)

    # Ensure stop > start
    stop = np.maximum(stop, start + 0.01)

    # Generate status (event indicator)
    status = np.random.binomial(1, 0.6, n)

    # Generate linear predictor
    eta = np.random.normal(0, 0.5, n)

    # Generate weights
    weights = np.random.exponential(1, n)
    # Add some zero weights
    zero_idx = np.random.choice(n, size=max(1, n // 10), replace=False)
    weights[zero_idx] = 0

    return {
        'start': start.astype(np.float64),
        'stop': stop.astype(np.float64),
        'status': status.astype(np.float64),
        'eta': eta.astype(np.float64),
        'weights': weights.astype(np.float64),
    }


def generate_stratified_data(n, n_strata=3, seed=42):
    """Generate stratified survival data."""
    data = generate_test_data(n, seed=seed)
    np.random.seed(seed + 100)
    data['strata'] = np.random.randint(0, n_strata, n)
    return data


def compare_coxdev_adelie(data, tie_method='efron', rtol=1e-10, atol=1e-12):
    """
    Compare coxdev and adelie outputs.

    Returns dict with comparison results.
    """
    start = data['start']
    stop = data['stop']
    status = data['status']
    eta = data['eta']
    weights = data['weights']
    strata = data.get('strata', None)

    n = len(status)
    weight_sum = np.sum(weights)
    weighted_events = np.sum(weights * status)

    # =========================================================================
    # coxdev computation (sample_weight at initialization)
    # =========================================================================
    if strata is not None:
        cox_coxdev = StratifiedCoxDeviance(
            event=stop,
            status=status,
            strata=strata,
            start=start,
            sample_weight=weights,
            tie_breaking=tie_method
        )
    else:
        cox_coxdev = CoxDeviance(
            event=stop,
            status=status,
            start=start,
            sample_weight=weights,
            tie_breaking=tie_method
        )

    result_coxdev = cox_coxdev(eta)

    deviance_coxdev = result_coxdev.deviance
    grad_coxdev = result_coxdev.gradient
    diag_hess_coxdev = result_coxdev.diag_hessian
    loglik_sat_coxdev = result_coxdev.loglik_sat

    # =========================================================================
    # adelie computation
    # =========================================================================
    if strata is not None:
        cox_adelie = adelie_glm.cox(
            start=start,
            stop=stop,
            status=status,
            strata=strata,
            weights=weights,
            tie_method=tie_method,
            dtype=np.float64
        )
    else:
        cox_adelie = adelie_glm.cox(
            start=start,
            stop=stop,
            status=status,
            weights=weights,
            tie_method=tie_method,
            dtype=np.float64
        )

    loss_adelie = cox_adelie.loss(eta)
    loss_full_adelie = cox_adelie.loss_full()

    grad_adelie = np.empty(n, dtype=np.float64)
    cox_adelie.gradient(eta, grad_adelie)

    hess_adelie = np.empty(n, dtype=np.float64)
    cox_adelie.hessian(eta, grad_adelie, hess_adelie)

    # =========================================================================
    # Convert between conventions
    # =========================================================================
    # Key relationships:
    # 1. grad_coxdev = -2 * weight_sum * grad_adelie
    # 2. hess_coxdev = 2 * weight_sum * hess_adelie
    # 3. For deviance with uniform weights only:
    #    loglik_coxdev = -weight_sum * loss_adelie - log(weight_sum) * weighted_events
    #    deviance_coxdev = -2 * loglik_coxdev

    grad_from_adelie = -2 * weight_sum * grad_adelie
    diag_hess_from_adelie = 2 * weight_sum * hess_adelie

    # For saturated log-likelihood (this should match regardless of weights)
    if weight_sum > 0:
        loglik_sat_from_adelie = -weight_sum * loss_full_adelie - np.log(weight_sum) * weighted_events
    else:
        loglik_sat_from_adelie = 0

    # =========================================================================
    # Compare
    # =========================================================================
    results = {
        'tie_method': tie_method,
        'n': n,
        'weight_sum': weight_sum,
        'weighted_events': weighted_events,

        # Raw values
        'deviance_coxdev': deviance_coxdev,
        'loss_adelie': loss_adelie,

        'loglik_sat_coxdev': loglik_sat_coxdev,
        'loss_full_adelie': loss_full_adelie,
        'loglik_sat_from_adelie': loglik_sat_from_adelie,

        # Comparison metrics
        'loglik_sat_close': np.allclose(loglik_sat_coxdev, loglik_sat_from_adelie, rtol=rtol, atol=atol),
        'loglik_sat_diff': abs(loglik_sat_coxdev - loglik_sat_from_adelie),

        'gradient_close': np.allclose(grad_coxdev, grad_from_adelie, rtol=rtol, atol=atol),
        'gradient_max_diff': np.max(np.abs(grad_coxdev - grad_from_adelie)),
        'gradient_coxdev': grad_coxdev,
        'gradient_adelie_scaled': grad_from_adelie,

        'hessian_close': np.allclose(diag_hess_coxdev, diag_hess_from_adelie, rtol=rtol, atol=atol),
        'hessian_max_diff': np.max(np.abs(diag_hess_coxdev - diag_hess_from_adelie)),
        'hessian_coxdev': diag_hess_coxdev,
        'hessian_adelie_scaled': diag_hess_from_adelie,
    }

    return results


class TestGradientHessian:
    """Test gradient and Hessian comparison - these should always match."""

    @pytest.mark.parametrize("n", [10, 50, 100, 200])
    @pytest.mark.parametrize("tie_method", ["efron", "breslow"])
    def test_unstratified_basic(self, n, tie_method):
        """Basic comparison for unstratified Cox model."""
        data = generate_test_data(n, seed=42)
        results = compare_coxdev_adelie(data, tie_method=tie_method)

        assert results['gradient_close'], \
            f"Gradient mismatch: max_diff={results['gradient_max_diff']:.2e}"

        assert results['hessian_close'], \
            f"Hessian mismatch: max_diff={results['hessian_max_diff']:.2e}"

    @pytest.mark.parametrize("n", [30, 100])
    @pytest.mark.parametrize("n_strata", [2, 3, 5])
    @pytest.mark.parametrize("tie_method", ["efron", "breslow"])
    def test_stratified(self, n, n_strata, tie_method):
        """Comparison for stratified Cox model."""
        data = generate_stratified_data(n, n_strata=n_strata, seed=42)
        results = compare_coxdev_adelie(data, tie_method=tie_method)

        assert results['gradient_close'], \
            f"Gradient mismatch: max_diff={results['gradient_max_diff']:.2e}"

        assert results['hessian_close'], \
            f"Hessian mismatch: max_diff={results['hessian_max_diff']:.2e}"

    @pytest.mark.parametrize("tie_method", ["efron", "breslow"])
    def test_no_ties(self, tie_method):
        """Test with no tied event times."""
        data = generate_test_data(50, seed=42, with_ties=False)
        results = compare_coxdev_adelie(data, tie_method=tie_method)

        assert results['gradient_close']
        assert results['hessian_close']

    @pytest.mark.parametrize("tie_method", ["efron", "breslow"])
    def test_many_ties(self, tie_method):
        """Test with many tied event times."""
        data = generate_test_data(100, seed=42, with_ties=True, tie_fraction=0.8)
        results = compare_coxdev_adelie(data, tie_method=tie_method)

        assert results['gradient_close']
        assert results['hessian_close']

    @pytest.mark.parametrize("tie_method", ["efron", "breslow"])
    def test_uniform_weights(self, tie_method):
        """Test with uniform (equal) weights."""
        data = generate_test_data(50, seed=42)
        data['weights'] = np.ones(len(data['status']))
        results = compare_coxdev_adelie(data, tie_method=tie_method)

        assert results['gradient_close']
        assert results['hessian_close']

    @pytest.mark.parametrize("tie_method", ["efron", "breslow"])
    def test_many_zero_weights(self, tie_method):
        """Test with many zero weights."""
        data = generate_test_data(100, seed=42)
        # Set 40% weights to zero
        n = len(data['weights'])
        np.random.seed(123)
        zero_idx = np.random.choice(n, size=int(0.4 * n), replace=False)
        data['weights'][zero_idx] = 0

        results = compare_coxdev_adelie(data, tie_method=tie_method)

        assert results['gradient_close']
        assert results['hessian_close']

    @pytest.mark.parametrize("tie_method", ["efron", "breslow"])
    def test_no_left_truncation(self, tie_method):
        """Test without left truncation (start=0)."""
        data = generate_test_data(50, seed=42)
        data['start'] = np.zeros_like(data['start'])

        results = compare_coxdev_adelie(data, tie_method=tie_method)

        assert results['gradient_close']
        assert results['hessian_close']

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("tie_method", ["efron", "breslow"])
    def test_multiple_seeds(self, seed, tie_method):
        """Test with multiple random seeds for robustness."""
        data = generate_test_data(50, seed=seed)
        results = compare_coxdev_adelie(data, tie_method=tie_method)

        assert results['gradient_close']
        assert results['hessian_close']


class TestSaturatedLoglikelihood:
    """Test saturated log-likelihood comparison."""

    @pytest.mark.parametrize("tie_method", ["efron", "breslow"])
    def test_saturated_loglik(self, tie_method):
        """Compare saturated log-likelihood between coxdev and adelie."""
        data = generate_test_data(100, seed=42)
        results = compare_coxdev_adelie(data, tie_method=tie_method)

        assert results['loglik_sat_close'], \
            f"Saturated loglik mismatch: coxdev={results['loglik_sat_coxdev']:.10f}, " \
            f"adelie_converted={results['loglik_sat_from_adelie']:.10f}, " \
            f"diff={results['loglik_sat_diff']:.2e}"

    @pytest.mark.parametrize("tie_method", ["efron", "breslow"])
    def test_saturated_loglik_with_many_ties(self, tie_method):
        """Test saturated log-likelihood with many tied event times.

        This test explicitly creates data with many ties to verify that
        the Efron saturated log-likelihood penalty term is computed correctly.
        """
        np.random.seed(42)
        n = 30
        # Create many ties: 5 unique times with 6 observations each
        stop = np.repeat([1.0, 2.0, 3.0, 4.0, 5.0], 6)
        start = np.zeros(n)
        status = np.random.binomial(1, 0.8, n).astype(float)
        eta = np.random.randn(n) * 0.3
        weights = np.random.uniform(0.5, 2.0, n)

        data = {
            'start': start,
            'stop': stop,
            'status': status,
            'eta': eta,
            'weights': weights,
        }

        results = compare_coxdev_adelie(data, tie_method=tie_method)

        assert results['loglik_sat_close'], \
            f"Saturated loglik mismatch with ties ({tie_method}): " \
            f"coxdev={results['loglik_sat_coxdev']:.10f}, " \
            f"adelie_converted={results['loglik_sat_from_adelie']:.10f}, " \
            f"diff={results['loglik_sat_diff']:.2e}"

    @pytest.mark.parametrize("tie_method", ["efron", "breslow"])
    def test_saturated_loglik_all_tied(self, tie_method):
        """Test saturated log-likelihood when all events are at the same time."""
        np.random.seed(123)
        n = 20
        # All events at the same time
        stop = np.ones(n) * 5.0
        start = np.zeros(n)
        status = np.ones(n)  # All events
        eta = np.random.randn(n) * 0.2
        weights = np.ones(n)

        data = {
            'start': start,
            'stop': stop,
            'status': status,
            'eta': eta,
            'weights': weights,
        }

        results = compare_coxdev_adelie(data, tie_method=tie_method)

        assert results['loglik_sat_close'], \
            f"Saturated loglik mismatch all tied ({tie_method}): " \
            f"coxdev={results['loglik_sat_coxdev']:.10f}, " \
            f"adelie_converted={results['loglik_sat_from_adelie']:.10f}, " \
            f"diff={results['loglik_sat_diff']:.2e}"

    def test_efron_differs_from_breslow_with_ties(self):
        """Verify that Efron and Breslow saturated log-likelihoods differ with ties."""
        np.random.seed(42)
        n = 10
        # Every pair tied
        stop = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])
        start = np.zeros(n)
        status = np.ones(n)  # All events
        eta = np.zeros(n)
        weights = np.ones(n)

        data = {
            'start': start,
            'stop': stop,
            'status': status,
            'eta': eta,
            'weights': weights,
        }

        results_efron = compare_coxdev_adelie(data, tie_method='efron')
        results_breslow = compare_coxdev_adelie(data, tie_method='breslow')

        # Both should match adelie
        assert results_efron['loglik_sat_close']
        assert results_breslow['loglik_sat_close']

        # Efron and Breslow should differ when there are ties
        assert results_efron['loglik_sat_coxdev'] != results_breslow['loglik_sat_coxdev'], \
            "Efron and Breslow saturated log-likelihoods should differ with ties"


class TestEdgeCases:
    """Test edge cases for gradient and Hessian."""

    @pytest.mark.parametrize("tie_method", ["efron", "breslow"])
    def test_all_events(self, tie_method):
        """Test when all observations are events (no censoring)."""
        data = generate_test_data(50, seed=42)
        data['status'] = np.ones_like(data['status'])

        results = compare_coxdev_adelie(data, tie_method=tie_method)

        assert results['gradient_close']
        assert results['hessian_close']

    @pytest.mark.parametrize("tie_method", ["efron", "breslow"])
    def test_few_events(self, tie_method):
        """Test with only a few events."""
        data = generate_test_data(50, seed=42)
        # Only 5 events
        data['status'] = np.zeros_like(data['status'])
        np.random.seed(456)
        event_idx = np.random.choice(50, size=5, replace=False)
        data['status'][event_idx] = 1

        results = compare_coxdev_adelie(data, tie_method=tie_method)

        assert results['gradient_close']
        assert results['hessian_close']

    @pytest.mark.parametrize("tie_method", ["efron", "breslow"])
    def test_small_sample(self, tie_method):
        """Test with very small sample size."""
        data = generate_test_data(5, seed=42)
        data['weights'] = np.ones(5)  # No zero weights for small sample

        results = compare_coxdev_adelie(data, tie_method=tie_method)

        assert results['gradient_close']
        assert results['hessian_close']


class TestTieBreaking:
    """Test that tie-breaking methods produce different results when there are ties."""

    def test_breslow_vs_efron_gradient_differs_with_ties(self):
        """Verify that Breslow and Efron give different gradients when ties exist."""
        # Use data with explicit ties
        np.random.seed(42)
        n = 50
        start = np.zeros(n)
        # Create ties: use only 10 unique stop times
        stop = np.random.choice(np.arange(1, 11), n).astype(float)
        status = np.ones(n)  # All events to ensure ties matter
        eta = np.random.normal(0, 0.5, n)
        weights = np.ones(n)

        data = {
            'start': start,
            'stop': stop,
            'status': status,
            'eta': eta,
            'weights': weights,
        }

        results_breslow = compare_coxdev_adelie(data, tie_method='breslow')
        results_efron = compare_coxdev_adelie(data, tie_method='efron')

        # Gradients should be different for the two methods when ties exist
        assert not np.allclose(
            results_breslow['gradient_coxdev'],
            results_efron['gradient_coxdev']
        ), "Breslow and Efron should give different gradients when ties exist"

        # But each method should match between coxdev and adelie
        assert results_breslow['gradient_close']
        assert results_efron['gradient_close']

    def test_breslow_equals_efron_without_ties(self):
        """Verify that Breslow and Efron give same results when no ties exist."""
        # Use data with unique stop times
        np.random.seed(42)
        n = 50
        start = np.zeros(n)
        stop = np.arange(1, n + 1).astype(float)  # Unique times
        status = np.random.binomial(1, 0.6, n).astype(float)
        eta = np.random.normal(0, 0.5, n)
        weights = np.ones(n)

        data = {
            'start': start,
            'stop': stop,
            'status': status,
            'eta': eta,
            'weights': weights,
        }

        results_breslow = compare_coxdev_adelie(data, tie_method='breslow')
        results_efron = compare_coxdev_adelie(data, tie_method='efron')

        # Gradients should be the same when no ties exist
        assert np.allclose(
            results_breslow['gradient_coxdev'],
            results_efron['gradient_coxdev']
        ), "Breslow and Efron should give same gradients when no ties exist"


def print_detailed_comparison(data, tie_method='efron'):
    """
    Print detailed comparison results for debugging.

    Usage:
        data = generate_test_data(50, seed=42)
        print_detailed_comparison(data, tie_method='efron')
    """
    results = compare_coxdev_adelie(data, tie_method=tie_method)

    print(f"\n{'='*60}")
    print(f"Comparison: coxdev vs adelie ({tie_method})")
    print(f"{'='*60}")
    print(f"Sample size: {results['n']}")
    print(f"Weight sum: {results['weight_sum']:.6f}")
    print(f"Weighted events: {results['weighted_events']:.6f}")
    print()

    print("DEVIANCE (different conventions, not directly comparable):")
    print(f"  coxdev deviance:   {results['deviance_coxdev']:.10f}")
    print(f"  adelie loss:       {results['loss_adelie']:.10f}")
    print()

    print("SATURATED LOGLIK:")
    print(f"  coxdev:            {results['loglik_sat_coxdev']:.10f}")
    print(f"  adelie (converted): {results['loglik_sat_from_adelie']:.10f}")
    print(f"  Difference:        {results['loglik_sat_diff']:.2e}")
    print(f"  Match: {'✓' if results['loglik_sat_close'] else '✗'}")
    print()

    print("GRADIENT:")
    print(f"  Max difference: {results['gradient_max_diff']:.2e}")
    print(f"  Match: {'✓' if results['gradient_close'] else '✗'}")
    print()

    print("DIAGONAL HESSIAN:")
    print(f"  Max difference: {results['hessian_max_diff']:.2e}")
    print(f"  Match: {'✓' if results['hessian_close'] else '✗'}")
    print()


if __name__ == '__main__':
    # Run a quick comparison for debugging
    print("Running coxdev vs adelie comparison...")
    print("\n" + "="*60)
    print("KEY FINDINGS:")
    print("="*60)
    print("1. Gradient: grad_coxdev = -2 * sum(weights) * grad_adelie")
    print("2. Hessian:  hess_coxdev = 2 * sum(weights) * hess_adelie")
    print("3. Deviance: Different conventions (see docstring)")
    print("="*60)

    for tie_method in ['efron', 'breslow']:
        data = generate_test_data(100, seed=42)
        print_detailed_comparison(data, tie_method=tie_method)

    print("\nRunning pytest...")
    pytest.main([__file__, '-v', '--tb=short'])
