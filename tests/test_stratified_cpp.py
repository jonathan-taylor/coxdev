"""
Tests for C++ Stratified Cox Proportional Hazards Model Implementation.

These tests verify that the C++ stratified implementation produces results
matching the Python stratified implementation.
"""

import numpy as np
import pytest
from coxdev import StratifiedCoxDeviance, StratifiedCoxDevianceCpp


class TestStratifiedCppVsPython:
    """Test that C++ implementation matches Python implementation."""

    def test_basic_stratified(self):
        """Test basic stratified Cox model with 3 strata."""
        np.random.seed(42)
        n = 100
        event = np.random.exponential(5, n)
        status = np.random.binomial(1, 0.7, n)
        strata = np.random.choice([0, 1, 2], n)
        eta = np.random.randn(n) * 0.5

        # Python implementation
        py_cox = StratifiedCoxDeviance(event=event, status=status, strata=strata)
        py_result = py_cox(eta)

        # C++ implementation
        cpp_cox = StratifiedCoxDevianceCpp(event=event, status=status, strata=strata)
        cpp_result = cpp_cox(eta)

        # Check deviance matches
        assert np.isclose(py_result.deviance, cpp_result.deviance, rtol=1e-10), \
            f"Deviance mismatch: Python={py_result.deviance}, C++={cpp_result.deviance}"

        # Check gradient matches
        assert np.allclose(py_result.gradient, cpp_result.gradient, rtol=1e-10), \
            f"Gradient mismatch: max diff={np.abs(py_result.gradient - cpp_result.gradient).max()}"

        # Check diagonal Hessian matches
        assert np.allclose(py_result.diag_hessian, cpp_result.diag_hessian, rtol=1e-10), \
            f"Diag Hessian mismatch: max diff={np.abs(py_result.diag_hessian - cpp_result.diag_hessian).max()}"

    def test_with_sample_weights(self):
        """Test stratified Cox with sample weights."""
        np.random.seed(123)
        n = 80
        event = np.random.exponential(5, n)
        status = np.random.binomial(1, 0.6, n)
        strata = np.random.choice([0, 1], n)
        eta = np.random.randn(n) * 0.3
        weights = np.random.uniform(0.5, 2.0, n)

        py_cox = StratifiedCoxDeviance(event=event, status=status, strata=strata)
        py_result = py_cox(eta, weights)

        cpp_cox = StratifiedCoxDevianceCpp(event=event, status=status, strata=strata)
        cpp_result = cpp_cox(eta, weights)

        assert np.isclose(py_result.deviance, cpp_result.deviance, rtol=1e-10)
        assert np.allclose(py_result.gradient, cpp_result.gradient, rtol=1e-10)
        assert np.allclose(py_result.diag_hessian, cpp_result.diag_hessian, rtol=1e-10)

    def test_single_stratum(self):
        """Test with a single stratum (should match non-stratified)."""
        np.random.seed(456)
        n = 50
        event = np.random.exponential(3, n)
        status = np.random.binomial(1, 0.8, n)
        strata = np.zeros(n, dtype=int)  # All in one stratum
        eta = np.random.randn(n) * 0.4

        py_cox = StratifiedCoxDeviance(event=event, status=status, strata=strata)
        py_result = py_cox(eta)

        cpp_cox = StratifiedCoxDevianceCpp(event=event, status=status, strata=strata)
        cpp_result = cpp_cox(eta)

        assert np.isclose(py_result.deviance, cpp_result.deviance, rtol=1e-10)
        assert np.allclose(py_result.gradient, cpp_result.gradient, rtol=1e-10)
        assert np.allclose(py_result.diag_hessian, cpp_result.diag_hessian, rtol=1e-10)

    def test_with_ties(self):
        """Test with tied event times (Efron tie-breaking)."""
        # Deliberately create ties
        event = np.array([1, 1, 2, 2, 2, 3, 3, 4, 5, 5,
                          1, 2, 2, 3, 3, 3, 4, 4, 5, 5])
        status = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 0,
                           1, 1, 1, 1, 0, 1, 1, 0, 1, 1])
        strata = np.repeat([0, 1], 10)
        eta = np.linspace(-1, 1, 20)

        py_cox = StratifiedCoxDeviance(event=event, status=status, strata=strata, tie_breaking='efron')
        py_result = py_cox(eta)

        cpp_cox = StratifiedCoxDevianceCpp(event=event, status=status, strata=strata, tie_breaking='efron')
        cpp_result = cpp_cox(eta)

        assert np.isclose(py_result.deviance, cpp_result.deviance, rtol=1e-10)
        assert np.allclose(py_result.gradient, cpp_result.gradient, rtol=1e-10)
        assert np.allclose(py_result.diag_hessian, cpp_result.diag_hessian, rtol=1e-10)

    def test_breslow_tie_breaking(self):
        """Test with Breslow tie-breaking."""
        event = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        status = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 0])
        strata = np.repeat([0, 1], 5)
        eta = np.linspace(-0.5, 0.5, 10)

        py_cox = StratifiedCoxDeviance(event=event, status=status, strata=strata, tie_breaking='breslow')
        py_result = py_cox(eta)

        cpp_cox = StratifiedCoxDevianceCpp(event=event, status=status, strata=strata, tie_breaking='breslow')
        cpp_result = cpp_cox(eta)

        assert np.isclose(py_result.deviance, cpp_result.deviance, rtol=1e-10)
        assert np.allclose(py_result.gradient, cpp_result.gradient, rtol=1e-10)
        assert np.allclose(py_result.diag_hessian, cpp_result.diag_hessian, rtol=1e-10)

    def test_with_start_times(self):
        """Test with left-truncation (start times)."""
        np.random.seed(789)
        n = 60
        start = np.random.uniform(0, 2, n)
        event = start + np.random.exponential(3, n)
        status = np.random.binomial(1, 0.7, n)
        strata = np.random.choice([0, 1, 2], n)
        eta = np.random.randn(n) * 0.3

        py_cox = StratifiedCoxDeviance(event=event, status=status, strata=strata, start=start)
        py_result = py_cox(eta)

        cpp_cox = StratifiedCoxDevianceCpp(event=event, status=status, strata=strata, start=start)
        cpp_result = cpp_cox(eta)

        assert np.isclose(py_result.deviance, cpp_result.deviance, rtol=1e-10)
        assert np.allclose(py_result.gradient, cpp_result.gradient, rtol=1e-10)
        assert np.allclose(py_result.diag_hessian, cpp_result.diag_hessian, rtol=1e-10)


class TestInformationMatrix:
    """Test information matrix (Hessian matvec) for stratified C++ implementation."""

    def test_matvec_matches_python(self):
        """Test that Hessian matvec matches Python implementation."""
        np.random.seed(42)
        n = 50
        event = np.random.exponential(5, n)
        status = np.random.binomial(1, 0.7, n)
        strata = np.random.choice([0, 1], n)
        eta = np.random.randn(n) * 0.3
        v = np.random.randn(n)

        py_cox = StratifiedCoxDeviance(event=event, status=status, strata=strata)
        py_cox(eta)  # Compute buffers
        py_info = py_cox.information(eta)

        cpp_cox = StratifiedCoxDevianceCpp(event=event, status=status, strata=strata)
        cpp_cox(eta)  # Compute buffers
        cpp_info = cpp_cox.information(eta)

        py_matvec = py_info @ v
        cpp_matvec = cpp_info @ v

        assert np.allclose(py_matvec, cpp_matvec, rtol=1e-9), \
            f"Matvec mismatch: max diff={np.abs(py_matvec - cpp_matvec).max()}"

    def test_matvec_symmetry(self):
        """Test that information matrix is symmetric."""
        np.random.seed(123)
        n = 40
        event = np.random.exponential(5, n)
        status = np.random.binomial(1, 0.7, n)
        strata = np.random.choice([0, 1, 2], n)
        eta = np.random.randn(n) * 0.3

        cpp_cox = StratifiedCoxDevianceCpp(event=event, status=status, strata=strata)
        cpp_cox(eta)
        info = cpp_cox.information(eta)

        # Test symmetry: v1^T * (H * v2) = v2^T * (H * v1)
        v1 = np.random.randn(n)
        v2 = np.random.randn(n)

        result1 = np.dot(v1, info @ v2)
        result2 = np.dot(v2, info @ v1)

        assert np.isclose(result1, result2, rtol=1e-10)


class TestZeroWeights:
    """Test handling of zero-weight observations."""

    def test_zero_weights_match_python(self):
        """Test that zero-weight handling matches Python."""
        np.random.seed(42)
        n = 60
        event = np.random.exponential(5, n)
        status = np.random.binomial(1, 0.7, n)
        strata = np.random.choice([0, 1], n)
        eta = np.random.randn(n) * 0.3

        # Set some weights to zero
        weights = np.ones(n)
        weights[::5] = 0.0  # Every 5th observation has zero weight

        py_cox = StratifiedCoxDeviance(event=event, status=status, strata=strata)
        py_result = py_cox(eta, weights)

        cpp_cox = StratifiedCoxDevianceCpp(event=event, status=status, strata=strata)
        cpp_result = cpp_cox(eta, weights)

        assert np.isclose(py_result.deviance, cpp_result.deviance, rtol=1e-10)
        assert np.allclose(py_result.gradient, cpp_result.gradient, rtol=1e-10)
        assert np.allclose(py_result.diag_hessian, cpp_result.diag_hessian, rtol=1e-10)


class TestEdgeCases:
    """Test edge cases for stratified C++ implementation."""

    def test_all_censored_stratum(self):
        """Test with a stratum where all observations are censored."""
        event = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        status = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        strata = np.repeat([0, 1], 5)
        eta = np.zeros(10)

        py_cox = StratifiedCoxDeviance(event=event, status=status, strata=strata)
        py_result = py_cox(eta)

        cpp_cox = StratifiedCoxDevianceCpp(event=event, status=status, strata=strata)
        cpp_result = cpp_cox(eta)

        assert np.isclose(py_result.deviance, cpp_result.deviance, rtol=1e-10)
        assert np.allclose(py_result.gradient, cpp_result.gradient, rtol=1e-10)
        assert np.allclose(py_result.diag_hessian, cpp_result.diag_hessian, rtol=1e-10)

    def test_many_strata(self):
        """Test with many small strata."""
        np.random.seed(42)
        n = 200
        event = np.random.exponential(5, n)
        status = np.random.binomial(1, 0.7, n)
        strata = np.arange(n) // 5  # 40 strata with 5 observations each
        eta = np.random.randn(n) * 0.3

        py_cox = StratifiedCoxDeviance(event=event, status=status, strata=strata)
        py_result = py_cox(eta)

        cpp_cox = StratifiedCoxDevianceCpp(event=event, status=status, strata=strata)
        cpp_result = cpp_cox(eta)

        assert np.isclose(py_result.deviance, cpp_result.deviance, rtol=1e-10)
        assert np.allclose(py_result.gradient, cpp_result.gradient, rtol=1e-10)
        assert np.allclose(py_result.diag_hessian, cpp_result.diag_hessian, rtol=1e-10)

    def test_large_eta_values(self):
        """Test numerical stability with large eta values."""
        np.random.seed(42)
        n = 50
        event = np.random.exponential(5, n)
        status = np.random.binomial(1, 0.7, n)
        strata = np.random.choice([0, 1], n)
        eta = np.random.randn(n) * 10  # Large eta values

        py_cox = StratifiedCoxDeviance(event=event, status=status, strata=strata)
        py_result = py_cox(eta)

        cpp_cox = StratifiedCoxDevianceCpp(event=event, status=status, strata=strata)
        cpp_result = cpp_cox(eta)

        # May have some numerical differences with large eta, use more tolerance
        assert np.isclose(py_result.deviance, cpp_result.deviance, rtol=1e-8)
        assert np.allclose(py_result.gradient, cpp_result.gradient, rtol=1e-8)
        assert np.allclose(py_result.diag_hessian, cpp_result.diag_hessian, rtol=1e-8)


class TestProperties:
    """Test properties of the StratifiedCoxDevianceCpp class."""

    def test_n_strata_property(self):
        """Test n_strata property."""
        event = np.array([1, 2, 3, 4, 5, 6])
        status = np.array([1, 0, 1, 1, 0, 1])
        strata = np.array([0, 0, 1, 1, 2, 2])

        cpp_cox = StratifiedCoxDevianceCpp(event=event, status=status, strata=strata)
        assert cpp_cox.n_strata == 3

    def test_n_total_property(self):
        """Test n_total property."""
        event = np.array([1, 2, 3, 4, 5, 6])
        status = np.array([1, 0, 1, 1, 0, 1])
        strata = np.array([0, 0, 1, 1, 2, 2])

        cpp_cox = StratifiedCoxDevianceCpp(event=event, status=status, strata=strata)
        assert cpp_cox.n_total == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
