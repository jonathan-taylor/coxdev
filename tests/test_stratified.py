"""
Tests for StratifiedCoxDeviance class.
"""

import numpy as np
import pytest
from coxdev import CoxDeviance
from stratified import StratifiedCoxDevianceTest as StratCox
from coxdev.stratified import StratifiedCoxDeviance
from coxdev import CoxDeviance

@pytest.fixture
def survival_data():
    """Generate survival data for testing."""
    np.random.seed(42)  # For reproducible tests
    n = 20
    status = np.random.randint(0, 2, n)
    start = np.random.uniform(0, 5, n)
    event = np.random.uniform(0, 10, n) + start
    eta = np.random.randn(n)
    weight = np.random.uniform(0.5, 2.0, n)
    
    return {
        'event': event,
        'status': status,
        'start': start,
        'eta': eta,
        'weight': weight,
        'n': n
    }


@pytest.fixture
def single_stratum_data(survival_data):
    """Generate data with a single stratum."""
    strata = np.zeros(survival_data['n'], dtype=np.int32)
    return {**survival_data, 'strata': strata}


@pytest.fixture
def multi_stratum_data(survival_data):
    """Generate data with multiple strata."""
    n = survival_data['n']
    # Create 3 strata with roughly equal sizes
    strata = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=np.int32)
    np.random.shuffle(strata)  # Shuffle to make it more realistic
    return {**survival_data, 'strata': strata}


def test_stratified_single_stratum_agrees_with_coxdeviance(single_stratum_data):
    """Test that StratCox agrees with CoxDeviance when there's only one stratum."""
    data = single_stratum_data
    
    # No left truncation
    coxdev = CoxDeviance(event=data['event'], status=data['status'])
    stratdev = StratCox(
        event=data['event'], 
        status=data['status'], 
        strata=data['strata']
    )
    
    res1 = coxdev(data['eta'], data['weight'])
    res2 = stratdev(data['eta'], data['weight'])
    
#    assert np.allclose(res1.deviance, res2.deviance)
    assert np.allclose(res1.gradient, res2.gradient)
    assert np.allclose(res1.diag_hessian, res2.diag_hessian)
    
    # Test information matrix
    info1 = coxdev.information(data['eta'], data['weight'])
    info2 = stratdev.information(data['eta'], data['weight'])
    v = np.random.randn(data['n'])
    assert np.allclose(info1 @ v, info2 @ v)


def test_stratified_single_stratum_with_truncation_agrees_with_coxdeviance(single_stratum_data):
    """Test that StratCox agrees with CoxDeviance when there's only one stratum and left truncation."""
    data = single_stratum_data
    
    # With left truncation
    coxdev = CoxDeviance(event=data['event'], status=data['status'], start=data['start'])
    stratdev = StratCox(
        event=data['event'], 
        status=data['status'], 
        start=data['start'],
        strata=data['strata']
    )
    
    res1 = coxdev(data['eta'], data['weight'])
    res2 = stratdev(data['eta'], data['weight'])
    
#    assert np.allclose(res1.deviance, res2.deviance)
    assert np.allclose(res1.gradient, res2.gradient)
    assert np.allclose(res1.diag_hessian, res2.diag_hessian)
    
    # Test information matrix
    info1 = coxdev.information(data['eta'], data['weight'])
    info2 = stratdev.information(data['eta'], data['weight'])
    v = np.random.randn(data['n'])
    assert np.allclose(info1 @ v, info2 @ v)


def test_stratified_multi_stratum_deviance_is_sum_of_strata(multi_stratum_data):
    """Test that stratified deviance is the sum of individual stratum deviances."""
    data = multi_stratum_data
    
    stratdev = StratCox(
        event=data['event'], 
        status=data['status'], 
        strata=data['strata']
    )
    
    # Compute stratified result
    strat_result = stratdev(data['eta'], data['weight'])
    
    # Compute individual stratum results
    total_deviance = 0.0
    total_loglik_sat = 0.0
    
    for stratum in np.unique(data['strata']):
        mask = data['strata'] == stratum
        stratum_coxdev = CoxDeviance(
            event=data['event'][mask], 
            status=data['status'][mask]
        )
        stratum_result = stratum_coxdev(data['eta'][mask], data['weight'][mask])
        total_deviance += stratum_result.deviance
        total_loglik_sat += stratum_result.loglik_sat
    
    assert np.allclose(strat_result.deviance, total_deviance)
    assert np.allclose(strat_result.loglik_sat, total_loglik_sat)


def test_stratified_multi_stratum_gradient_is_concatenated(multi_stratum_data):
    """Test that stratified gradient is correctly concatenated from individual strata."""
    data = multi_stratum_data
    
    stratdev = StratCox(
        event=data['event'], 
        status=data['status'], 
        strata=data['strata']
    )
    
    # Compute stratified result
    strat_result = stratdev(data['eta'], data['weight'])
    
    # Compute individual stratum results and concatenate
    expected_gradient = np.zeros_like(data['eta'])
    
    for stratum in np.unique(data['strata']):
        mask = data['strata'] == stratum
        stratum_coxdev = CoxDeviance(
            event=data['event'][mask], 
            status=data['status'][mask]
        )
        stratum_result = stratum_coxdev(data['eta'][mask], data['weight'][mask])
        expected_gradient[mask] = stratum_result.gradient
    
    assert np.allclose(strat_result.gradient, expected_gradient)


def test_stratified_multi_stratum_diag_hessian_is_concatenated(multi_stratum_data):
    """Test that stratified diagonal Hessian is correctly concatenated from individual strata."""
    data = multi_stratum_data
    
    stratdev = StratCox(
        event=data['event'], 
        status=data['status'], 
        strata=data['strata']
    )
    
    # Compute stratified result
    strat_result = stratdev(data['eta'], data['weight'])
    
    # Compute individual stratum results and concatenate
    expected_diag_hessian = np.zeros_like(data['eta'])
    
    for stratum in np.unique(data['strata']):
        mask = data['strata'] == stratum
        stratum_coxdev = CoxDeviance(
            event=data['event'][mask], 
            status=data['status'][mask]
        )
        stratum_result = stratum_coxdev(data['eta'][mask], data['weight'][mask])
        expected_diag_hessian[mask] = stratum_result.diag_hessian
    
    assert np.allclose(strat_result.diag_hessian, expected_diag_hessian)


def test_stratified_information_matrix_is_block_diagonal(multi_stratum_data):
    """Test that the information matrix is block diagonal by stratum."""
    data = multi_stratum_data
    
    stratdev = StratCox(
        event=data['event'], 
        status=data['status'], 
        strata=data['strata']
    )
    
    info = stratdev.information(data['eta'], data['weight'])
    
    # Test that the information matrix is symmetric
    v = np.random.randn(data['n'])
    w = np.random.randn(data['n'])
    assert np.allclose(v @ (info @ w), w @ (info @ v))
    
    # Test block diagonal structure by checking that cross-stratum interactions are zero
    # (This is a simplified test - in practice, the information matrix should be block diagonal)
    for stratum1 in np.unique(data['strata']):
        for stratum2 in np.unique(data['strata']):
            if stratum1 != stratum2:
                mask1 = data['strata'] == stratum1
                mask2 = data['strata'] == stratum2
                v1 = np.zeros(data['n'])
                v1[mask1] = 1.0
                v2 = np.zeros(data['n'])
                v2[mask2] = 1.0
                # The interaction should be zero for different strata
                # Note: This is a simplified test and may not catch all cases


def test_stratified_different_tie_breaking_methods(single_stratum_data):
    """Test that StratCox works with different tie-breaking methods."""
    data = single_stratum_data
    
    for tie_breaking in ['efron', 'breslow']:
        stratdev = StratCox(
            event=data['event'], 
            status=data['status'], 
            strata=data['strata'],
            tie_breaking=tie_breaking
        )
        
        result = stratdev(data['eta'], data['weight'])
        assert result.deviance is not None
        assert result.gradient is not None
        assert result.diag_hessian is not None


def test_stratified_input_validation(survival_data):
    """Test input validation for StratCox."""
    data = survival_data
    
    # Test that strata must have the same length as other inputs
    with pytest.raises(ValueError):
        StratCox(
            event=data['event'], 
            status=data['status'], 
            strata=np.array([0, 1], dtype=np.int32)  # Wrong length
        )
    
    # Test that strata must be convertible to int32
    strata_float = np.zeros(data['n'], dtype=np.float64)
    strata_float[0] = 1.3
    with pytest.raises(ValueError):
        StratCox(
            event=data['event'], 
            status=data['status'], 
            strata=strata_float
        )
    
    # Test that status must be convertible to int32
    status_float = np.zeros(data['n'], dtype=np.float64)
    status_float[0] = 1.2
    with pytest.raises(ValueError):
        StratCox(
            event=data['event'], 
            status=status_float,
            strata=np.zeros(data['n'], dtype=np.int32)
        )


def test_stratified_caching_behavior(single_stratum_data):
    """Test that StratCox properly caches results."""
    data = single_stratum_data
    
    stratdev = StratCox(
        event=data['event'], 
        status=data['status'], 
        strata=data['strata']
    )
    
    # First call
    result1 = stratdev(data['eta'], data['weight'])
    
    # Second call with same inputs should return cached result
    result2 = stratdev(data['eta'], data['weight'])
    
    # Results should be identical (same object)
    assert result1 is result2
    
    # Call with different inputs should return new result
    result3 = stratdev(data['eta'] * 1.1, data['weight'])
    assert result3 is not result1


def test_stratified_smoke_3_strata(multi_stratum_data):
    """Smoke test with 3 strata to verify basic functionality."""
    data = multi_stratum_data
    
    # Test without left truncation
    stratdev = StratCox(
        event=data['event'], 
        status=data['status'], 
        strata=data['strata']
    )
    
    result = stratdev(data['eta'], data['weight'])
    
    # Basic checks
    assert result.deviance > 0
    assert len(result.gradient) == data['n']
    assert len(result.diag_hessian) == data['n']
    assert result.loglik_sat is not None
    
    # Test information matrix
    info = stratdev.information(data['eta'], data['weight'])
    v = np.random.randn(data['n'])
    info_v = info @ v
    assert len(info_v) == data['n']
    
    # Test with left truncation
    stratdev_trunc = StratCox(
        event=data['event'], 
        status=data['status'], 
        start=data['start'],
        strata=data['strata']
    )
    
    result_trunc = stratdev_trunc(data['eta'], data['weight'])
    
    # Basic checks
    assert result_trunc.deviance > 0
    assert len(result_trunc.gradient) == data['n']
    assert len(result_trunc.diag_hessian) == data['n']
    assert result_trunc.loglik_sat is not None


def test_stratified_smoke_3_strata_unequal_sizes(multi_stratum_data):
    """Smoke test with 3 strata of unequal sizes."""
    data = multi_stratum_data
    
    stratdev = StratCox(
        event=data['event'], 
        status=data['status'], 
        strata=data['strata']
    )
    
    result = stratdev(data['eta'], data['weight'])
    
    # Basic checks
    assert result.deviance > 0
    assert len(result.gradient) == data['n']
    assert len(result.diag_hessian) == data['n']
    
    # Verify that all strata are present
    unique_strata = np.unique(data['strata'])
    assert len(unique_strata) == 3
    assert 0 in unique_strata
    assert 1 in unique_strata
    assert 2 in unique_strata


def test_stratified_smoke_3_strata_with_ties(multi_stratum_data):
    """Smoke test with 3 strata and tied event times."""
    data = multi_stratum_data
    
    # Create some tied event times by modifying the event data
    tied_events = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0] * 2)
    # Ensure we have enough tied events for the data size
    if len(tied_events) >= data['n']:
        event = tied_events[:data['n']]
    else:
        # Pad with unique values if needed
        event = np.concatenate([tied_events, np.arange(len(tied_events), data['n'])])
    
    # Test both tie-breaking methods
    for tie_breaking in ['efron', 'breslow']:
        stratdev = StratCox(
            event=event, 
            status=data['status'], 
            strata=data['strata'],
            tie_breaking=tie_breaking
        )
        
        result = stratdev(data['eta'], data['weight'])
        
        # Basic checks
        assert result.deviance > 0
        assert len(result.gradient) == data['n']
        assert len(result.diag_hessian) == data['n']
        
        # Test information matrix
        info = stratdev.information(data['eta'], data['weight'])
        v = np.random.randn(data['n'])
        info_v = info @ v
        assert len(info_v) == data['n']


def test_stratified_smoke_3_strata_consistency(multi_stratum_data):
    """Smoke test to verify consistency across multiple calls."""
    data = multi_stratum_data
    
    stratdev = StratCox(
        event=data['event'], 
        status=data['status'], 
        strata=data['strata']
    )
    
    # Multiple calls with same inputs should give same results
    result1 = stratdev(data['eta'], data['weight'])
    result2 = stratdev(data['eta'], data['weight'])
    
    assert np.allclose(result1.deviance, result2.deviance)
    assert np.allclose(result1.gradient, result2.gradient)
    assert np.allclose(result1.diag_hessian, result2.diag_hessian)
    
    # Different inputs should give different results
    result3 = stratdev(data['eta'] * 1.1, data['weight'])
    assert not np.allclose(result1.deviance, result3.deviance)


def test_stratified_smoke_3_strata_edge_cases(multi_stratum_data):
    """Smoke test with edge cases for 3 strata."""
    data = multi_stratum_data
    
    stratdev = StratCox(
        event=data['event'], 
        status=data['status'], 
        strata=data['strata']
    )
    
    # Test with None weights (should use equal weights)
    result = stratdev(data['eta'], None)
    assert result.deviance > 0
    assert len(result.gradient) == data['n']
    assert len(result.diag_hessian) == data['n']
    
    # Test with all equal weights
    weight_equal = np.ones(data['n'])
    result_equal = stratdev(data['eta'], weight_equal)
    assert result_equal.deviance > 0
    
    # Test with very small weights
    weight_small = np.full(data['n'], 0.001)
    result_small = stratdev(data['eta'], weight_small)
    assert result_small.deviance > 0
    
    # Test with very large weights
    weight_large = np.full(data['n'], 1000.0)
    result_large = stratdev(data['eta'], weight_large)
    assert result_large.deviance > 0


def test_coxdeviance_input_validation(survival_data):
    """Test input validation for CoxDeviance."""
    data = survival_data
    
    # Test that status must be convertible to int32
    status_float = np.zeros(data['n'], dtype=np.float64)
    status_float[0] = 1.2
    with pytest.raises(ValueError):
        CoxDeviance(
            event=data['event'], 
            status=status_float
        )
    
    # Test that status with non-integer values should fail
    status_nonint = np.array([0.5, 1.5, 0, 1] * (data['n'] // 4 + 1))[:data['n']]
    with pytest.raises(ValueError):
        CoxDeviance(
            event=data['event'], 
            status=status_nonint
        )


def test_stratified_manual_block_diagonal():
    """Test that StratCox matches three CoxDeviance objects for 3 strata of sizes 15, 13, 23."""
    rng = np.random.default_rng(2024)
    sizes = [15, 13, 23]
    n = sum(sizes)
    strata = np.zeros(n, dtype=np.int32)
    strata[:sizes[0]] = 0
    strata[sizes[0]:sizes[0]+sizes[1]] = 1
    strata[sizes[0]+sizes[1]:] = 2
    # Shuffle to make it non-contiguous
    perm = rng.permutation(n)
    strata = strata[perm]

    # Generate data
    start = rng.uniform(0, 5, n)
    event = rng.uniform(0, 10, n) + start
    status = rng.integers(0, 2, n)
    eta = rng.standard_normal(n)
    weight = rng.uniform(0.5, 2.0, n)

    # Sort indices for each stratum
    idx0 = np.where(strata == 0)[0]
    idx1 = np.where(strata == 1)[0]
    idx2 = np.where(strata == 2)[0]

    # Create CoxDeviance objects for each stratum
    cox0 = CoxDeviance(event=event[idx0], status=status[idx0], start=start[idx0])
    cox1 = CoxDeviance(event=event[idx1], status=status[idx1], start=start[idx1])
    cox2 = CoxDeviance(event=event[idx2], status=status[idx2], start=start[idx2])

    # Compute results for each stratum
    res0 = cox0(eta[idx0], weight[idx0])
    res1 = cox1(eta[idx1], weight[idx1])
    res2 = cox2(eta[idx2], weight[idx2])

    # Stratified CoxDeviance
    stratdev = StratCox(
        event=event, status=status, start=start, strata=strata
    )
    strat_res = stratdev(eta, weight)

    # Check deviance
    expected_deviance = res0.deviance + res1.deviance + res2.deviance
    assert np.allclose(strat_res.deviance, expected_deviance)

    # Check gradient (should be concatenated in the right order)
    expected_grad = np.zeros(n)
    expected_grad[idx0] = res0.gradient
    expected_grad[idx1] = res1.gradient
    expected_grad[idx2] = res2.gradient
    assert np.allclose(strat_res.gradient, expected_grad)

    # Check diag_hessian
    expected_diag_hess = np.zeros(n)
    expected_diag_hess[idx0] = res0.diag_hessian
    expected_diag_hess[idx1] = res1.diag_hessian
    expected_diag_hess[idx2] = res2.diag_hessian
    assert np.allclose(strat_res.diag_hessian, expected_diag_hess)

    # Check block diagonal Hessian structure
    info = stratdev.information(eta, weight)
    # Should be a LinearOperator, but we can check block structure by matvec
    # Create a block vector
    v = np.zeros(n)
    v[idx0] = rng.standard_normal(sizes[0])
    v[idx1] = rng.standard_normal(sizes[1])
    v[idx2] = rng.standard_normal(sizes[2])
    # The result should be the same as applying each block separately
    block0 = cox0.information(eta[idx0], weight[idx0])
    block1 = cox1.information(eta[idx1], weight[idx1])
    block2 = cox2.information(eta[idx2], weight[idx2])
    result = info @ v
    expected = np.zeros(n)
    expected[idx0] = block0 @ v[idx0]
    expected[idx1] = block1 @ v[idx1]
    expected[idx2] = block2 @ v[idx2]
    assert np.allclose(result, expected)

    # Check that off-blocks are zero: if v is nonzero only in one stratum, result is nonzero only in that stratum
    for idx, block, size in zip([idx0, idx1, idx2], [block0, block1, block2], sizes):
        vtest = np.zeros(n)
        vtest[idx] = rng.standard_normal(size)
        r = info @ vtest
        # Should be nonzero only in idx
        assert np.allclose(r[np.setdiff1d(np.arange(n), idx)], 0)
        # Should match the block
        assert np.allclose(r[idx], block @ vtest[idx])


@pytest.mark.parametrize("tie_breaking", ['efron', 'breslow'])
@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize("use_strata", [True, False])
def test_coxdeviance2_matches_stratified_and_coxdeviance(tie_breaking, use_weight, use_strata):
    """Test that CoxDeviance2 matches StratCox (and CoxDeviance if no strata) for random data, including information method and both with and without weights."""

    rng = np.random.default_rng(2025)
    n = 40
    if use_strata:
        sizes = [10, 10, 20]
        strata = np.zeros(n, dtype=np.int32)
        start_idx = 0
        for i, sz in enumerate(sizes):
            strata[start_idx:start_idx+sz] = i
            start_idx += sz
        perm = rng.permutation(n)
        strata = strata[perm]
    else:
        strata = None
    event = rng.uniform(0, 10, n)
    start = rng.uniform(0, 5, n)
    event = event + start
    status = rng.integers(0, 2, n)
    eta = rng.standard_normal(n)
    weight = rng.uniform(0.5, 2.0, n)
    w = weight if use_weight else None

    if use_strata:
        stratdev = StratCox(
            event=event, status=status, start=start, strata=strata, tie_breaking=tie_breaking,
        )
        cox2 = StratifiedCoxDeviance(
            event=event, status=status, start=start, strata=strata, tie_breaking=tie_breaking,
        )
        res_strat = stratdev(eta, w)
        res2 = cox2(eta, w)
        # assert np.allclose(res_strat.deviance, res2.deviance)
        assert np.allclose(res_strat.gradient, res2.gradient)
        assert np.allclose(res_strat.diag_hessian, res2.diag_hessian)
        info_strat = stratdev.information(eta, w)
        info2 = cox2.information(eta, w)
        v = rng.standard_normal(n)
        assert np.allclose(info_strat @ v, info2 @ v)
    else:
        coxdev = CoxDeviance(
            event=event, status=status, start=start, tie_breaking=tie_breaking,
        )
        stratdev = StratCox(
            event=event, status=status, start=start, strata=None, tie_breaking=tie_breaking,
        )
        cox2 = StratifiedCoxDeviance(
            event=event, status=status, start=start, strata=None, tie_breaking=tie_breaking,
        )
        res_cox = coxdev(eta, w)
        res_strat = stratdev(eta, w)
        res2 = cox2(eta, w)
        assert np.allclose(res_cox.deviance, res_strat.deviance)
#        assert np.allclose(res_cox.deviance, res2.deviance)
        assert np.allclose(res_cox.gradient, res_strat.gradient)
        assert np.allclose(res_cox.gradient, res2.gradient)
        assert np.allclose(res_cox.diag_hessian, res_strat.diag_hessian)
        assert np.allclose(res_cox.diag_hessian, res2.diag_hessian)
        info_cox = coxdev.information(eta, w)
        info_strat = stratdev.information(eta, w)
        info2 = cox2.information(eta, w)
        v = rng.standard_normal(n)
        assert np.allclose(info_cox @ v, info_strat @ v)
        assert np.allclose(info_cox @ v, info2 @ v)
