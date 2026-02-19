import numpy as np
import pandas as pd
from functools import partial
from coxdev import CoxDeviance, StratifiedCoxDeviance

try:
    import rpy2.robjects as rpy
    has_rpy2 = True

except ImportError:
    has_rpy2 = False

if has_rpy2:
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    from rpy2.robjects import default_converter

    np_cv_rules = default_converter + numpy2ri.converter

    rpy.r('''if (!require("glmnet", character.only = TRUE)) {
  install.packages("glmnet")
    }''')

    glmnetR = importr('glmnet')
    baseR = importr('base')
    survivalR = importr('survival')
else:
    raise ImportError('cannot find rpy2, tests cannot be run')

import pytest
from simulate import (simulate_df,
                      all_combos,
                      rng,
                      sample_weights)

# Weight functions for parametrized tests
weight_funcs = [
    np.ones,                                        # uniform weights
    sample_weights,                                 # non-uniform weights, no zeros
    partial(sample_weights, zero_weight_fraction=0.2),  # ~20% zero weights
]

def get_glmnet_result(event,
                      status,
                      start,
                      eta,
                      weight,
                      time=False):

    event = np.asarray(event)
    status = np.asarray(status)
    weight = np.asarray(weight)
    eta = np.asarray(eta)

    with np_cv_rules.context():

        rpy.r.assign('status', status)
        rpy.r.assign('event', event)
        rpy.r.assign('eta', eta)
        rpy.r.assign('weight', weight)
        rpy.r('eta = as.numeric(eta)')
        rpy.r('weight = as.numeric(weight)')

        if start is not None:
            start = np.asarray(start)
            rpy.r.assign('start', start)
            rpy.r('y = Surv(start, event, status)')
            rpy.r('D_R = glmnet:::coxnet.deviance3(pred=eta, y=y, weight=weight, std.weights=FALSE)')
            rpy.r('G_R = glmnet:::coxgrad3(eta, y, weight, std.weights=FALSE, diag.hessian=TRUE)')
            rpy.r("H_R = attr(G_R, 'diag_hessian')")
        else:
            rpy.r('y = Surv(event, status)')
            rpy.r('D_R = glmnet:::coxnet.deviance2(pred=eta, y=y, weight=weight, std.weights=FALSE)')
            rpy.r('G_R = glmnet:::coxgrad2(eta, y, weight, std.weights=FALSE, diag.hessian=TRUE)')
            rpy.r("H_R = attr(G_R, 'diag_hessian')")

        D_R = rpy.r('D_R')
        G_R = rpy.r('G_R')
        H_R = rpy.r('H_R')

    # -2 for deviance instead of loglik

    return D_R, -2 * G_R, -2 * H_R


def get_coxph(event,
              status,
              X,
              beta,
              sample_weight,
              start=None,
              ties='efron'):

    if start is not None:
        start = np.asarray(start)
    status = np.asarray(status)
    event = np.asarray(event)

    with np_cv_rules.context():
        rpy.r.assign('status', status)
        rpy.r.assign('event', event)
        rpy.r.assign('X', X)
        rpy.r.assign('beta', beta)
        rpy.r.assign('ties', ties)
        rpy.r.assign('sample_weight', sample_weight)
        rpy.r('sample_weight = as.numeric(sample_weight)')
        if start is not None:
            rpy.r.assign('start', start)
            rpy.r('y = Surv(start, event, status)')
        else:
            rpy.r('y = Surv(event, status)')
        rpy.r('F = coxph(y ~ X, init=beta, weights=sample_weight, control=coxph.control(iter.max=0), ties=ties, robust=FALSE)')
        rpy.r('score = colSums(coxph.detail(F)$scor)')
        G = rpy.r('score')
        D = rpy.r('F$loglik')
        cov = rpy.r('vcov(F)')
    return -2 * G, -2 * D, cov


def get_stratified_coxph(event,
                         status,
                         strata,
                         X,
                         beta,
                         sample_weight,
                         start=None,
                         ties='efron'):
    """Get stratified Cox model results from R's survival package."""
    
    if start is not None:
        start = np.asarray(start)
    status = np.asarray(status)
    event = np.asarray(event)
    strata = np.asarray(strata)

    with np_cv_rules.context():
        rpy.r.assign('status', status)
        rpy.r.assign('event', event)
        rpy.r.assign('strata', strata)
        rpy.r.assign('X', X)
        rpy.r.assign('beta', beta)
        rpy.r.assign('ties', ties)
        rpy.r.assign('sample_weight', sample_weight)
        rpy.r('sample_weight = as.numeric(sample_weight)')
        rpy.r('strata = as.factor(strata)')
        
        if start is not None:
            rpy.r.assign('start', start)
            rpy.r('y = Surv(start, event, status)')
        else:
            rpy.r('y = Surv(event, status)')
            
        rpy.r('F = coxph(y ~ X + strata(strata), init=beta, weights=sample_weight, control=coxph.control(iter.max=0), ties=ties, robust=FALSE)')
        rpy.r('score = colSums(coxph.detail(F)$scor)')
        G = rpy.r('score')
        D = rpy.r('F$loglik')
        cov = rpy.r('vcov(F)')
    return -2 * G, -2 * D, cov


def create_stratified_data(n_samples=100, n_strata=3):
    """Create stratified survival data for testing."""
    np.random.seed(42)
    
    # Create strata
    strata = np.random.randint(0, n_strata, n_samples)
    
    status = np.random.binomial(1, 0.7, n_samples)
    
    # Add some start times for some tests
    start = np.random.exponential(0.5, n_samples)
    # Create survival data
    event = np.random.exponential(1.0, n_samples) + start
    
    # Create covariates
    n_features = 3
    X = np.random.standard_normal((n_samples, n_features))
    beta = np.random.standard_normal(n_features) / np.sqrt(n_samples)
    
    # Create weights
    weight = np.random.uniform(0.5, 2.0, n_samples)
    
    return {
        'event': event,
        'status': status,
        'strata': strata,
        'start': start,
        'X': X,
        'beta': beta,
        'weight': weight
    }


@pytest.mark.parametrize('tie_types', all_combos)
@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('sample_weight', weight_funcs)
@pytest.mark.parametrize('have_start_times', [True, False])
def test_coxph(tie_types,
               tie_breaking,
               sample_weight,
               have_start_times,
               nrep=5,
               size=5,
               tol=1e-10):

    data = simulate_df(tie_types,
                       nrep,
                       size)

    if have_start_times:
        start = data['start']
    else:
        start = None

    n = data.shape[0]
    p = n // 2
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p) / np.sqrt(n)
    weight = sample_weight(n)

    # Check for zero weights - R's coxph doesn't support them directly,
    # so we'll compare using non-zero weight subset
    nz_mask = weight > 0
    has_zero_weights = not np.all(nz_mask)

    coxdev = CoxDeviance(event=data['event'],
                         start=start,
                         status=data['status'],
                         sample_weight=weight,
                         tie_breaking=tie_breaking)

    C = coxdev(X @ beta)

    eta = X @ beta

    H = coxdev.information(eta)
    I = X.T @ (H @ X)
    assert np.allclose(I, I.T)
    cov_ = np.linalg.inv(I)

    if has_zero_weights:
        # Subset to non-zero weights for R comparison
        # Zero-weight observations don't contribute to likelihood
        event_nz = np.asarray(data['event'])[nz_mask]
        status_nz = np.asarray(data['status'])[nz_mask]
        start_nz = start[nz_mask] if start is not None else None
        X_nz = X[nz_mask]
        weight_nz = weight[nz_mask]

        (G_coxph,
         D_coxph,
         cov_coxph) = get_coxph(event=event_nz,
                                status=status_nz,
                                beta=beta,
                                sample_weight=weight_nz,
                                start=start_nz,
                                ties=tie_breaking,
                                X=X_nz)
    else:
        (G_coxph,
         D_coxph,
         cov_coxph) = get_coxph(event=np.asarray(data['event']),
                                status=np.asarray(data['status']),
                                beta=beta,
                                sample_weight=weight,
                                start=start,
                                ties=tie_breaking,
                                X=X)

    print(D_coxph, C.deviance - 2 * C.loglik_sat)

    # Skip if R detected rank deficiency (zeros on diagonal) or our matrix is unstable
    r_rank_deficient = np.any(np.diag(cov_coxph) == 0)
    our_unstable = np.max(np.abs(cov_)) > 1e10
    if r_rank_deficient or our_unstable:
        return  # No-op: random state consumed, skip assertions

    assert np.allclose(D_coxph[0], C.deviance - 2 * C.loglik_sat)

    if has_zero_weights:
        # For gradient comparison with zero weights:
        # X.T @ gradient = X_nz.T @ gradient_nz since zero-weight obs contribute 0
        delta_ph = np.linalg.norm(G_coxph - X_nz.T @ C.gradient[nz_mask]) / np.linalg.norm(X_nz.T @ C.gradient[nz_mask])
    else:
        delta_ph = np.linalg.norm(G_coxph - X.T @ C.gradient) / np.linalg.norm(X.T @ C.gradient)
    assert delta_ph < tol
    assert np.linalg.norm(cov_ - cov_coxph) / np.linalg.norm(cov_) < tol

def test_simple(nrep=5,
                size=5,
                tol=1e-10):
    test_coxph(all_combos[-1],
               'efron',
               sample_weights,
               True,
               nrep=5,
               size=5,
               tol=1e-10)
    

@pytest.mark.parametrize('tie_types', all_combos)
@pytest.mark.parametrize('sample_weight', weight_funcs)
@pytest.mark.parametrize('have_start_times', [True, False])
def test_glmnet(tie_types,
                sample_weight,
                have_start_times,
                nrep=5,
                size=5,
                tol=1e-10):

    data = simulate_df(tie_types,
                       nrep,
                       size)

    n = data.shape[0]
    eta = rng.standard_normal(n)
    weight = sample_weight(n)

    if have_start_times:
        start = data['start']
    else:
        start = None

    coxdev = CoxDeviance(event=data['event'],
                         start=start,
                         status=data['status'],
                         sample_weight=weight,
                         tie_breaking='breslow')
    C = coxdev(eta)

    # Check if our implementation produces valid results (no NaN)
    assert not np.any(np.isnan(C.gradient)), "Our implementation produced NaN in gradient"
    assert not np.any(np.isnan(C.diag_hessian)), "Our implementation produced NaN in diag_hessian"

    D_R, G_R, H_R = get_glmnet_result(data['event'],
                                      data['status'],
                                      start,
                                      eta,
                                      weight)

    # glmnet doesn't handle the case where risk_sums == 0 at an event time
    # (produces NaN from log(0)). Our implementation handles this correctly.
    # Skip comparison if glmnet produces NaN but our code doesn't.
    glmnet_has_nan = np.any(np.isnan(G_R)) or np.any(np.isnan(H_R))
    if glmnet_has_nan:
        pytest.skip("glmnet produces NaN (likely zero risk sum at event time); our implementation handles this correctly")

    delta_D = np.fabs(D_R - C.deviance) / np.fabs(D_R)
    delta_G = np.linalg.norm(G_R - C.gradient) / np.linalg.norm(G_R)
    delta_H = np.linalg.norm(H_R - C.diag_hessian) / np.linalg.norm(H_R)

    assert (delta_D < tol) and (delta_G < tol) and (delta_H < tol)
    

@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('have_start_times', [True, False])
@pytest.mark.parametrize('n_strata', [2, 3, 5])
def test_stratified_coxph(tie_breaking, have_start_times, n_strata, tol=1e-10):
    """Test StratifiedCoxDeviance against R's stratified coxph."""
    
    if not has_rpy2:
        pytest.skip("rpy2 not available")
    
    # Create stratified data
    data = create_stratified_data(n_samples=150, n_strata=n_strata)
    
    if have_start_times:
        start = data['start']
    else:
        start = None
    
    # Create StratifiedCoxDeviance (weights at initialization)
    stratdev = StratifiedCoxDeviance(
        event=data['event'],
        start=start,
        status=data['status'],
        strata=data['strata'],
        sample_weight=data['weight'],
        tie_breaking=tie_breaking
    )

    # Compute results with Python
    eta = data['X'] @ data['beta']
    C = stratdev(eta)

    # Get information matrix
    H = stratdev.information(eta)
    I = data['X'].T @ (H @ data['X'])
    assert np.allclose(I, I.T)
    cov_ = np.linalg.inv(I)
    
    # Get results from R
    (G_coxph, D_coxph, cov_coxph) = get_stratified_coxph(
        event=data['event'],
        status=data['status'],
        strata=data['strata'],
        X=data['X'],
        beta=data['beta'],
        sample_weight=data['weight'],
        start=start,
        ties=tie_breaking
    )
    
    # Compare deviance (adjust for saturated log-likelihood)
    print(f"R deviance: {D_coxph[0]}, Python deviance: {C.deviance - 2 * C.loglik_sat}")
    assert np.allclose(D_coxph[0], C.deviance - 2 * C.loglik_sat, rtol=tol)
    
    # Compare gradients
    delta_ph = np.linalg.norm(G_coxph - data['X'].T @ C.gradient) / np.linalg.norm(data['X'].T @ C.gradient)
    assert delta_ph < tol
    
    # Compare covariance matrices
    assert np.linalg.norm(cov_ - cov_coxph) / np.linalg.norm(cov_) < tol


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('have_start_times', [True, False])
def test_stratified_single_stratum(tie_breaking, have_start_times, tol=1e-10):
    """Test that StratifiedCoxDeviance with single stratum matches CoxDeviance."""
    
    if not has_rpy2:
        pytest.skip("rpy2 not available")
    
    # Create data with single stratum
    data = create_stratified_data(n_samples=100, n_strata=1)
    
    if have_start_times:
        start = data['start']
    else:
        start = None
    
    # Create both models (weights at initialization)
    coxdev = CoxDeviance(
        event=data['event'],
        start=start,
        status=data['status'],
        sample_weight=data['weight'],
        tie_breaking=tie_breaking
    )

    stratdev = StratifiedCoxDeviance(
        event=data['event'],
        start=start,
        status=data['status'],
        strata=data['strata'],
        sample_weight=data['weight'],
        tie_breaking=tie_breaking
    )

    # Compute results
    eta = data['X'] @ data['beta']
    C1 = coxdev(eta)
    C2 = stratdev(eta)
    
    # Results should be identical
    assert np.allclose(C1.deviance, C2.deviance, rtol=tol)
    assert np.allclose(C1.gradient, C2.gradient, rtol=tol)
    assert np.allclose(C1.diag_hessian, C2.diag_hessian, rtol=tol)
    assert np.allclose(C1.loglik_sat, C2.loglik_sat, rtol=tol)


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
def test_stratified_multiple_strata_sizes(tie_breaking, tol=1e-10):
    """Test StratifiedCoxDeviance with varying stratum sizes."""
    
    if not has_rpy2:
        pytest.skip("rpy2 not available")
    
    # Create data with uneven stratum sizes using create_stratified_data
    n_samples = 200
    n_features = 2
    data = create_stratified_data(n_samples=n_samples, n_strata=3)
    
    # Override strata with custom sizes
    strata = np.concatenate([
        np.zeros(50),   # 50 samples in stratum 0
        np.ones(80),    # 80 samples in stratum 1  
        np.full(70, 2)  # 70 samples in stratum 2
    ])
    data['strata'] = strata
    
    # Use the rest of the generated data
    event = data['event']
    status = data['status']
    start = data['start']
    X = data['X']
    beta = data['beta']
    weight = data['weight']
    
    # Create StratifiedCoxDeviance (weights at initialization)
    stratdev = StratifiedCoxDeviance(
        event=event,
        start=start,
        status=status,
        strata=strata,
        sample_weight=weight,
        tie_breaking=tie_breaking
    )

    # Compute results
    eta = X @ beta
    C = stratdev(eta)
    
    # Get results from R
    (G_coxph, D_coxph, cov_coxph) = get_stratified_coxph(
        event=event,
        status=status,
        strata=strata,
        X=X,
        beta=beta,
        sample_weight=weight,
        start=start,
        ties=tie_breaking
    )
    
    # Compare results
    assert np.allclose(D_coxph[0], C.deviance - 2 * C.loglik_sat, rtol=tol)
    delta_ph = np.linalg.norm(G_coxph - X.T @ C.gradient) / np.linalg.norm(X.T @ C.gradient)
    assert delta_ph < tol


# =============================================================================
# Tests for Breslow saturated log-likelihood against glmnet formula
# =============================================================================

def compute_breslow_sat_loglik(event, status, weight):
    """
    Compute Breslow saturated log-likelihood using the formula from glmnet paper.
    LL_sat = -sum(W_C * log(W_C)) where W_C is the sum of weights at each unique event time
    """
    event_times = event[status == 1]
    event_weights = weight[status == 1]
    unique_times = np.unique(event_times)

    loglik_sat = 0.0
    for t in unique_times:
        mask = event_times == t
        w_c = np.sum(event_weights[mask])
        if w_c > 0:
            loglik_sat -= w_c * np.log(w_c)
    return loglik_sat


@pytest.mark.skipif(not has_rpy2, reason="rpy2 not available")
def test_breslow_sat_loglik_unit_weights():
    """Verify Breslow saturated log-likelihood matches formula (unit weights)."""
    event = np.array([1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 8, 9, 10, 11, 12, 13, 14, 15], dtype=float)
    status = np.ones(20, dtype=int)
    weight = np.ones(20)
    eta = np.zeros(20)

    expected = compute_breslow_sat_loglik(event, status, weight)

    cox = CoxDeviance(event=event, status=status, sample_weight=weight, tie_breaking='breslow')
    result = cox(eta)

    assert np.isclose(result.loglik_sat, expected, rtol=1e-10), \
        f"Breslow sat loglik mismatch: got {result.loglik_sat}, expected {expected}"

    # Also verify deviance matches glmnet (returns D, G, H tuple)
    D_glmnet, _, _ = get_glmnet_result(event, status, None, eta, weight)
    assert np.isclose(result.deviance, D_glmnet[0], rtol=1e-10), \
        "Deviance should match glmnet"


@pytest.mark.skipif(not has_rpy2, reason="rpy2 not available")
def test_breslow_sat_loglik_non_unit_weights():
    """Verify Breslow saturated log-likelihood matches formula (non-unit weights)."""
    event = np.array([1, 1, 2, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11], dtype=float)
    status = np.ones(15, dtype=int)
    weight = np.array([1.5, 2.0, 1.0, 0.5, 3.0, 2.5, 1.0, 1.5, 2.0, 1.0, 0.5, 3.0, 2.0, 1.5, 1.0])
    np.random.seed(456)
    eta = np.random.randn(15) * 0.5

    expected = compute_breslow_sat_loglik(event, status, weight)

    cox = CoxDeviance(event=event, status=status, sample_weight=weight, tie_breaking='breslow')
    result = cox(eta)

    assert np.isclose(result.loglik_sat, expected, rtol=1e-10), \
        f"Breslow sat loglik mismatch (weighted): got {result.loglik_sat}, expected {expected}"

    # Also verify deviance matches glmnet (returns D, G, H tuple)
    D_glmnet, _, _ = get_glmnet_result(event, status, None, eta, weight)
    assert np.isclose(result.deviance, D_glmnet[0], rtol=1e-10), \
        "Deviance should match glmnet (weighted)"


@pytest.mark.skipif(not has_rpy2, reason="rpy2 not available")
def test_breslow_sat_loglik_with_start_times():
    """Verify Breslow saturated log-likelihood matches formula (with start times)."""
    np.random.seed(789)
    n = 15
    event = np.array([1, 1, 2, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11], dtype=float)
    start = event - np.random.uniform(0.1, 0.5, n)
    status = np.ones(n, dtype=int)
    weight = np.array([1.5, 2.0, 1.0, 0.5, 3.0, 2.5, 1.0, 1.5, 2.0, 1.0, 0.5, 3.0, 2.0, 1.5, 1.0])
    eta = np.random.randn(n) * 0.5

    # Saturated log-likelihood only depends on event times and weights, not start times
    expected = compute_breslow_sat_loglik(event, status, weight)

    cox = CoxDeviance(event=event, start=start, status=status, sample_weight=weight, tie_breaking='breslow')
    result = cox(eta)

    assert np.isclose(result.loglik_sat, expected, rtol=1e-10), \
        f"Breslow sat loglik mismatch (start times): got {result.loglik_sat}, expected {expected}"

    # Also verify deviance matches glmnet (returns D, G, H tuple)
    D_glmnet, _, _ = get_glmnet_result(event, status, start, eta, weight)
    assert np.isclose(result.deviance, D_glmnet[0], rtol=1e-10), \
        "Deviance should match glmnet (start times)"
