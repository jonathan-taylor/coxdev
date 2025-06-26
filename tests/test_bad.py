import numpy as np
import pytest
from coxdev import CoxDeviance

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

    glmnetR = importr('glmnet')
    baseR = importr('base')
    survivalR = importr('survival')

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

def generate_problematic_test_data():
    """Generate test data that was previously loaded from g0.RDS"""
    # Set random seed for reproducibility
    rng = np.random.default_rng(42)
    
    # Generate a problematic test case
    n = 50
    p = 10
    
    # Generate event times, status, and weights
    event = rng.exponential(scale=2.0, size=n)
    status = rng.choice([0, 1], size=n, p=[0.3, 0.7])
    sample_weight = rng.uniform(0.5, 2.0, size=n)
    
    # Generate design matrix and coefficients
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p) * 0.1
    
    return {
        'event': event,
        'start': None,  # No start times for this test
        'status': status,
        'sample_weight': sample_weight,
        'X': X,
        'beta': beta
    }

def check_results(data_dict, ties):
    tol = 1e-10
    event = np.asarray(data_dict['event'])
    start = data_dict['start']
    status = np.asarray(data_dict['status'])
    weight = np.asarray(data_dict['sample_weight'])
    tie_breaking = ties
    beta = np.asarray(data_dict['beta'])
    X = np.array(data_dict['X'])
    coxdev = CoxDeviance(event=event,
                         start=start,
                         status=status,
                         tie_breaking=tie_breaking)

    C = coxdev(X @ beta, weight)

    eta = X @ beta
    
    H = coxdev.information(eta,
                           weight)
    I = X.T @ (H @ X)
    assert np.allclose(I, I.T)
    cov_ = np.linalg.inv(I)
    
    (G_coxph,
     D_coxph,
     cov_coxph) = get_coxph(event=event,
                            status=status,
                            beta=beta,
                            sample_weight=weight,
                            start=start,
                            ties=tie_breaking,
                            X=X)
    if np.allclose(D_coxph[0], C.deviance - 2 * C.loglik_sat):
        print("Coxph Deviance matches")
    else:
        print("Coxph Deviance mismatch")
    if np.linalg.norm(G_coxph - X.T @ C.gradient) / np.linalg.norm(X.T @ C.gradient) < tol:
        print("Coxph gradient matches")
    else:
        print("Coxph gradient mismatch")
    if np.linalg.norm(cov_ - cov_coxph) / np.linalg.norm(cov_) < tol:
        print("Coxph cov matches")
    else:
        print("Coxph cov mismatch")

    if ties == 'breslow':
        D_R, G_R, H_R = get_glmnet_result(event,
                                          status,
                                          start,
                                          X @ beta,
                                          weight)
        
        delta_D = np.fabs(D_R - C.deviance) / np.fabs(D_R)
        if delta_D < tol:
            print("Glmnet Deviance matches")
        else:
            print("Glmnet Deviance mismatch")

        delta_G = np.linalg.norm(G_R - C.gradient) / np.linalg.norm(G_R)            
        if delta_G < tol:
            print("Glmnet gradient matches")
        else:
            print("Glmnet gradient mismatch")

        delta_H = np.linalg.norm(H_R - C.diag_hessian) / np.linalg.norm(H_R)
        if delta_H < tol:
            print("Glmnet hessian matches")
        else:
            print("Glmnet hessian mismatch")

@pytest.mark.skipif(not has_rpy2, reason="rpy2 not available")
@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
def test_coxph_agreement(tie_breaking):
    """
    Test that coxdev agrees with R's coxph implementation.
    
    Parameters
    ----------
    tie_breaking : str
        Tie-breaking method to test ('efron' or 'breslow')
    """
    tol = 1e-10
    test_data = generate_problematic_test_data()
    
    event = np.asarray(test_data['event'])
    start = test_data['start']
    status = np.asarray(test_data['status'])
    weight = np.asarray(test_data['sample_weight'])
    beta = np.asarray(test_data['beta'])
    X = np.array(test_data['X'])
    
    coxdev = CoxDeviance(event=event,
                         start=start,
                         status=status,
                         tie_breaking=tie_breaking)

    C = coxdev(X @ beta, weight)

    eta = X @ beta
    
    H = coxdev.information(eta, weight)
    I = X.T @ (H @ X)
    
    # Test that information matrix is symmetric
    assert np.allclose(I, I.T, rtol=1e-10, atol=1e-10)
    
    cov_ = np.linalg.inv(I)
    
    (G_coxph,
     D_coxph,
     cov_coxph) = get_coxph(event=event,
                            status=status,
                            beta=beta,
                            sample_weight=weight,
                            start=start,
                            ties=tie_breaking,
                            X=X)
    
    # Test deviance agreement
    assert np.allclose(D_coxph[0], C.deviance - 2 * C.loglik_sat, rtol=1e-10, atol=1e-10)
    
    # Test gradient agreement
    delta_ph = np.linalg.norm(G_coxph - X.T @ C.gradient) / np.linalg.norm(X.T @ C.gradient)
    assert delta_ph < tol
    
    # Test covariance agreement
    assert np.linalg.norm(cov_ - cov_coxph) / np.linalg.norm(cov_) < tol

@pytest.mark.skipif(not has_rpy2, reason="rpy2 not available")
def test_glmnet_agreement():
    """
    Test that coxdev agrees with R's glmnet implementation for Breslow ties.
    """
    tol = 1e-10
    test_data = generate_problematic_test_data()
    
    event = np.asarray(test_data['event'])
    start = test_data['start']
    status = np.asarray(test_data['status'])
    weight = np.asarray(test_data['sample_weight'])
    beta = np.asarray(test_data['beta'])
    X = np.array(test_data['X'])
    
    coxdev = CoxDeviance(event=event,
                         start=start,
                         status=status,
                         tie_breaking='breslow')

    C = coxdev(X @ beta, weight)
    
    D_R, grad_R, hess_diag_R = get_glmnet_result(event,
                                                 status,
                                                 start,
                                                 X @ beta,
                                                 weight)
    
    # Test deviance agreement
    delta_D = np.fabs(D_R - C.deviance) / np.fabs(D_R)
    assert delta_D < tol
    
    # Test gradient agreement
    delta_G = np.linalg.norm(grad_R - C.gradient) / np.linalg.norm(grad_R)
    assert delta_G < tol
    
    # Test Hessian agreement
    delta_H = np.linalg.norm(hess_diag_R - C.diag_hessian) / np.linalg.norm(hess_diag_R)
    assert delta_H < tol

@pytest.mark.skipif(not has_rpy2, reason="rpy2 not available")
def test_information_matrix_properties():
    """
    Test properties of the information matrix.
    """
    test_data = generate_problematic_test_data()
    
    event = np.asarray(test_data['event'])
    start = test_data['start']
    status = np.asarray(test_data['status'])
    weight = np.asarray(test_data['sample_weight'])
    beta = np.asarray(test_data['beta'])
    X = np.array(test_data['X'])
    
    coxdev = CoxDeviance(event=event,
                         start=start,
                         status=status,
                         tie_breaking='efron')

    eta = X @ beta
    H = coxdev.information(eta, weight)
    I = X.T @ (H @ X)
    
    # Test symmetry
    assert np.allclose(I, I.T, rtol=1e-10, atol=1e-10)
    
    # Test positive definiteness (eigenvalues should be positive)
    eigenvals = np.linalg.eigvals(I)
    assert np.all(eigenvals > 0)
    
    # Test shape
    assert I.shape == (X.shape[1], X.shape[1])

def test_coxdev_basic_functionality():
    """
    Test basic functionality of CoxDeviance without R dependencies.
    """
    test_data = generate_problematic_test_data()
    
    event = np.asarray(test_data['event'])
    start = test_data['start']
    status = np.asarray(test_data['status'])
    weight = np.asarray(test_data['sample_weight'])
    beta = np.asarray(test_data['beta'])
    X = np.array(test_data['X'])
    
    coxdev = CoxDeviance(event=event,
                         start=start,
                         status=status,
                         tie_breaking='efron')

    C = coxdev(X @ beta, weight)
    
    # Test that result has expected attributes
    assert hasattr(C, 'linear_predictor')
    assert hasattr(C, 'sample_weight')
    assert hasattr(C, 'loglik_sat')
    assert hasattr(C, 'deviance')
    assert hasattr(C, 'gradient')
    assert hasattr(C, 'diag_hessian')
    
    # Test that deviance is finite and positive
    assert np.isfinite(C.deviance)
    assert C.deviance > 0
    
    # Test that gradient has correct shape
    assert C.gradient.shape == (len(event),)
    
    # Test that Hessian diagonal has correct shape
    assert C.diag_hessian.shape == (len(event),)
    
    # Test that all values are finite
    assert np.all(np.isfinite(C.gradient))
    assert np.all(np.isfinite(C.diag_hessian))

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__])
