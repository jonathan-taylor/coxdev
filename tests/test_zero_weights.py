import numpy as np
import pandas as pd
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
from .simulate import (simulate_df,
                       all_combos,
                       rng,
                       sample_weights,
                       sample_weights_zeros)


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


#@pytest.mark.skip(reason='some of the zero weights are correct, not all'
@pytest.mark.parametrize('tie_types', all_combos)
@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('sample_weight', [sample_weights_zeros])
@pytest.mark.parametrize('have_start_times', [True, False])
def test_zero_weights(tie_types,
                      tie_breaking,
                      sample_weight,
                      have_start_times,
                      nrep=5,
                      size=5,
                      tol=1e-10):

    data = simulate_df(tie_types,
                       nrep,
                       size)
    n = data.shape[0]
    
    weight = sample_weight(n)

    keep = weight > 0

    if have_start_times:
        start = data['start']
        start_no0 = start[keep]
    else:
        start = start_no0 = None

    coxdev = CoxDeviance(event=data['event'],
                         start=start,
                         status=data['status'],
                         tie_breaking=tie_breaking)

    n = data.shape[0]
    eta = rng.standard_normal(n)

    C = coxdev(eta, weight)

    H = coxdev.information(eta,
                           weight)

    coxdev_no0 = CoxDeviance(event=data['event'][keep],
                             start=start_no0,
                             status=data['status'][keep],
                             tie_breaking=tie_breaking)
    C0 = coxdev_no0(eta[keep], weight[keep])

    H0 = coxdev_no0.information(eta[keep],
                            weight[keep])

    mask = np.isnan(C0.gradient) + np.isnan(C.gradient[keep])
    G0 = C0.gradient[~mask]
    G1 = C.gradient[keep][~mask]
    G0 = G0[np.fabs(G0) > 1e-12]
    G1 = G1[np.fabs(G1) > 1e-12]
    assert np.allclose(G0, G1)
