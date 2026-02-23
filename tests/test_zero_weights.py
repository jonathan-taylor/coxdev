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
                      size=10,
                      tol=1e-10):

    data = simulate_df(tie_types,
                       nrep,
                       size,
                       noinfo=True)
    
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
    H = H @ np.eye(H.shape[0])
    
    coxdev_no0 = CoxDeviance(event=data['event'][keep],
                             start=start_no0,
                             status=data['status'][keep],
                             tie_breaking=tie_breaking)
    C0 = coxdev_no0(eta[keep], weight[keep])

    H0 = coxdev_no0.information(eta[keep],
                                weight[keep])
    H0 = H0 @ np.eye(H0.shape[0])
    
    keep = np.nonzero(keep)[0]
    G0 = C0.gradient
    G1 = C.gradient[keep]
    G0 = G0[np.fabs(G0) > 1e-12]
    G1 = G1[np.fabs(G1) > 1e-12]
    assert np.allclose(G0, G1)
    assert np.allclose(C0.deviance, C.deviance)
    H_keep = H[np.ix_(keep, keep)]
    mask = (np.isnan(H_keep).sum(1) + np.isnan(H_keep).sum(0) > 0)
    assert np.allclose(H0[np.ix_(~mask, ~mask)], H_keep[np.ix_(~mask, ~mask)])
