## Which version
import os
from dataclasses import dataclass

# Set an environment variable
os.environ['PY'] = 'true'


import numpy as np
import pandas as pd
from coxdev import CoxDeviance, CoxInformation
from coxc import hessian_matvec
@dataclass
class CoxInformationTest(CoxInformation):

    C_flag: bool = False

    def _matvec(self, arg):

        # this will compute risk sums if not already computed
        # at this linear_predictor and sample_weight
        
        result = self.result
        coxdev = self.coxdev

        if not self.C_flag:
            return CoxInformation._matvec(self, arg)
        else:

        # negative will give 2nd derivative of negative
        # loglikelihood

            hessian_matvec(-np.asarray(arg).reshape(-1),
                           np.asarray(result.linear_predictor),
                           np.asarray(result.sample_weight),
                           coxdev._risk_sum_buffers[0],
                           coxdev._diag_part_buffer,
                           coxdev._w_avg_buffer,
                           coxdev._exp_w_buffer,
                           coxdev._event_cumsum,
                           coxdev._start_cumsum,
                           coxdev._event_order,
                           coxdev._start_order,
                           coxdev._status,
                           coxdev._first,
                           coxdev._last,
                           coxdev._scaling,
                           coxdev._event_map,
                           coxdev._start_map,
                           coxdev._risk_sum_buffers,
                           coxdev._forward_cumsum_buffers,
                           coxdev._forward_scratch_buffer,
                           coxdev._reverse_cumsum_buffers,
                           coxdev._hess_matvec_buffer,
                           coxdev._have_start_times,                        
                           coxdev._efron)

        return coxdev._hess_matvec_buffer.copy()

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

import pytest
from simulate import (simulate_df,
                      all_combos,
                      rng,
                      sample_weights)

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


# @pytest.mark.parametrize('tie_types', all_combos)
# @pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
# @pytest.mark.parametrize('sample_weight', [np.ones, sample_weights])
# @pytest.mark.parametrize('have_start_times', [True, False])
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
    coxdev = CoxDeviance(event=data['event'],
                         start=start,
                         status=data['status'],
                         tie_breaking=tie_breaking)

    n = data.shape[0]
    p = n // 2
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p) / np.sqrt(n)
    weight = sample_weight(n)

    C = coxdev(X @ beta, weight)

    eta = X @ beta

    H = coxdev.information(eta,
                           weight)

    # this is the "information" method basically

    result = coxdev(eta,
                    weight)
    H_test = CoxInformationTest(result=result,
                                coxdev=coxdev,
                                C_flag=True)

    v = rng.standard_normal(H.shape[0])
    Hv = H @ v
    Hv_test = H_test @ v

    
test_coxph(tie_types = all_combos[100], tie_breaking = 'efron', sample_weight = sample_weights,
           have_start_times = True, nrep = 1, size =5, tol = 1e-10)
