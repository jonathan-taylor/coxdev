import numpy as np
import pandas as pd
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

import pytest
from simulate import (simulate_df,
                      all_combos,
                      rng,
                      sample_weights)

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


@pytest.mark.parametrize('tie_types', all_combos)
@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('sample_weight', [np.ones, sample_weights])
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
    I = X.T @ (H @ X)
    assert np.allclose(I, I.T)
    cov_ = np.linalg.inv(I)

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
    assert np.allclose(D_coxph[0], C.deviance - 2 * C.loglik_sat)
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
@pytest.mark.parametrize('sample_weight', [np.ones, sample_weights])
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
    D_R, G_R, H_R = get_glmnet_result(data['event'],
                                      data['status'],
                                      start,
                                      eta,
                                      weight)

    coxdev = CoxDeviance(event=data['event'],
                         start=start,
                         status=data['status'],
                         tie_breaking='breslow')
    C = coxdev(eta,
               weight)

    delta_D = np.fabs(D_R - C.deviance) / np.fabs(D_R)
    delta_G = np.linalg.norm(G_R - C.gradient) / np.linalg.norm(G_R)
    delta_H = np.linalg.norm(H_R - C.diag_hessian) / np.linalg.norm(H_R)

    assert (delta_D < tol) and (delta_G < tol) and (delta_H < tol)
    
