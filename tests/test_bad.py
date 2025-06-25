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

def check_results(fname, ties):
    tol = 1e-10
    rpy.r('d <- readRDS("./g0.RDS")')
    event = np.asarray(rpy.r('d$event'))
    start = None
    status = np.asarray(rpy.r('d$status'))
    weight = np.asarray(rpy.r('d$sample_weight'))
    tie_breaking = ties
    beta = np.asarray(rpy.r('d$beta'))
    X = np.array(rpy.r('d$X'))
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

check_results("g0.RDS", 'efron')
check_results("g0.RDS", 'breslow')
