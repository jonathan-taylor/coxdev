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


tol=1e-10

rpy.r('d <- readRDS("./g0.RDS")')
event = np.asarray(rpy.r('d$event'))
start = None
status = np.asarray(rpy.r('d$status'))
weight = np.asarray(rpy.r('d$sample_weight'))
tie_breaking = 'efron'
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

print(D_coxph, C.deviance - 2 * C.loglik_sat)
assert np.allclose(D_coxph[0], C.deviance - 2 * C.loglik_sat)
delta_ph = np.linalg.norm(G_coxph - X.T @ C.gradient) / np.linalg.norm(X.T @ C.gradient)
assert delta_ph < tol
assert np.linalg.norm(cov_ - cov_coxph) / np.linalg.norm(cov_) < tol
