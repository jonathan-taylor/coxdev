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

from itertools import product, combinations

# basic model for times
rng = np.random.default_rng(0)
def sample(size=1):
    W = rng.uniform(size=size) + 0.5
    W[size//3] += 2
    W[size//3:size//2] += 1
    return W

# simulate different types of ties occurring
def simulate(start_count, event_count, size=1):
    size = rng.poisson(size) + 1
    if start_count == 0 and event_count == 0:
        return None
 # both of this well ensure there are unique starts or events 
    elif ((start_count == 0 and event_count == 1) or  
                                                      
          (start_count == 1 and event_count == 0)):
        start = sample(size=size)
        event = start + sample(size=size)
        
    # ties in event but not starts
    elif (start_count == 0) and (event_count == 2): 
        event = sample() * np.ones(size)
        start = event - sample(size=size)
        min_start = start.min()
        E = sample()
        event += min_start + E
        start += min_start + E
    # ties in starts but not events
    elif (start_count == 2) and (event_count == 0): 
        start = sample() * np.ones(size)
        event = start + sample(size=size)
    # single tie in start and event
    elif (start_count == 1) and (event_count == 1): 
        start = []
        event = []
        for _ in range(size):
            U = sample()
            start.extend([U-sample(), U])
            event.extend([U, U+sample()])
        start = np.asarray(start).reshape(-1)
        event = np.asarray(event).reshape(-1)
        
    # multiple starts at single event
    elif (start_count == 2) and (event_count == 1): 
        start = sample() * np.ones(size)
        event = start + sample(size=size)
        E = sample()
        event = np.hstack([event, start[0]])
        start = np.hstack([start, start[0] - sample()])

    # multiple events at single start
    elif (start_count == 1) and (event_count == 2):
        event = sample() * np.ones(size)
        start = event - sample(size=size)
        E = sample()
        start = np.hstack([start, event[0]])
        event = np.hstack([event, event[0] + sample()])

    # multiple events and starts
    elif (start_count == 2) and (event_count == 2): 
        U = sample()
        event = U * np.ones(size)
        start = event - sample(size=size)
        size2 = rng.poisson(size) + 1
        start2 = U * np.ones(size2)
        event2 = start2 + sample(size=size2)
        start = np.hstack([start, start2])
        event = np.hstack([event, event2])

    size = start.shape[0]
    status = rng.choice([0,1], size=size)
    return pd.DataFrame({'start':start, 'event':event, 'status':status})

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


def get_coxph_grad(event,
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
        rpy.r('F = coxph(y ~ X, init=beta, weights=sample_weight, control=coxph.control(iter.max=0), ties=ties)')
        rpy.r('score = colSums(coxph.detail(F)$scor)')
        G = rpy.r('score')
        D = rpy.r('F$loglik')

    return -2 * G, -2 * D


dataset_types = [(0,1), (1,0), (1, 1), (0, 2), (2, 0), (2, 1), (1, 2), (2, 2)]
all_combos = []
for i in range(1, 9):
    for v in combinations(dataset_types, i):
        all_combos.append(v)

@pytest.mark.parametrize('tie_types', all_combos)
@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'][1:])
@pytest.mark.parametrize('sample_weight', [np.ones, lambda n: sample(n)])
@pytest.mark.parametrize('have_start_times', [True, False])
def test_coxph(tie_types,
               tie_breaking,
               sample_weight,
               have_start_times,
               nrep=5,
               size=5,
               tol=1e-10):

    dfs = []
    for tie_type in tie_types:
        for _ in range(nrep):
            dfs.append(simulate(tie_type[0],
                                tie_type[1], size=size))
    data = pd.concat(dfs)
    
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
    (G_coxph,
     D_coxph) = get_coxph_grad(event=np.asarray(data['event']),
                               status=np.asarray(data['status']),
                               beta=beta,
                               sample_weight=weight,
                               start=start,
                               ties=tie_breaking,
                               X=X)

    assert np.allclose(D_coxph[0], C.deviance - 2 * C.loglik_sat)
    delta_ph = np.linalg.norm(G_coxph - X.T @ C.gradient) / np.linalg.norm(X.T @ C.gradient)
    assert delta_ph < tol

@pytest.mark.parametrize('tie_types', all_combos)
@pytest.mark.parametrize('sample_weight', [np.ones, lambda n: sample(n)])
@pytest.mark.parametrize('have_start_times', [True, False])
def test_glmnet(tie_types,
                sample_weight,
                have_start_times,
                nrep=5,
                size=5,
                tol=1e-10):

    dfs = []
    for tie_type in tie_types:
        for _ in range(nrep):
            dfs.append(simulate(tie_type[0],
                                tie_type[1], size=size))
    data = pd.concat(dfs)

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
    
