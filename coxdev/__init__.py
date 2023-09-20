from dataclasses import dataclass, InitVar
from typing import Literal

import numpy as np
import pandas as pd

@dataclass
class CoxDeviance(object):

    event: InitVar(np.ndarray)
    status: InitVar(np.ndarray)
    start: InitVar(np.ndarray)
    tie_breaking: Literal['efron', 'breslow'] = 'efron'
    
    def __post_init__(self,
                      event,
                      status,
                      start=None):

        event = np.asarray(event)
        status = np.asarray(status)
        nevent = event.shape[0]

        if start is None:
            start = -np.ones(nevent) * np.inf
            self._have_start_times = False
        else:
            self._have_start_times = True

        (self._preproc,
         self._event_order,
         self._start_order,
         self._loglik_sat) = _preprocess(start,
                                         event,
                                         status)
        self._efron = self.efron and np.linalg.norm(self._preproc['scaling']) > 0

    def __call__(self,
                 linear_predictor,
                 sample_weight):

        return _log_like(np.asarray(linear_predictor),
                         np.asarray(sample_weight),
                         self._event_order,
                         self._start_order,
                         self._preproc['status'],
                         self._preproc['event'],
                         self._preproc['start'],
                         self._preproc['first'],
                         self._preproc['last'],
                         self._preproc['scaling'],
                         self._preproc['event_map'],
                         self._preproc['start_map'],
                         self._loglik_sat,
                         efron=self._efron,
                         have_start_times=self._have_start_times)


def _preprocess(start,
                event,
                status):
    
    start = np.asarray(start)
    event = np.asarray(event)
    status = np.asarray(status)
    
    stacked_df = pd.DataFrame({'time':np.hstack([start, event]),
                               'status':np.hstack([np.zeros_like(start), 
                                                   status]),
                               'is_start':np.hstack([np.ones(n, int), np.zeros(n, int)]),
                               'index':np.hstack([np.arange(n), np.arange(n)])})

    sorted_df = stacked_df.sort_values(by=['time', 'status', 'is_start'], ascending=[True,False,True])
    sorted_df

    # do the joint sort

    event_count, start_count = 0, 0
    event_order, start_order = [], []
    start_map, event_map = [], []
    first = []
    event_idx = []
    last_row = None
    which_event = -1
    first_event = -1
    num_successive_event = 1
    ties = {}    
    for _r in range(sorted_df.shape[0]):
        row = sorted_df.iloc[_r]
        if row['is_start'] == 1: # a start time
            start_order.append(row['index'])
            start_map.append(event_count)
            start_count += 1
        else: # an event / stop time
            if row['status'] == 1:
                # if it's an event and the time is same as last row 
                # it is the same event
                # else it's the next "which_event"
                
                if (last_row is not None and 
                    row['time'] != last_row['time']): # index of next `status==1`
                    first_event += num_successive_event
                    num_successive_event = 1
                    which_event += 1
                else:
                    num_successive_event += 1
                    
                first.append(first_event)
            else:
                first_event += num_successive_event
                num_successive_event = 1
                first.append(first_event) # this event time was not an failure time

            event_map.append(start_count)
            event_order.append(row['index'])
            event_count += 1
        last_row = row

    first = np.array(first)
    start_order = np.array(start_order).astype(int)
    event_order = np.array(event_order).astype(int)
    start_map = np.array(start_map, int)
    event_map = np.array(event_map, int)

    # reset start_map to original order
    start_map_cp = start_map.copy()
    start_map[start_order] = start_map_cp

    preprocessed_df = pd.DataFrame({'status':status[event_order],
                                    'first':first,
                                    'start_map':start_map[event_order].astype(int), 
                                    'event_map':event_map.astype(int) # already in event order
                                    }, index=event_order)

    # compute `last`
    
    last = []
    last_event = n-1
    for i, f in enumerate(preprocessed_df['first'][::-1]):
        last.append(last_event)
        # immediately following a last event, `first` will agree with np.arange
        if f - (n - 1 - i) == 0:
            last_event = f - 1        

    preprocessed_df.insert(2, 'last', last[::-1])
    
    preproc = preprocessed_df # shorthand
    preproc['event'] = event[event_order]
    preproc['start'] = start[event_order]

    # compute the saturated log-likelihood

    sat_df = preproc[['first', 'status']] # recall, this is in event order
    sat_df['sample_weight'] = weight[event_order]
    loglik_sat = 0
    for _, df in sat_df[['first', 'status', 'sample_weight']].groupby('first'):
        if df['status'].sum() > 0:
            W = (df['sample_weight'] * df['status']).sum()
            loglik_sat -= W * np.log(W)

    # compute scaling vector for Efron's tie breaking method

    den = preproc['last'] + 1 - preproc['first']
    preproc['scaling'] = (np.arange(n) - preproc['first']) / den

    return preproc, event_order, start_order, loglik_sat

# Evaluation in `python` code that is similar to what the `C` code will look like.

def _cox_dev(eta,           # eta is in native order 
             sample_weight, # sample_weight is in native order
             event_order,   
             start_order,
             status,        # everything below in event order
             event,
             start,
             first,
             last,
             scaling,
             event_map,
             start_map,
             loglik_sat,
             have_start_times=True,
             efron=False):

    # be sure they're arrays so that no weird pandas indexing is used
    eta = np.asarray(eta)
    sample_weight = np.asarray(sample_weight)
    event_order = np.asarray(event_order)   
    start_order = np.asarray(start_order)
    status = np.asarray(status)
    event = np.asarray(event)
    start = np.asarray(start)
    first = np.asarray(first)
    last = np.asarray(last)
    scaling = np.asarray(scaling)
    event_map = np.asarray(event_map)
    start_map = np.asarray(start_map)
    
    _status = (status==1)
    
    eta = eta - eta.mean()
    
    # compute the event ordered reversed cumsum
    eta_event = eta[event_order]
    w_event = sample_weight[event_order]
    exp_eta_w_event = w_event * np.exp(eta_event)
    event_cumsum = np.hstack([np.cumsum(exp_eta_w_event[::-1])[::-1], 0]) # length=n+1 for when last=n-1

    # compute the start oredered reversed cumsum, if necessary
    # then compute the cumsums (or difference of cumsums) for Breslow approximation
    
    if have_start_times:
        exp_eta_w_start = np.hstack([(sample_weight * np.exp(eta))[start_order], 0]) # length=n+1
        start_cumsum = np.cumsum(exp_eta_w_start[::-1])[::-1]  # length=n+1
        risk_sums = event_cumsum[first] - start_cumsum[event_map]
    else:
        risk_sums = event_cumsum[first]
        
    # compute the Efron correction, adjusting risk_sum if necessary
    
    if efron == True:
        # XXXXX is last term handled correctly?
        n = eta.shape[0]
        num = (event_cumsum[first] - 
               event_cumsum[last+1])
        risk_sums -= num * scaling
    
    log_terms = np.log(np.array(risk_sums)) * w_event * _status
    loglik = (w_event * eta_event * _status).sum() - np.sum(log_terms)

    # cumsums for gradient and Hessian
    
    # length of cumsums is n+1
    # 0 is prepended for first(k)-1, start(k)-1 lookups
    # a 1 is added to all indices

    A_10 = _status * w_event / risk_sums
    C_10 = np.hstack([0, np.cumsum(A_10)]) 
    
    A_20 = _status * w_event / risk_sums**2
    C_20 = np.hstack([0, np.cumsum(A_20)]) # length=n+1

    # if there are no ties, scaling should be identically 0
    # don't bother with cumsums below 

    if not efron:
        if have_start_times:
            T_1_term = C_10[last+1] - C_10[start_map]   # +1 for start_map? depends on how  
                                                         # a tie between a start time and an event time
                                                         # if that means the start individual is excluded
                                                         # we should add +1, otherwise there should be
                                                         # no +1 in the [start_map+1] above
            T_2_term = C_20[last+1] - C_20[start_map]
        else:
            T_1_term = C_10[last+1]
            T_2_term = C_20[last+1]
    else:
        # compute the other necessary cumsums
        
        A_11 = _status * w_event * scaling / risk_sums
        C_11 = np.hstack([0, np.cumsum(A_11)]) # length=n+1

        A_21 = _status * w_event * scaling / risk_sums
        C_21 = np.hstack([0, np.cumsum(A_21)]) # length=n+1

        A_22 = _status * w_event * scaling / risk_sums
        C_22 = np.hstack([0, np.cumsum(A_22)]) # length=n+1

        T_1_term = (C_10[last+1] - 
                    (C_11[last+1] - C_11[first]))
        T_2_term = ((C_22[last+1] - C_22[first]) 
                    - 2 * (C_21[last+1] - C_21[first]) + 
                    C_20[last+1])
        if have_start_times:
            T_1_term -= C_10[start_map]
            T_2_term -= C_20[first]
    
    grad = w_event * _status - exp_eta_w_event * T_1_term
    grad_cp = grad.copy()
    grad[event_order] = grad_cp

    # now the diagonal of the Hessian

    diag_hess = exp_eta_w_event**2 * T_2_term - exp_eta_w_event * T_1_term
    diag_hess_cp = diag_hess.copy()
    diag_hess[event_order] = diag_hess_cp

    deviance = 2 * (loglik_sat - loglik)
    return deviance, -2 * grad, -2 * diag_hess

# eta = data_df['eta'] # in native order
# dev, G, H = log_like(eta,
#                      data_df['weight'],
#                      event_order,
#                      start_order,
#                      preproc['status'],
#                      preproc['event'],
#                      preproc['start'],
#                      preproc['first'],
#                      preproc['last'],
#                      preproc['scaling'],
#                      preproc['event_map'],
#                      preproc['start_map'],
#                      np.linalg.norm(preproc['scaling']),
#                      loglik_sat,
#                      efron=False)

# import rpy2
# # %load_ext rpy2.ipython
# start = data_df['start']
# event = data_df['event']
# status = data_df['status']
# weight = data_df['weight']
# # %R -i start,event,status,eta,weight

# # + magic_args="-o G_R,H_R,D_R" language="R"
# # library(survival)
# # library(glmnet)
# # Y = Surv(start, event, status)
# # D_R = glmnet:::coxnet.deviance3(pred=eta, y=Y, weight=weight, std.weights=FALSE)
# # # glmnet computes grad and hessian of the log-likelihood, not deviance
# # # need to multiply by -2 to get grad and hessian of deviance
# # G_R = glmnet:::coxgrad3(eta, Y, weight, std.weights=FALSE, diag.hessian=TRUE)
# # H_R = attr(G_R, 'diag_hessian')
# # G_R = -2 * G_R
# # H_R = -2 * H_R
# # -

# np.linalg.norm(G-G_R)/ np.linalg.norm(G)

# np.linalg.norm(H-H_R) / np.linalg.norm(H)

# np.fabs(dev - D_R)
