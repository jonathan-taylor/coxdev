from dataclasses import dataclass, InitVar
from typing import Literal, Optional

from . import _version
__version__ = _version.get_versions()['version']

import numpy as np
from joblib import hash

@dataclass
class CoxDevianceResult(object):

    loglik_sat: float
    deviance: float
    gradient: Optional[np.ndarray]
    diag_hessian: Optional[np.ndarray]
    __hash_args__: str

@dataclass
class CoxDeviance(object):

    event: InitVar(np.ndarray)
    status: InitVar(np.ndarray)
    start: InitVar(np.ndarray)=None
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
         self._start_order) = _preprocess(start,
                                         event,
                                         status)
        self._efron = self.tie_breaking == 'efron' and np.linalg.norm(self._preproc['scaling']) > 0

        self._status = np.asarray(self._preproc['status'])
        self._event = np.asarray(self._preproc['event'])
        self._start = np.asarray(self._preproc['start'])
        self._first = np.asarray(self._preproc['first'])
        self._last = np.asarray(self._preproc['last'])
        self._scaling = np.asarray(self._preproc['scaling'])
        self._event_map = np.asarray(self._preproc['event_map'])
        self._start_map = np.asarray(self._preproc['start_map'])

    def __call__(self,
                 linear_predictor,
                 sample_weight=None):

        if sample_weight is None:
            sample_weight = np.ones_like(linear_predictor)

        cur_hash = hash([linear_predictor, sample_weight])
        if not hasattr(self, "_result") or self._result.__hash_args__ != cur_hash:

            # compute the saturated log-likelihood

            _last, _first = self._preproc['last'], self._preproc['first']
            sample_weight = np.asarray(sample_weight)
            W_status = np.cumsum(np.hstack([0, sample_weight[self._event_order] * self._preproc['status']]))
            sums = W_status[_last+1] - W_status[_first]
            loglik_sat = 0
            prev_first = -1
            for f, s in zip(_first, sums):
                if s > 0 and f != prev_first:
                    loglik_sat -= s * np.log(s)
                prev_first = f
            
            _result = _cox_dev(np.asarray(linear_predictor),
                               np.asarray(sample_weight),
                               self._event_order,
                               self._start_order,
                               self._status,
                               self._event,
                               self._start,
                               self._first,
                               self._last,
                               self._scaling,
                               self._event_map,
                               self._start_map,
                               loglik_sat,
                               efron=self._efron,
                               have_start_times=self._have_start_times,
                               asarray=False)
            self._result = CoxDevianceResult(*(_result + (cur_hash,)))
            
        return self._result

def _preprocess(start,
                event,
                status):
    
    start = np.asarray(start)
    event = np.asarray(event)
    status = np.asarray(status)
    nevent = status.shape[0]
    
    # second column of stacked_array is 1-status...
    stacked_time = np.hstack([start, event])
    stacked_status_c = np.hstack([np.ones(nevent, int), 1-status]) # complement of status
    stacked_is_start = np.hstack([np.ones(nevent, int), np.zeros(nevent, int)])
    stacked_index = np.hstack([np.arange(nevent), np.arange(nevent)])

    argsort = np.lexsort((stacked_is_start,
                          stacked_status_c,
                          stacked_time))
    sorted_time = stacked_time[argsort]
    sorted_status = 1 - stacked_status_c[argsort]
    sorted_is_start = stacked_is_start[argsort]
    sorted_index = stacked_index[argsort]
    
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
    for row in zip(sorted_time,
                   sorted_status,
                   sorted_is_start,
                   sorted_index):
        (_time, _status, _is_start, _index) = row
        if _is_start == 1: # a start time
            start_order.append(_index)
            start_map.append(event_count)
            start_count += 1
        else: # an event / stop time
            if _status == 1:
                # if it's an event and the time is same as last row 
                # it is the same event
                # else it's the next "which_event"
                
                if (last_row is not None and 
                    _time != last_row[0]): # index of next `status==1`
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
            event_order.append(_index)
            event_count += 1
        last_row = row

    first = np.array(first)
    start_order = np.array(start_order, int)
    event_order = np.array(event_order, int)
    start_map = np.array(start_map, int)
    event_map = np.array(event_map, int)

    # reset start_map to original order
    start_map_cp = start_map.copy()
    start_map[start_order] = start_map_cp

    # set to event order

    _status = status[event_order]
    _first = first
    _start_map = start_map[event_order]
    _event_map = event_map

    # compute `last`
    
    last = []
    last_event = nevent-1
    for i, f in enumerate(_first[::-1]):
        last.append(last_event)
        # immediately following a last event, `first` will agree with np.arange
        if f - (nevent - 1 - i) == 0:
            last_event = f - 1        
    _last = np.array(last[::-1])
    _event = event[event_order]
    _start = event[start_order]

    den = _last + 1 - _first

    _scaling = (np.arange(nevent) - _first) / den

    preproc = {'start':_start,
               'event':_event,
               'first':_first,
               'last':_last,
               'scaling':_scaling,
               'start_map':_start_map,
               'event_map':_event_map,
               'status':_status}

    return preproc, event_order, start_order

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
             efron=False,
             asarray=True):

    # be sure they're arrays so that no weird pandas indexing is used

    if asarray:
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
    return loglik_sat, deviance, -2 * grad, -2 * diag_hess
