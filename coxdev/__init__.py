from dataclasses import dataclass, InitVar
from typing import Literal, Optional

# for Hessian

from scipy.sparse.linalg import LinearOperator

from . import _version
__version__ = _version.get_versions()['version']

import numpy as np
from joblib import hash

from .base import (_cox_dev,
                   _hessian_matvec,
                   _compute_sat_loglik)

@dataclass
class CoxDevianceResult(object):

    linear_predictor: np.ndarray
    sample_weight: np.ndarray
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
        self._first_start = self._first[self._start_map]
        
        if not np.all(self._first_start == self._start_map):
            raise ValueError('first_start disagrees with start_map')

        n = self._status.shape[0]

        # allocate necessary memory
        
        self._T_1_term = np.zeros(n)
        self._T_2_term = np.zeros(n)
        self._event_reorder_buffers = np.zeros((3, n))
        self._forward_cumsum_buffers = np.zeros((5, n+1))
        self._forward_scratch_buffer = np.zeros(n)
        self._reverse_cumsum_buffers = np.zeros((4, n+1))
        self._risk_sum_buffers = np.zeros((2, n))
        self._hess_matvec_buffer = np.zeros(n)
        self._grad_buffer = np.zeros(n)
        self._diag_hessian_buffer = np.zeros(n)
        self._diag_part_buffer = np.zeros(n)
        self._w_avg_buffer = np.zeros(n)
        self._exp_w_buffer = np.zeros(n)

    def __call__(self,
                 linear_predictor,
                 sample_weight=None):

        if sample_weight is None:
            sample_weight = np.ones_like(linear_predictor)
        else:
            sample_weight = np.asarray(sample_weight)

        linear_predictor = np.asarray(linear_predictor)
            
        cur_hash = hash([linear_predictor, sample_weight])
        if not hasattr(self, "_result") or self._result.__hash_args__ != cur_hash:

            loglik_sat = _compute_sat_loglik(self._first,
                                             self._last,
                                             sample_weight, # in natural order
                                             self._event_order,
                                             self._status,
                                             self._forward_cumsum_buffers[0]) 

            eta = np.asarray(linear_predictor)
            sample_weight = np.asarray(sample_weight)
            eta = eta - eta.mean()
            self._exp_w_buffer[:] = sample_weight * np.exp(eta)

            deviance = _cox_dev(eta,
                                sample_weight,
                                self._exp_w_buffer,
                                self._event_order,
                                self._start_order,
                                self._status,
                                self._first,
                                self._last,
                                self._scaling,
                                self._event_map,
                                self._start_map,
                                loglik_sat,
                                self._T_1_term,
                                self._T_2_term,
                                self._grad_buffer,
                                self._diag_hessian_buffer,
                                self._diag_part_buffer,
                                self._w_avg_buffer,
                                self._event_reorder_buffers,
                                self._risk_sum_buffers, #[0] is for coxdev, [1] is for hessian...
                                self._forward_cumsum_buffers,
                                self._forward_scratch_buffer,
                                self._reverse_cumsum_buffers, #[0:2] are for risk sums, [2:4] used for hessian risk*arg sums
                                efron=self._efron,
                                have_start_times=self._have_start_times)

            # shorthand, for reference in hessian_matvec
            self._event_cumsum = self._reverse_cumsum_buffers[0]
            self._start_cumsum = self._reverse_cumsum_buffers[1]

            self._result = CoxDevianceResult(linear_predictor=linear_predictor,
                                             sample_weight=sample_weight,
                                             loglik_sat=loglik_sat,
                                             deviance=deviance,
                                             gradient=self._grad_buffer.copy(),
                                             diag_hessian=self._diag_hessian_buffer.copy(),
                                             __hash_args__=cur_hash)

        return self._result

    def information(self,
                    linear_predictor,
                    sample_weight=None):

        result = self(linear_predictor,
                      sample_weight)
        return CoxInformation(result=result,
                              coxdev=self)

@dataclass
class CoxInformation(LinearOperator):

    coxdev: CoxDeviance
    result: CoxDevianceResult

    def __post_init__(self):
        n = self.coxdev._status.shape[0]
        self.shape = (n, n)
        self.dtype = float
        
    def _matvec(self, arg):

        # this will compute risk sums if not already computed
        # at this linear_predictor and sample_weight
        
        result = self.result
        coxdev = self.coxdev

        # negative will give 2nd derivative of negative
        # loglikelihood

        _hessian_matvec(-np.asarray(arg).reshape(-1),
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
                        efron=coxdev._efron,
                        have_start_times=coxdev._have_start_times)
        return coxdev._hess_matvec_buffer.copy()

    
    def _adjoint(self, arg):
        # it is symmetric
        return self._matvec(arg)


# private functions

def _preprocess(start,
                event,
                status):
    """
    Compute various functions of the start / event / status
    to be used to help in computing cumsums

    This can probably stay in python, and have a separate
    implementation in R
    """
    
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

    _event = event[event_order]
    _start = event[start_order]

    # compute `last`
    
    last = []
    last_event = nevent-1
    for i, f in enumerate(_first[::-1]):
        last.append(last_event)
        # immediately following a last event, `first` will agree with np.arange
        if f - (nevent - 1 - i) == 0:
            last_event = f - 1        
    _last = np.array(last[::-1])

    den = _last + 1 - _first

    # XXXX
    _scaling = (np.arange(nevent) - _first) / den
    
    preproc = {'start':np.asarray(_start),
               'event':np.asarray(_event),
               'first':np.asarray(_first),
               'last':np.asarray(_last),
               'scaling':np.asarray(_scaling),
               'start_map':np.asarray(_start_map),
               'event_map':np.asarray(_event_map),
               'status':np.asarray(_status)}

    return preproc, event_order, start_order

