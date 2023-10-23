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
                   _preprocess,
                   _compute_sat_loglik)

@dataclass
class CoxDevianceResult(object):

    linear_predictor: np.ndarray
    sample_weight: np.ndarray
    loglik_sat: float
    deviance: float
    gradient: Optional[np.ndarray]
    diag_hessian: Optional[np.ndarray]
    risk_sums: Optional[np.ndarray]
    diag_part: Optional[np.ndarray]
    w_avg: Optional[np.ndarray]
    exp_w: Optional[np.ndarray]
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
                                             self._status) 

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
                               self._first_start,
                               loglik_sat,
                               efron=self._efron,
                               have_start_times=self._have_start_times,
                               asarray=False)
            self._result = CoxDevianceResult(*((linear_predictor,
                                                sample_weight) +
                                               _result + (cur_hash,)))
            
        return self._result

    def information(self,
                    X,
                    beta,
                    sample_weight=None):

        linear_predictor = X @ beta

        result = self(linear_predictor,
                      sample_weight)
        H = CoxInformation(result=result,
                           coxdev=self)

        return X.T @ (H @ X)

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

        return _hessian_matvec(-np.asarray(arg).reshape(-1),
                               np.asarray(result.linear_predictor),
                               np.asarray(result.sample_weight),
                               result.risk_sums,
                               result.diag_part,
                               result.w_avg,
                               result.exp_w,
                               coxdev._event_order,
                               coxdev._start_order,
                               coxdev._status,
                               coxdev._event,
                               coxdev._start,
                               coxdev._first,
                               coxdev._last,
                               coxdev._scaling,
                               coxdev._event_map,
                               coxdev._start_map,
                               coxdev._first_start,
                               efron=coxdev._efron,
                               have_start_times=coxdev._have_start_times,
                               asarray=False)                        

    def _adjoint(self, arg):
        # it is symmetric
        return self._matvec(arg)


