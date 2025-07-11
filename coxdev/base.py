"""
Cox Proportional Hazards Model Deviance Computation.

This module provides efficient computation of Cox model deviance, gradients,
and Hessian information matrices for survival analysis with support for
different tie-breaking methods (Efron and Breslow).
"""

from dataclasses import dataclass, InitVar
from typing import Literal, Optional
# for Hessian

from scipy.sparse.linalg import LinearOperator

from . import _version
__version__ = _version.get_versions()['version']

import numpy as np
from joblib import hash as _hash

from .coxc import (cox_dev as _cox_dev,
                   hessian_matvec as _hessian_matvec,
                   compute_sat_loglik as _compute_sat_loglik,
                   c_preprocess)

    
@dataclass
class CoxDevianceResult(object):
    """
    Result object containing Cox model deviance computation results.
    
    Attributes
    ----------
    linear_predictor : np.ndarray
        The linear predictor values (X @ beta) used in the computation.
    sample_weight : np.ndarray
        Sample weights used in the computation.
    loglik_sat : float
        Saturated log-likelihood value.
    deviance : float
        Computed deviance value.
    gradient : Optional[np.ndarray]
        Gradient of the deviance with respect to the linear predictor.
    diag_hessian : Optional[np.ndarray]
        Diagonal of the Hessian matrix.
    __hash_args__ : str
        Hash string for caching results.
    """

    linear_predictor: np.ndarray
    sample_weight: np.ndarray
    loglik_sat: float
    deviance: float
    gradient: Optional[np.ndarray]
    diag_hessian: Optional[np.ndarray]
    __hash_args__: str


@dataclass
class CoxDeviance(object):
    """
    Cox Proportional Hazards Model Deviance Calculator.
    
    This class provides efficient computation of Cox model deviance, gradients,
    and Hessian information matrices. It supports both Efron and Breslow
    tie-breaking methods and handles left-truncated survival data.
    
    Parameters
    ----------
    event : np.ndarray
        Event times (failure times) for each observation.
    status : np.ndarray
        Event indicators (1 for event occurred, 0 for censored).
    start : np.ndarray, optional
        Start times for left-truncated data. If None, assumes no truncation.
    tie_breaking : {'efron', 'breslow'}, default='efron'
        Method for handling tied event times.
        
    Attributes
    ----------
    tie_breaking : str
        The tie-breaking method being used.
    _have_start_times : bool
        Whether start times are provided.
    _efron : bool
        Whether Efron's method is being used for tie-breaking.

    Examples
    --------
    >>> import numpy as np
    >>> from coxdev import CoxDeviance
    >>> event = np.array([3, 6, 8, 4, 6, 4, 3, 2, 2, 5, 3, 4])
    >>> status = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1])
    >>> cox = CoxDeviance(event=event, status=status)
    >>> eta = np.linspace(-1, 1, len(event))
    >>> result = cox(eta)
    >>> print(round(result.deviance, 4))
    20.7998
    """
    
    event: InitVar[np.ndarray]
    status: InitVar[np.ndarray]
    start: InitVar[np.ndarray]=None
    tie_breaking: Literal['efron', 'breslow'] = 'efron'
    
    def __post_init__(self,
                      event,
                      status,
                      start=None):
        """
        Initialize the CoxDeviance object with survival data.
        
        Parameters
        ----------
        event : np.ndarray
            Event times for each observation.
        status : np.ndarray
            Event indicators (1 for event, 0 for censored).
        start : np.ndarray, optional
            Start times for left-truncated data.
        """
        event = np.asarray(event).astype(float)

        status_arr = np.asarray(status)
        if not set(np.unique(status_arr)).issubset(set([0,1])):
            raise ValueError('status must be binary')
        status = status_arr.astype(np.int32)
        
        nevent = event.shape[0]

        if start is None:
            start = -np.ones(nevent) * np.inf
            self._have_start_times = False
        else:
            start = np.asarray(start)
            self._have_start_times = True

        (self._preproc,
         self._event_order,
         self._start_order) = c_preprocess(start,
                                           event,
                                           status)
        self._event_order = self._event_order.astype(np.int32)
        self._start_order = self._start_order.astype(np.int32)
        
        self._efron = self.tie_breaking == 'efron' and np.linalg.norm(self._preproc['scaling']) > 0

        self._status = np.asarray(self._preproc['status'])
        self._event = np.asarray(self._preproc['event'])
        self._start = np.asarray(self._preproc['start'])
        self._first = np.asarray(self._preproc['first']).astype(np.int32)
        self._last = np.asarray(self._preproc['last']).astype(np.int32)
        self._scaling = np.asarray(self._preproc['scaling'])
        self._event_map = np.asarray(self._preproc['event_map']).astype(np.int32)
        self._start_map = np.asarray(self._preproc['start_map']).astype(np.int32)
        self._first_start = self._first[self._start_map]
        
        if not np.all(self._first_start == self._start_map):
            raise ValueError('first_start disagrees with start_map')

        n = self._status.shape[0]

        # allocate necessary memory
        
        self._T_1_term = np.zeros(n)
        self._T_2_term = np.zeros(n)
        self._event_reorder_buffers = [np.zeros(n) for i in range(3)]
        self._forward_cumsum_buffers = [np.zeros(n+1) for i in range(5)]
        self._forward_scratch_buffer = np.zeros(n)
        self._reverse_cumsum_buffers = [np.zeros(n+1) for i in range(4)]
        self._risk_sum_buffers = [np.zeros(n) for i in range(2)]
        self._hess_matvec_buffer = np.zeros(n)
        self._grad_buffer = np.zeros(n)
        self._diag_hessian_buffer = np.zeros(n)
        self._diag_part_buffer = np.zeros(n)
        self._w_avg_buffer = np.zeros(n)
        self._exp_w_buffer = np.zeros(n)

    def __call__(self,
                 linear_predictor,
                 sample_weight=None):
        """
        Compute Cox model deviance and related quantities.
        
        Parameters
        ----------
        linear_predictor : np.ndarray
            Linear predictor values (X @ beta).
        sample_weight : np.ndarray, optional
            Sample weights. If None, uses equal weights.
            
        Returns
        -------
        CoxDevianceResult
            Object containing deviance, gradient, and Hessian diagonal.
        """
        if sample_weight is None:
            sample_weight = np.ones_like(linear_predictor)
        else:
            sample_weight = np.asarray(sample_weight)

        linear_predictor = np.asarray(linear_predictor)
            
        cur_hash = _hash([linear_predictor, sample_weight])
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
            self._exp_w_buffer[:] = sample_weight * np.exp(np.clip(eta, -np.inf, 30))




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
                                self._have_start_times,
                                self._efron)
                                
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
        """
        Compute the information matrix (negative Hessian) as a linear operator.
        
        Parameters
        ----------
        linear_predictor : np.ndarray
            Linear predictor values (X @ beta).
        sample_weight : np.ndarray, optional
            Sample weights. If None, uses equal weights.
            
        Returns
        -------
        CoxInformation
            Linear operator representing the information matrix.
        """
        result = self(linear_predictor,
                      sample_weight)
        return CoxInformation(result=result,
                              coxdev=self)

@dataclass
class CoxInformation(LinearOperator):
    """
    Linear operator representing the Cox model information matrix.
    
    This class provides matrix-vector multiplication with the information
    matrix (negative Hessian) of the Cox model, allowing efficient computation
    without explicitly forming the full matrix.
    
    Parameters
    ----------
    coxdev : CoxDeviance
        The CoxDeviance object used for computations.
    result : CoxDevianceResult
        Result from the most recent deviance computation.
        
    Attributes
    ----------
    shape : tuple
        Shape of the information matrix (n, n).
    dtype : type
        Data type of the matrix elements.
    """

    coxdev: CoxDeviance
    result: CoxDevianceResult

    def __post_init__(self):
        """Initialize the linear operator dimensions."""
        n = self.coxdev._status.shape[0]
        self.shape = (n, n)
        self.dtype = float
        
    def _matvec(self, arg):
        """
        Compute matrix-vector product with the information matrix.
        
        Parameters
        ----------
        arg : np.ndarray
            Vector to multiply with the information matrix.
            
        Returns
        -------
        np.ndarray
            Result of the matrix-vector multiplication.
        """
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
                        coxdev._have_start_times,                        
                        coxdev._efron)

        return coxdev._hess_matvec_buffer.copy()

    
    def _adjoint(self, arg):
        """
        Compute the adjoint (transpose) matrix-vector product.
        
        Since the information matrix is symmetric, this is the same as _matvec.
        
        Parameters
        ----------
        arg : np.ndarray
            Vector to multiply with the adjoint matrix.
            
        Returns
        -------
        np.ndarray
            Result of the adjoint matrix-vector multiplication.
        """
        # it is symmetric
        return self._matvec(arg)


# private functions

def _preprocess(start,
                event,
                status):
    """
    Preprocess survival data for Cox model computations.
    
    This function handles data preprocessing including sorting, tie detection,
    and creation of indexing arrays for efficient computation.
    
    Parameters
    ----------
    start : np.ndarray
        Start times for left-truncated data.
    event : np.ndarray
        Event times.
    status : np.ndarray
        Event indicators.
        
    Returns
    -------
    tuple
        Preprocessed data structures for efficient computation.
    """
    return c_preprocess(start, event, status)

