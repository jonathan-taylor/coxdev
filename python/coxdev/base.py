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

import numpy as np
from joblib import hash as _hash

from .coxc import CoxDeviance as _CoxDeviance

    
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
    sample_weight : np.ndarray, optional
        If None, defaults to one. Necessary at construction only to determine
        which entries are 0.
    
    tie_breaking : {'efron', 'breslow'}, default='efron'
        Method for handling tied event times.
        
    Attributes
    ----------
    tie_breaking : str
        The tie-breaking method being used.

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
    sample_weight: InitVar[np.ndarray]=None
    tie_breaking: Literal['efron', 'breslow'] = 'efron'
    
    def __post_init__(self,
                      event,
                      status,
                      start=None,
                      sample_weight=None):
        """
        Initialize the CoxDeviance object with survival data.
        """
        event = np.asarray(event).astype(float)
        status = np.asarray(status)
        if not set(np.unique(status)).issubset({0, 1}):
            raise ValueError('status must be binary')
        status = status.astype(np.int32)
            
        if start is None:
            start = -np.ones_like(event) * np.inf
        else:
            start = np.asarray(start).astype(float)
            
        if sample_weight is None:
            sample_weight = np.ones_like(event).astype(float)
        else:
            sample_weight = np.asarray(sample_weight).astype(float)

        self._raw_event = event
        self._raw_status = status
        self._raw_start = start
        
        self._coxc = _CoxDeviance(self._raw_start, 
                                  self._raw_event, 
                                  self._raw_status, 
                                  sample_weight, 
                                  self.tie_breaking == 'efron')
        self._weights_hash = _hash(sample_weight)

    @property
    def event_order(self):
        return self._coxc.event_order
        
    @property
    def start_order(self):
        return self._coxc.start_order
        
    @property
    def _first(self):
        return self._coxc.first
        
    @property
    def _last(self):
        return self._coxc.last
        
    @property
    def _start_map(self):
        return self._coxc.start_map
        
    @property
    def _event_map(self):
        return self._coxc.event_map
        
    @property
    def _scaling(self):
        return self._coxc.scaling
        
    @property
    def _status(self):
        return self._coxc.status
        
    @property
    def _event(self):
        return self._coxc.event
        
    @property
    def _start(self):
        return self._coxc.start

    @property
    def _first_start(self):
        return self._first[self._start_map]

    @property
    def _event_order(self):
        return self.event_order

    @property
    def _preproc(self):
        return {
            'last': self._last,
            'start_map': self._start_map,
            'first': self._first,
            'event_map': self._event_map,
            'status': self._status,
            'event': self._event,
            'start': self._start,
            'scaling': self._scaling
        }

    def __call__(self,
                 linear_predictor,
                 sample_weight=None):
        """
        Compute Cox model deviance and related quantities.
        """
        if sample_weight is None:
            sample_weight = np.ones_like(linear_predictor, float)
        else:
            sample_weight = np.asarray(sample_weight).astype(float)

        linear_predictor = np.asarray(linear_predictor).astype(float)
            
        cur_weight_hash = _hash(sample_weight)
        if getattr(self, "_weights_hash", None) != cur_weight_hash:
            self._coxc = _CoxDeviance(self._raw_start, 
                                      self._raw_event, 
                                      self._raw_status, 
                                      sample_weight, 
                                      self.tie_breaking == 'efron')
            self._weights_hash = cur_weight_hash

        cur_hash = _hash([linear_predictor, sample_weight])

        if not hasattr(self, "_result") or self._result.__hash_args__ != cur_hash:
            deviance = self._coxc.compute_deviance(linear_predictor, sample_weight)
            
            # These are now references to internal Eigen buffers
            self._result = CoxDevianceResult(linear_predictor=self._coxc.linear_predictor,
                                             sample_weight=self._coxc.sample_weight,
                                             loglik_sat=self._coxc.loglik_sat,
                                             deviance=deviance,
                                             gradient=self._coxc.gradient.copy(),
                                             diag_hessian=self._coxc.diag_hessian.copy(),
                                             __hash_args__=cur_hash)
            
        return self._result

    def information(self,
                    linear_predictor,
                    sample_weight=None):
        """
        Compute the information matrix (negative Hessian) as a linear operator.
        """
        result = self(linear_predictor,
                      sample_weight)
        return CoxInformation(result=result,
                              coxdev=self)


@dataclass
class CoxInformation(LinearOperator):
    """
    Linear operator representing the Cox model information matrix.
    """

    coxdev: CoxDeviance
    result: CoxDevianceResult

    def __post_init__(self):
        """Initialize the linear operator dimensions."""
        n = self.result.linear_predictor.shape[0]
        self.shape = (n, n)
        self.dtype = float
        
    def _matvec(self, arg):
        """
        Compute matrix-vector product with the information matrix.
        """
        # arg is in native order
        return self.coxdev._coxc.compute_hessian_matvec(np.asarray(arg).astype(float))

    def _adjoint(self, arg):
        """
        Compute the adjoint (transpose) matrix-vector product.
        """
        return self._matvec(arg)

    
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


