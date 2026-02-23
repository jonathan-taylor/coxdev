import numpy as np
from dataclasses import dataclass, InitVar
from typing import Optional, Literal
from scipy.sparse.linalg import LinearOperator
from joblib import hash as _hash

from .base import (CoxDevianceResult,
                   CoxInformation)
from .coxc import StratifiedCoxDeviance as _StratifiedCoxDeviance

@dataclass
class StratifiedCoxDeviance:

    """
    Stratified Cox Proportional Hazards Model Deviance Calculator.

    Efficiently computes deviance, gradient, and block-diagonal information matrix for the
    stratified Cox model, supporting Efron and Breslow tie-breaking and left truncation.

    Parameters
    ----------
    event : np.ndarray
        Event (failure) times.
    status : np.ndarray
        Event indicators (1=event, 0=censored).
    strata : np.ndarray, optional
        Stratum labels for each observation.
    start : np.ndarray, optional
        Start times for left-truncated data.
    tie_breaking : {'efron', 'breslow'}, default='efron'
        Tie-breaking method.

    Examples
    --------
    >>> import numpy as np
    >>> from coxdev import StratifiedCoxDeviance
    >>> event = np.array([3, 6, 8, 4, 6, 4, 3, 2, 2, 5, 3, 4])
    >>> status = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1])
    >>> strata = np.repeat([0, 1, 2], 4)
    >>> cox = StratifiedCoxDeviance(event=event, status=status, strata=strata)
    >>> eta = np.linspace(-1, 1, len(event))
    >>> result = cox(eta)
    >>> print(round(result.deviance, 4))
    14.2741
    """

    event: InitVar[np.ndarray]
    status: InitVar[np.ndarray]
    strata: InitVar[Optional[np.ndarray]] = None
    start: InitVar[Optional[np.ndarray]] = None
    tie_breaking: Literal['efron', 'breslow'] = 'efron'

    def __post_init__(self,
                      event,
                      status,
                      strata,
                      start=None,
                      tie_breaking='efron'):
        """
        Initialize the CoxDeviance object with survival data.
        """
        event = np.asarray(event).astype(float)

        status = np.asarray(status)
        if not set(np.unique(status)).issubset(set([0,1])):
            raise ValueError('status must be binary')
        self._status = status.astype(np.int32)
        self._event = event
        nevent = event.shape[0]

        if start is None:
            start = -np.ones(nevent) * np.inf
            self._have_start_times = False
        else:
            start = np.asarray(start, float)
            self._have_start_times = True

        self._start = start
        if strata is None:
            strata = np.zeros(event.shape[0], np.int32)
        else:
            strata = np.asarray(strata)
        if not np.issubdtype(strata.dtype, np.integer):
            raise ValueError(f"strata must be integer type, got {strata.dtype}")
        self._strata = np.asarray(strata, np.int32)
        
        if ((status.shape != event.shape)
            or (strata.shape != event.shape)
            or (start.shape != event.shape)):
            raise ValueError("status, event, start and strata must have same shape")

        # Initialize result cache
        self._result = None
        self._weights_hash = None
        self._coxc = None

    def __call__(self, linear_predictor, sample_weight=None):
        linear_predictor = np.asarray(linear_predictor).astype(float)
        if sample_weight is None:
            sample_weight = np.ones_like(linear_predictor).astype(float)
        else:
            sample_weight = np.asarray(sample_weight).astype(float)

        cur_weight_hash = _hash(sample_weight)
        if self._weights_hash != cur_weight_hash:
            self._coxc = _StratifiedCoxDeviance(
                self._start,
                self._event,
                self._status,
                self._strata,
                sample_weight,
                self.tie_breaking == 'efron'
            )
            self._weights_hash = cur_weight_hash
            
        cur_hash = _hash([linear_predictor, sample_weight])

        if not hasattr(self, "_result") or getattr(self._result, "__hash_args__", None) != cur_hash:
            deviance = self._coxc.compute_deviance(linear_predictor, sample_weight)

            self._result = CoxDevianceResult(
                linear_predictor=self._coxc.linear_predictor,
                sample_weight=self._coxc.sample_weight,
                loglik_sat=self._coxc.loglik_sat, # Use loglik_sat from C++
                deviance=deviance,
                gradient=self._coxc.gradient.copy(),
                diag_hessian=self._coxc.diag_hessian.copy(),
                __hash_args__=cur_hash
            )

        return self._result

    def information(self, linear_predictor, sample_weight=None):
        """Return a block-diagonal LinearOperator representing the information matrix."""
        result = self(linear_predictor, sample_weight)
        return StratifiedCoxInformation(self, result)

@dataclass
class StratifiedCoxInformation(LinearOperator):

    strat_cox: StratifiedCoxDeviance
    result: CoxDevianceResult

    def __post_init__(self):
        self.n = self.result.linear_predictor.shape[0]
        self.shape = (self.n, self.n)
        self.dtype = float

    def _matvec(self, v):
        v = np.asarray(v).reshape(-1).astype(float)
        # arg is in native order
        return self.strat_cox._coxc.compute_hessian_matvec(v)

    def _adjoint(self, v):
        return self._matvec(v)


