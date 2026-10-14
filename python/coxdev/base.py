"""
Cox Proportional Hazards Model Deviance Computation.

This module provides efficient computation of Cox model deviance, gradients,
and Hessian information matrices for survival analysis with support for
different tie-breaking methods (Efron and Breslow) and stratified models.
"""

from dataclasses import dataclass, InitVar
from typing import Literal, Optional

from scipy.sparse.linalg import LinearOperator

import numpy as np
from joblib import hash as _hash

from .coxc import CoxDeviance as _CoxDeviance
from .coxc import StratifiedCoxDeviance as _StratifiedCoxDeviance

    
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
    tie-breaking methods, handles left-truncated survival data, and supports
    stratified models.
    
    Parameters
    ----------
    event : np.ndarray
        Event times (failure times) for each observation.
    status : np.ndarray
        Event indicators (1 for event occurred, 0 for censored).
    strata : np.ndarray, optional
        Stratum labels for each observation. If None, computes an unstratified model.
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
    strata: InitVar[Optional[np.ndarray]] = None
    start: InitVar[Optional[np.ndarray]] = None
    sample_weight: InitVar[Optional[np.ndarray]] = None
    tie_breaking: Literal['efron', 'breslow'] = 'efron'
    
    def __post_init__(self,
                      event,
                      status,
                      strata=None,
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

        if strata is not None:
            strata = np.asarray(strata)
            if not np.issubdtype(strata.dtype, np.integer):
                raise ValueError(f"strata must be integer type, got {strata.dtype}")
            strata = np.asarray(strata, np.int32)

        if strata is not None and ((status.shape != event.shape) or (strata.shape != event.shape) or (start.shape != event.shape)):
            raise ValueError("status, event, start and strata must have same shape")

        self._raw_event = event
        self._raw_status = status
        self._raw_start = start
        self._strata = strata
        
        self._reconstruct_coxc(sample_weight)
        self._weights_hash = _hash(sample_weight)

    def _reconstruct_coxc(self, sample_weight):
        if self._strata is None:
            self._coxc = _CoxDeviance(self._raw_start, 
                                      self._raw_event, 
                                      self._raw_status, 
                                      sample_weight, 
                                      self.tie_breaking == 'efron')
        else:
            self._coxc = _StratifiedCoxDeviance(self._raw_start, 
                                                self._raw_event, 
                                                self._raw_status,
                                                self._strata,
                                                sample_weight, 
                                                self.tie_breaking == 'efron')

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
            self._reconstruct_coxc(sample_weight)
            self._weights_hash = cur_weight_hash

        cur_hash = _hash([linear_predictor, sample_weight])

        if not hasattr(self, "_result") or getattr(self._result, "__hash_args__", None) != cur_hash:
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

    def sample_score(self,
                       linear_predictor,
                       sample_weight=None,
                       method='multinomial',
                       n_draws=None,
                       seed=None):
        """
        Draws of the score at ``linear_predictor``: from the Cox model with
        that linear predictor, or (``'wild'``) from its score residuals.

        Each draw ``r`` has one entry per observation, on the scale of the
        score ``-gradient / 2``, so that ``X.T @ r`` is a draw of the score of
        the coefficients of a design ``X``.

        Parameters
        ----------
        linear_predictor : np.ndarray
            The linear predictor, e.g. of a fitted model.
        sample_weight : np.ndarray, optional
            Defaults to one. The model-based methods (all but ``'wild'``)
            need equal positive weights.
        method : {'multinomial', 'multinomial_centered', 'bernoulli', 'wallenius', 'wild'}
            ``'multinomial'``: at each event time, the failures are drawn with
            replacement from the risk set with probabilities proportional to
            ``sample_weight * exp(linear_predictor)``, and the draw is the
            difference of two such draws over ``sqrt(2)``; its covariance is the
            Breslow information. ``'multinomial_centered'``: one such draw minus
            its expectation. ``'bernoulli'``: each subject at risk fails
            independently with probability ``min(1, d p)``, centered within the
            risk set, two draws. ``'wallenius'``: the failures drawn without
            replacement, one at a time, two draws. ``'wild'``: the score
            residuals multiplied by independent standard normals; its
            covariance is the sandwich, the sum of the outer products of the
            score residuals.
        n_draws : int, optional
            The number of draws. If None, one draw, as a vector.
        seed : int, optional
            Seed of the random number generator (a nonnegative integer below
            2**64). The same seed gives the same draws on every platform, and
            from R. If None, a seed is drawn from numpy's random entropy.

        Returns
        -------
        np.ndarray
            ``(n,)`` if ``n_draws`` is None, else ``(n, n_draws)``.
        """
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1, np.uint64)[0])
        self(linear_predictor, sample_weight)
        draws = self._coxc.sample_score(method, 1 if n_draws is None else int(n_draws), int(seed))
        return draws[:, 0] if n_draws is None else draws

    def wild_score(self,
                   linear_predictor,
                   multipliers,
                   sample_weight=None):
        """
        The wild draw of the score for given multipliers: ``r`` with
        ``X.T @ r = sum_i multipliers[i] * U_i``, where ``U_i`` is the score
        residual of observation ``i`` (its contribution to the score, centered
        within each risk set, with Efron's weights if ``tie_breaking='efron'``).
        With ``multipliers`` all one, ``r`` is the score ``-gradient / 2``.
        """
        self(linear_predictor, sample_weight)
        return self._coxc.wild_score(np.asarray(multipliers, float))


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
        # sets shape and dtype (and, in SciPy >= 1.18, the array namespace used by @)
        LinearOperator.__init__(self, dtype=float, shape=(n, n))
        
    def _matvec(self, arg):
        """
        Compute matrix-vector product with the information matrix.
        """
        # arg is in native order
        arg_arr = np.asarray(arg).reshape(-1).astype(float)
        return self.coxdev._coxc.compute_hessian_matvec(arg_arr)

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
