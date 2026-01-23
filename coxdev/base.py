"""
Cox Proportional Hazards Model Deviance Computation.

This module provides efficient computation of Cox model deviance, gradients,
and Hessian information matrices for survival analysis with support for
different tie-breaking methods (Efron and Breslow).
"""

from dataclasses import dataclass, InitVar
from typing import Literal, Optional
from scipy.sparse.linalg import LinearOperator

from . import _version
__version__ = _version.get_versions()['version']

import numpy as np

from .coxc import (StratifiedCoxDevianceCpp as _StratifiedCoxDevianceCpp,
                   CoxIRLSStateCpp as _CoxIRLSStateCpp,
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


class CoxDeviance:
    """
    Cox Proportional Hazards Model Deviance Calculator.

    This class provides efficient computation of Cox model deviance, gradients,
    and Hessian information matrices. It supports both Efron and Breslow
    tie-breaking methods and handles left-truncated survival data.

    Internally uses the stratified C++ implementation with a single stratum
    for unified code path and consistent behavior.

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

    def __init__(self,
                 event: np.ndarray,
                 status: np.ndarray,
                 start: np.ndarray = None,
                 tie_breaking: Literal['efron', 'breslow'] = 'efron'):
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
        tie_breaking : {'efron', 'breslow'}, default='efron'
            Method for handling tied event times.
        """
        event = np.asarray(event).astype(float)

        status_arr = np.asarray(status)
        if not set(np.unique(status_arr)).issubset({0, 1}):
            raise ValueError('status must be binary')
        status = status_arr.astype(np.int32)

        n = event.shape[0]
        self._n = n
        self.tie_breaking = tie_breaking

        # Handle start times
        if start is None:
            start_arr = -np.ones(n) * np.inf
            self._have_start_times = False
        else:
            start_arr = np.asarray(start).astype(float)
            self._have_start_times = True

        # Use stratified C++ implementation with a single stratum
        strata = np.zeros(n, dtype=np.int32)

        self._cpp = _StratifiedCoxDevianceCpp(
            event,
            status,
            strata,
            start_arr,
            tie_breaking == 'efron'
        )

        # Store for compatibility with existing code
        self._status = status

        # Store preprocessing results for backward compatibility with tests
        # that access internal attributes
        (self._preproc,
         self._event_order,
         self._start_order) = c_preprocess(start_arr, event, status)
        self._event_order = self._event_order.astype(np.int32)
        self._start_order = self._start_order.astype(np.int32)
        self._first = np.asarray(self._preproc['first']).astype(np.int32)
        self._last = np.asarray(self._preproc['last']).astype(np.int32)
        self._event_map = np.asarray(self._preproc['event_map']).astype(np.int32)
        self._start_map = np.asarray(self._preproc['start_map']).astype(np.int32)
        self._scaling = np.asarray(self._preproc['scaling'])
        self._first_start = self._first[self._start_map]

    def __call__(self,
                 linear_predictor: np.ndarray,
                 sample_weight: np.ndarray = None) -> CoxDevianceResult:
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
        linear_predictor = np.asarray(linear_predictor).astype(float)

        if sample_weight is None:
            sample_weight = np.ones_like(linear_predictor)
        else:
            sample_weight = np.asarray(sample_weight).astype(float)

        # Call C++ stratified implementation
        deviance, loglik_sat, gradient, diag_hessian = self._cpp(
            linear_predictor, sample_weight
        )

        return CoxDevianceResult(
            linear_predictor=linear_predictor,
            sample_weight=sample_weight,
            loglik_sat=loglik_sat,
            deviance=deviance,
            gradient=gradient,
            diag_hessian=diag_hessian,
            __hash_args__=""
        )

    def information(self,
                    linear_predictor: np.ndarray,
                    sample_weight: np.ndarray = None) -> LinearOperator:
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
        return CoxInformation(self, linear_predictor, sample_weight)


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
    linear_predictor : np.ndarray
        Linear predictor values (X @ beta).
    sample_weight : np.ndarray, optional
        Sample weights. If None, uses equal weights.

    Attributes
    ----------
    shape : tuple
        Shape of the information matrix (n, n).
    dtype : type
        Data type of the matrix elements.
    """

    def __init__(self,
                 coxdev: CoxDeviance,
                 linear_predictor: np.ndarray,
                 sample_weight: np.ndarray = None):
        """Initialize the linear operator dimensions."""
        self.coxdev = coxdev
        self.linear_predictor = np.asarray(linear_predictor).astype(float)

        if sample_weight is None:
            self.sample_weight = np.ones_like(self.linear_predictor)
        else:
            self.sample_weight = np.asarray(sample_weight).astype(float)

        n = coxdev._n

        # Ensure buffers are computed by calling __call__
        self.coxdev(self.linear_predictor, self.sample_weight)

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
        # Negate the input (information = -Hessian of log-likelihood)
        v = -np.asarray(arg).reshape(-1).astype(float)
        result = self.coxdev._cpp.hessian_matvec(
            v, self.linear_predictor, self.sample_weight
        )
        return result

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


class CoxIRLSState:
    """
    Stateful IRLS/coordinate descent interface for Cox regression.

    This class caches expensive quantities (exp(eta), risk sums, working
    weights/response) computed once per outer IRLS iteration, enabling
    efficient coordinate descent optimization.

    Parameters
    ----------
    coxdev : CoxDeviance
        The CoxDeviance object containing preprocessed survival data.

    Examples
    --------
    >>> import numpy as np
    >>> from coxdev import CoxDeviance, CoxIRLSState
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n, p = 100, 5
    >>> X = np.random.randn(n, p)
    >>> beta_true = np.array([1, -0.5, 0.3, 0, 0])
    >>> eta_true = X @ beta_true
    >>> time = np.random.exponential(1 / np.exp(eta_true))
    >>> status = np.random.binomial(1, 0.7, n)
    >>> # Create Cox deviance and IRLS state
    >>> cox = CoxDeviance(event=time, status=status)
    >>> irls = CoxIRLSState(cox)
    >>> # Initialize
    >>> beta = np.zeros(p)
    >>> eta = X @ beta
    >>> weights = np.ones(n)
    >>> # Outer IRLS iteration
    >>> irls.recompute_outer(eta, weights)
    >>> print(f"Initial deviance: {irls.current_deviance():.4f}")
    >>> # Inner coordinate descent pass
    >>> for j in range(p):
    ...     grad_j, hess_jj = irls.weighted_inner_product(X[:, j])
    ...     delta = grad_j / hess_jj
    ...     beta[j] += delta
    ...     irls.update_residuals(delta, X[:, j])
    """

    def __init__(self, coxdev: CoxDeviance):
        """
        Initialize the IRLS state from a CoxDeviance object.

        Parameters
        ----------
        coxdev : CoxDeviance
            The CoxDeviance object containing preprocessed survival data.
        """
        self._coxdev = coxdev
        self._cpp = _CoxIRLSStateCpp(coxdev._cpp)

    def recompute_outer(self,
                        eta: np.ndarray,
                        weights: np.ndarray = None) -> float:
        """
        Recompute all cached quantities (call once per outer IRLS iteration).

        This is the expensive O(15n) operation that should be called once
        per outer IRLS iteration, not in the inner coordinate descent loop.

        Parameters
        ----------
        eta : np.ndarray
            Current linear predictor values.
        weights : np.ndarray, optional
            Sample weights. If None, uses equal weights.

        Returns
        -------
        float
            Current deviance value.
        """
        eta = np.asarray(eta).astype(float)
        if weights is None:
            weights = np.ones_like(eta)
        else:
            weights = np.asarray(weights).astype(float)
        return self._cpp.recompute_outer(eta, weights)

    def working_weights(self) -> np.ndarray:
        """
        Get cached working weights.

        Returns
        -------
        np.ndarray
            Working weights w = -diag(Hessian) / 2.
        """
        return self._cpp.working_weights()

    def working_response(self) -> np.ndarray:
        """
        Get cached working response.

        Returns
        -------
        np.ndarray
            Working response z = eta - gradient / diag(Hessian).
        """
        return self._cpp.working_response()

    def residuals(self) -> np.ndarray:
        """
        Get cached residuals.

        Returns
        -------
        np.ndarray
            Residuals r = w * (z - eta).
        """
        return self._cpp.residuals()

    def current_deviance(self) -> float:
        """
        Get deviance from the last recompute_outer call.

        Returns
        -------
        float
            Current deviance value.
        """
        return self._cpp.current_deviance()

    def weighted_inner_product(self, x_j: np.ndarray) -> tuple:
        """
        Compute gradient and Hessian diagonal for coordinate j.

        Parameters
        ----------
        x_j : np.ndarray
            The j-th column of the design matrix.

        Returns
        -------
        tuple
            (gradient_j, hessian_jj) where:
            - gradient_j = sum(x_j * residuals)
            - hessian_jj = sum(w * x_j^2)
        """
        x_j = np.asarray(x_j).astype(float)
        return self._cpp.weighted_inner_product(x_j)

    def update_residuals(self, delta: float, x_j: np.ndarray) -> None:
        """
        Update residuals after beta_j changes by delta.

        This is an O(n) incremental update: r -= delta * w * x_j

        Parameters
        ----------
        delta : float
            Change in coefficient beta_j.
        x_j : np.ndarray
            The j-th column of the design matrix.
        """
        x_j = np.asarray(x_j).astype(float)
        self._cpp.update_residuals(float(delta), x_j)

    def reset_residuals(self, eta_current: np.ndarray) -> None:
        """
        Reset residuals for a new coordinate descent pass.

        Parameters
        ----------
        eta_current : np.ndarray
            Current linear predictor values.
        """
        eta_current = np.asarray(eta_current).astype(float)
        self._cpp.reset_residuals(eta_current)
