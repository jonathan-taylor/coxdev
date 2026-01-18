"""
C++ Implementation of Stratified Cox Proportional Hazards Model.

This module provides a Python wrapper around the C++ stratified Cox implementation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
from scipy.sparse.linalg import LinearOperator

from .base import CoxDevianceResult
from .coxc import StratifiedCoxDevianceCpp as _StratifiedCoxDevianceCpp


@dataclass
class StratifiedCoxDevianceCppResult:
    """Result from stratified Cox deviance computation."""
    linear_predictor: np.ndarray
    sample_weight: np.ndarray
    loglik_sat: float
    deviance: float
    gradient: np.ndarray
    diag_hessian: np.ndarray


class StratifiedCoxDevianceCpp:
    """
    C++ Stratified Cox Proportional Hazards Model Deviance Calculator.

    Uses pure C++ implementation for all strata processing, avoiding
    Python-level loops for better performance.

    Parameters
    ----------
    event : np.ndarray
        Event (failure) times.
    status : np.ndarray
        Event indicators (1=event, 0=censored).
    strata : np.ndarray, optional
        Stratum labels for each observation. If None, all observations
        are treated as belonging to a single stratum.
    start : np.ndarray, optional
        Start times for left-truncated data. If None, all start times
        are set to -infinity.
    tie_breaking : {'efron', 'breslow'}, default='efron'
        Tie-breaking method. 'efron' is more accurate for tied event times,
        'breslow' is simpler and compatible with glmnet.

    Examples
    --------
    >>> import numpy as np
    >>> from coxdev import StratifiedCoxDevianceCpp
    >>> event = np.array([3, 6, 8, 4, 6, 4, 3, 2, 2, 5, 3, 4])
    >>> status = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1])
    >>> strata = np.repeat([0, 1, 2], 4)
    >>> cox = StratifiedCoxDevianceCpp(event=event, status=status, strata=strata)
    >>> eta = np.linspace(-1, 1, len(event))
    >>> result = cox(eta)
    >>> print(round(result.deviance, 4))
    14.2741
    """

    def __init__(
        self,
        event: np.ndarray,
        status: np.ndarray,
        strata: Optional[np.ndarray] = None,
        start: Optional[np.ndarray] = None,
        tie_breaking: Literal['efron', 'breslow'] = 'efron'
    ):
        event = np.asarray(event).astype(float)
        status = np.asarray(status)

        if not set(np.unique(status)).issubset({0, 1}):
            raise ValueError('status must be binary')

        status = np.asarray(status).astype(np.int32)
        n = event.shape[0]

        if strata is None:
            strata = np.zeros(n, dtype=np.int32)
        else:
            strata = np.asarray(strata).astype(np.int32)

        if start is None:
            start = -np.ones(n) * np.inf
            self._have_start_times = False
        else:
            start = np.asarray(start).astype(float)
            self._have_start_times = True

        self._efron = tie_breaking == 'efron'
        self._n = n
        self._event = event
        self._status = status
        self._strata = strata
        self._start = start

        # Create the C++ object
        self._cpp = _StratifiedCoxDevianceCpp(
            event,
            status,
            strata,
            start,
            self._efron
        )

        # Store last linear predictor and sample weight for information matrix
        self._last_linear_predictor = None
        self._last_sample_weight = None

    def __call__(
        self,
        linear_predictor: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> CoxDevianceResult:
        """
        Compute Cox deviance, gradient, and diagonal Hessian.

        Parameters
        ----------
        linear_predictor : np.ndarray
            Linear predictor (X @ beta) for each observation.
        sample_weight : np.ndarray, optional
            Sample weights. If None, all weights are set to 1.

        Returns
        -------
        CoxDevianceResult
            Named tuple with deviance, gradient, diagonal Hessian, etc.
        """
        linear_predictor = np.asarray(linear_predictor).astype(float)

        if sample_weight is None:
            sample_weight = np.ones_like(linear_predictor)
        else:
            sample_weight = np.asarray(sample_weight).astype(float)

        # Store for information matrix
        self._last_linear_predictor = linear_predictor.copy()
        self._last_sample_weight = sample_weight.copy()

        # Call C++ implementation
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

    def information(
        self,
        linear_predictor: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> LinearOperator:
        """
        Return a LinearOperator representing the information matrix.

        Parameters
        ----------
        linear_predictor : np.ndarray
            Linear predictor (X @ beta) for each observation.
        sample_weight : np.ndarray, optional
            Sample weights. If None, all weights are set to 1.

        Returns
        -------
        LinearOperator
            A scipy LinearOperator for computing information matrix-vector products.
        """
        return StratifiedCoxInformationCpp(self, linear_predictor, sample_weight)

    @property
    def n_strata(self) -> int:
        """Number of strata."""
        return self._cpp.n_strata

    @property
    def n_total(self) -> int:
        """Total number of observations."""
        return self._cpp.n_total


class StratifiedCoxInformationCpp(LinearOperator):
    """
    LinearOperator for stratified Cox information matrix using C++ implementation.
    """

    def __init__(
        self,
        strat_cox: StratifiedCoxDevianceCpp,
        linear_predictor: np.ndarray,
        sample_weight: Optional[np.ndarray]
    ):
        self.strat_cox = strat_cox
        self.linear_predictor = np.asarray(linear_predictor).astype(float)

        if sample_weight is None:
            self.sample_weight = np.ones_like(self.linear_predictor)
        else:
            self.sample_weight = np.asarray(sample_weight).astype(float)

        self.n = self.linear_predictor.shape[0]
        self.shape = (self.n, self.n)
        self.dtype = float

        # Ensure buffers are computed by calling __call__
        self.strat_cox(self.linear_predictor, self.sample_weight)

    def _matvec(self, v: np.ndarray) -> np.ndarray:
        """Compute information matrix-vector product."""
        # Negate the input (same as base.py CoxInformation)
        # This is because information = -Hessian of log-likelihood
        v = -np.asarray(v).reshape(-1).astype(float)
        result = self.strat_cox._cpp.hessian_matvec(
            v, self.linear_predictor, self.sample_weight
        )
        return result

    def _adjoint(self, v: np.ndarray) -> np.ndarray:
        """Information matrix is symmetric."""
        return self._matvec(v)
