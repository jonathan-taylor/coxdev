"""
Stratified Cox Proportional Hazards Model.

This module provides the StratifiedCoxDeviance class which wraps the
C++ StratifiedCoxDevianceCpp implementation for backward compatibility.
"""

import numpy as np
from typing import Optional, Literal
from scipy.sparse.linalg import LinearOperator

from .base import CoxDevianceResult
from .stratified_cpp import StratifiedCoxDevianceCpp, StratifiedCoxInformationCpp


class StratifiedCoxDeviance:
    """
    Stratified Cox Proportional Hazards Model Deviance Calculator.

    This class wraps StratifiedCoxDevianceCpp for backward compatibility.
    For new code, consider using StratifiedCoxDevianceCpp directly.

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

    def __init__(
        self,
        event: np.ndarray,
        status: np.ndarray,
        strata: Optional[np.ndarray] = None,
        start: Optional[np.ndarray] = None,
        tie_breaking: Literal['efron', 'breslow'] = 'efron'
    ):
        self.tie_breaking = tie_breaking
        self._cpp = StratifiedCoxDevianceCpp(
            event=event,
            status=status,
            strata=strata,
            start=start,
            tie_breaking=tie_breaking
        )

    def __call__(self, linear_predictor, sample_weight=None):
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
        return self._cpp(linear_predictor, sample_weight)

    def information(self, linear_predictor, sample_weight=None):
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
        return self._cpp.information(linear_predictor, sample_weight)

    @property
    def n_strata(self):
        """Number of strata."""
        return self._cpp.n_strata

    @property
    def n_total(self):
        """Total number of observations."""
        return self._cpp.n_total
