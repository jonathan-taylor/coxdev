"""
Cox Proportional Hazards Model Deviance Computation.

This module provides efficient computation of Cox model deviance, gradients,
and Hessian information matrices for survival analysis with support for
different tie-breaking methods (Efron and Breslow) and stratified models.
"""

from dataclasses import dataclass, InitVar
from typing import Literal, Optional

from scipy.sparse import issparse
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
class BaselineHazard(object):
    """
    The estimate of the baseline hazard of a Cox model.

    One entry per distinct failure time with positive weight, in increasing
    time within each stratum. The hazard is for the linear predictor as
    given: the cumulative hazard of observation ``i`` at time ``t`` is the
    ``cumulative_hazard`` at the last failure time ``<= t`` in its stratum,
    times ``exp(linear_predictor[i])``.

    Attributes
    ----------
    time : np.ndarray
        The distinct failure times.
    hazard : np.ndarray
        The increment of the cumulative hazard at each time.
    cumulative_hazard : np.ndarray
        The cumulative hazard, the running sum of ``hazard`` within each
        stratum.
    stratum : np.ndarray
        The stratum label of each entry (0 without strata).
    """

    time: np.ndarray
    hazard: np.ndarray
    cumulative_hazard: np.ndarray
    stratum: np.ndarray


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
    25.1943
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
        method : {'multinomial', 'bernoulli', 'fisher', 'wild'}
            The model-based methods draw the failures at each event time from
            the model and return their score: the sum over the failures of
            ``x_j`` minus its expectation, with risk weights ``sample_weight *
            exp(linear_predictor)``. They differ only at tied event times.

            - ``'multinomial'``: the ``d`` failures drawn with replacement,
              with probabilities proportional to the risk weights: the
              sampling model of Breslow's likelihood [2]_, whose score it is.
              Its covariance is the Breslow information.
            - ``'bernoulli'``: each subject at risk fails independently with
              probability ``min(1, d p)``, a Cox-model analogue of the weird
              bootstrap [1]_ [3]_, centered with the number of failures drawn.
              Its mean is zero only while ``d p < 1``: the cap biases it when
              ties are heavy.
            - ``'fisher'``: the set of ``d`` failures drawn from Fisher's
              noncentral hypergeometric distribution [6]_ [8]_ (independent
              Bernoullis given ``d`` failures), the sampling model of the
              exact discrete partial likelihood [4]_ [9]_, whose score it is,
              centered at its exact mean. Its covariance is the information of
              the exact likelihood. Sampling and centering use the recursion
              of Gail, Lubin and Rubinstein [7]_ on the log scale.
            - ``'wild'``: the score residuals (of the likelihood set by
              ``tie_breaking`` [2]_ [5]_) multiplied by independent standard
              normals [10]_ [12]_; its covariance is the sandwich, the sum of
              the outer products of the score residuals [11]_, which does not
              need the model to be right.
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

        References
        ----------
        .. [1] Andersen, P. K., Borgan, O., Gill, R. D. and Keiding, N. (1993).
           *Statistical Models Based on Counting Processes*. Springer, New York.
        .. [2] Breslow, N. E. (1974). Covariance analysis of censored survival
           data. *Biometrics*, 30(1), 89-99.
        .. [3] Burr, D. (1994). A comparison of certain bootstrap confidence
           intervals in the Cox model. *Journal of the American Statistical
           Association*, 89(428), 1290-1302.
        .. [4] Cox, D. R. (1972). Regression models and life-tables (with
           discussion). *Journal of the Royal Statistical Society, Series B*,
           34(2), 187-220.
        .. [5] Efron, B. (1977). The efficiency of Cox's likelihood function for
           censored data. *Journal of the American Statistical Association*,
           72(359), 557-565.
        .. [6] Fog, A. (2008). Sampling methods for Wallenius' and Fisher's
           noncentral hypergeometric distributions. *Communications in
           Statistics: Simulation and Computation*, 37(2), 241-257.
        .. [7] Gail, M. H., Lubin, J. H. and Rubinstein, L. V. (1981). Likelihood
           calculations for matched case-control studies and survival studies
           with tied death times. *Biometrika*, 68(3), 703-707.
        .. [8] Harkness, W. L. (1965). Properties of the extended hypergeometric
           distribution. *The Annals of Mathematical Statistics*, 36(3),
           938-945.
        .. [9] Kalbfleisch, J. D. and Prentice, R. L. (2002). *The Statistical
           Analysis of Failure Time Data*, 2nd edition. Wiley, Hoboken, NJ.
        .. [10] Kline, P. and Santos, A. (2012). A score based approach to wild
           bootstrap inference. *Journal of Econometric Methods*, 1(1), 23-41.
        .. [11] Lin, D. Y. and Wei, L. J. (1989). The robust inference for the
           Cox proportional hazards model. *Journal of the American Statistical
           Association*, 84(408), 1074-1078.
        .. [12] Lin, D. Y., Wei, L. J. and Ying, Z. (1993). Checking the Cox
           model with cumulative sums of martingale-based residuals.
           *Biometrika*, 80(3), 557-572.
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

    def score_residuals(self,
                        linear_predictor,
                        X,
                        sample_weight=None):
        """
        The score residuals of a design ``X`` at ``linear_predictor``.

        Row ``i`` is ``U_i``, the contribution of observation ``i`` to the
        score of the coefficients ``X.T @ (-gradient / 2)``: the sum over the
        failure times of ``(w_i 1{i fails} / d - wbar p_i) (x_i - xbar)``,
        centered within each risk set, with Efron's weights for ties if
        ``tie_breaking='efron'`` (Lin and Wei, 1989). The rows sum to the
        score, ``multipliers @ U`` is ``X.T @ wild_score(eta, multipliers)``,
        and ``U.T @ U`` is the middle of the robust (sandwich) variance.

        The residuals are built a column at a time, each in ``O(n)``. ``U``
        is dense even for a sparse ``X``, since the risk-set means spread
        each covariate over the subjects at risk. For draws ``U.T @ G`` (a
        wild bootstrap) there is no need to form ``U``:
        ``X.T @ wild_score(eta, G)`` is the same, in one ``O(n)`` call.

        Parameters
        ----------
        linear_predictor : np.ndarray
            The linear predictor, e.g. of a fitted model.
        X : np.ndarray or scipy.sparse matrix
            The design, one row per observation; a sparse ``X`` is made
            dense a column at a time.
        sample_weight : np.ndarray, optional
            Defaults to one.

        Returns
        -------
        np.ndarray
            ``(n, p)``, or ``(n,)`` for a one-dimensional ``X``.
        """
        if issparse(X):
            X2 = X.tocsc()
            one_dim = False
            column = lambda j: X2[:, [j]].toarray().ravel().astype(float)
        else:
            X = np.asarray(X, float)
            one_dim = X.ndim == 1
            X2 = X.reshape((-1, 1)) if one_dim else X
            column = lambda j: X2[:, j]
        if X2.shape[0] != np.asarray(linear_predictor).shape[0]:
            raise ValueError('X must have one row per observation')
        self(linear_predictor, sample_weight)
        # the C++ takes one covariate at a time
        U = np.empty(X2.shape)
        for j in range(X2.shape[1]):
            U[:, j] = self._coxc.score_residuals(column(j))
        return U[:, 0] if one_dim else U

    def baseline_hazard(self,
                        linear_predictor,
                        sample_weight=None):
        """
        The estimate of the baseline hazard at ``linear_predictor``:
        Breslow's, or with ``tie_breaking='efron'`` Efron's, which spreads the
        tied failures at a time over Efron's shrinking risk sets.

        Parameters
        ----------
        linear_predictor : np.ndarray
            The linear predictor, e.g. of a fitted model.
        sample_weight : np.ndarray, optional
            Defaults to one.

        Returns
        -------
        BaselineHazard
            The failure times, the increments of the cumulative hazard and
            the cumulative hazard, for ``linear_predictor`` as given, with
            the stratum of each.
        """
        self(linear_predictor, sample_weight)
        time, hazard, cumulative_hazard, stratum = self._coxc.baseline_hazard()
        return BaselineHazard(time=np.asarray(time),
                              hazard=np.asarray(hazard),
                              cumulative_hazard=np.asarray(cumulative_hazard),
                              stratum=np.asarray(stratum))


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
