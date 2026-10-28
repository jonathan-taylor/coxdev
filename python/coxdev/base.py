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
    __hash_args__ : str, optional
        Unused; kept for compatibility.
    """

    linear_predictor: np.ndarray
    sample_weight: np.ndarray
    loglik_sat: float
    deviance: float
    gradient: Optional[np.ndarray]
    diag_hessian: Optional[np.ndarray]
    __hash_args__: Optional[str] = None

    @property
    def loglik(self):
        """The (weighted) log partial likelihood, ``loglik_sat - deviance / 2``:
        R's ``coxph(...)$loglik``."""
        return self.loglik_sat - self.deviance / 2


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
class SurvivalCurves(object):
    """
    Survival curves of a Cox model, laid out as R's ``survfit.coxph``: from
    :meth:`CoxDeviance.survfit`.

    The rows are the distinct event times (of failures and censorings) in
    increasing time within each stratum, stratum by stratum; the columns are
    the curves. A curve of a subject in one stratum is NaN on the rows of the
    other strata.

    Attributes
    ----------
    time : np.ndarray
        The times, ``(m,)``.
    stratum : np.ndarray
        The stratum label of each row (0 without strata).
    n_risk, n_event, n_censor : np.ndarray
        The weight at risk at each time, and the weights of the failures and
        of the censored observations there.
    cumulative_hazard : np.ndarray
        The cumulative hazard of each curve, ``(m, n_curves)``: Breslow's
        estimate, or Efron's for Efron's ties, constant between failure times.
    """

    time: np.ndarray
    stratum: np.ndarray
    n_risk: np.ndarray
    n_event: np.ndarray
    n_censor: np.ndarray
    cumulative_hazard: np.ndarray

    @property
    def survival(self):
        """The survival probabilities ``exp(-cumulative_hazard)``, R's default
        (``stype = 2``)."""
        return np.exp(-self.cumulative_hazard)

    def predict(self, times):
        """
        The survival probabilities of the curves at ``times``.

        Parameters
        ----------
        times : np.ndarray
            ``(k,)``.

        Returns
        -------
        np.ndarray
            ``(k, n_curves)``. Each curve is a right-continuous step function
            of its stratum's rows, 1 before its first time.
        """
        times = np.atleast_1d(np.asarray(times, float))
        value = np.empty((times.shape[0], self.cumulative_hazard.shape[1]))
        for j in range(self.cumulative_hazard.shape[1]):
            rows = np.nonzero(~np.isnan(self.cumulative_hazard[:, j]))[0]
            pos = np.searchsorted(self.time[rows], times, side='right') - 1
            H = np.where(pos >= 0, self.cumulative_hazard[rows[np.maximum(pos, 0)], j], 0)
            value[:, j] = np.exp(-H)
        return value


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
        Stratum labels for each observation: integers, strings or any other
        values ``np.unique`` can sort (not NaN). If None, computes an
        unstratified model. ``baseline_hazard`` reports each stratum by its
        label.
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

        # the C++ takes the strata as integer codes 0, 1, ...: the labels, in
        # sorted order, are kept to report them
        self._strata_labels = None
        if strata is not None:
            strata = np.asarray(strata)
            if strata.dtype.kind == 'f' and np.isnan(strata).any():
                raise ValueError('strata must not contain NaN')
            try:
                labels, codes = np.unique(strata, return_inverse=True)
            except TypeError as e:
                raise ValueError(f'strata must be labels that can be sorted: {e}')
            self._strata_labels = labels
            strata = codes.reshape(strata.shape).astype(np.int32)

        if strata is not None and ((status.shape != event.shape) or (strata.shape != event.shape) or (start.shape != event.shape)):
            raise ValueError("status, event, start and strata must have same shape")

        self._raw_event = event
        self._raw_status = status
        self._raw_start = start
        self._strata = strata
        
        self._reconstruct_coxc(sample_weight)

    def _reconstruct_coxc(self, sample_weight):
        # copies of the weights and of the last linear predictor, against which
        # each call is compared: equal arrays reuse the object and its result
        self._weight = np.array(sample_weight, float)
        self._last_linear_predictor = None
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
        The deviance, its gradient and the diagonal of its Hessian at a
        linear predictor.

        The deviance is ``2 * (loglik_sat - loglik)``, with ``loglik`` the
        (weighted) log partial likelihood, and the derivatives are with
        respect to ``linear_predictor``: for a design ``X``, the score of
        the coefficients is ``X.T @ (-gradient / 2)``.

        The object is rebuilt, sorting the data again, if ``sample_weight``
        differs from the weights of the last call, and the result is reused if
        ``linear_predictor`` is also that of the last call.

        Parameters
        ----------
        linear_predictor : np.ndarray
            The linear predictor, one entry per observation.
        sample_weight : np.ndarray, optional
            Defaults to one.

        Returns
        -------
        CoxDevianceResult
            With ``deviance``, ``gradient``, ``diag_hessian`` and
            ``loglik_sat``.
        """
        if sample_weight is None:
            sample_weight = np.ones_like(linear_predictor, float)
        else:
            sample_weight = np.asarray(sample_weight).astype(float)

        linear_predictor = np.asarray(linear_predictor).astype(float)
            
        if not np.array_equal(sample_weight, self._weight):
            self._reconstruct_coxc(sample_weight)

        if (self._last_linear_predictor is None or
            not np.array_equal(linear_predictor, self._last_linear_predictor)):
            deviance = self._coxc.compute_deviance(linear_predictor, sample_weight)
            self._last_linear_predictor = linear_predictor.copy()

            # copies, not views of the C++ buffers, which the next call
            # overwrites
            self._result = CoxDevianceResult(linear_predictor=self._last_linear_predictor.copy(),
                                             sample_weight=self._weight.copy(),
                                             loglik_sat=self._coxc.loglik_sat,
                                             deviance=deviance,
                                             gradient=self._coxc.gradient.copy(),
                                             diag_hessian=self._coxc.diag_hessian.copy())
            
        return self._result

    def c_index(self,
                linear_predictor,
                sample_weight=None):
        """
        Harrell's concordance index (C-index) of a linear predictor.

        For each failure ``i`` at time ``t``, every ``j`` at risk at ``t``
        (``start[j] < t <= event[j]``) is compared with ``i``, except the
        other failures at ``t``. The pair has weight ``w_i w_j``, and is
        concordant if ``linear_predictor[i] > linear_predictor[j]``, a higher
        risk failing first; tied predictions count half. The C-index is the
        weighted fraction of concordant pairs: 1/2 for a predictor unrelated
        to survival, 1 for one that orders the failures perfectly. With
        strata, only pairs within a stratum are compared.

        This is R's ``survival::concordance(y ~ -linear_predictor, weights =
        w)`` (with ``+ strata(...)`` for strata) and, without strata,
        ``glmnet::Cindex``. It is computed in ``O(n log n)`` per column, by
        the algorithm of the ``survival`` package's concordance vignette
        (Therneau and Atkinson): subjects in decreasing time, with the
        weights of the risk set in a Fenwick tree (Fenwick, 1994) over the
        ranks of the predictor. It does not need the deviance to be computed.

        Parameters
        ----------
        linear_predictor : np.ndarray
            The linear predictor (a risk score), ``(n,)``, or ``(n, k)`` for
            ``k`` predictors, such as those along a path.
        sample_weight : np.ndarray, optional
            Defaults to one.

        Returns
        -------
        float or np.ndarray
            The C-index, one per column of a two-dimensional
            ``linear_predictor``; NaN if no pair is comparable.

        References
        ----------
        Harrell, F. E., Califf, R. M., Pryor, D. B., Lee, K. L. and Rosati,
        R. A. (1982). Evaluating the yield of medical tests. *JAMA*, 247(18),
        2543-2546.

        Harrell, F. E., Lee, K. L. and Mark, D. B. (1996). Multivariable
        prognostic models: issues in developing models, evaluating
        assumptions and adequacy, and measuring and reducing errors.
        *Statistics in Medicine*, 15(4), 361-387.

        Therneau, T. M. and Atkinson, E. J. The concordance statistic.
        Vignette of the R package ``survival``.

        Fenwick, P. M. (1994). A new data structure for cumulative frequency
        tables. *Software: Practice and Experience*, 24(3), 327-336.
        """
        pred = np.asarray(linear_predictor, float)
        one_dim = pred.ndim == 1
        pred = pred.reshape((pred.shape[0], -1))
        n = self._raw_event.shape[0]
        if pred.shape[0] != n:
            raise ValueError('linear_predictor must have one row per observation')
        if sample_weight is None:
            sample_weight = np.ones(n)
        sample_weight = np.asarray(sample_weight, float)
        values = np.empty(pred.shape[1])
        for j in range(pred.shape[1]):
            num, den = self._coxc.concordance(np.ascontiguousarray(pred[:, j]), sample_weight)
            values[j] = num / den if den > 0 else np.nan
        return values[0] if one_dim else values

    def information(self,
                    linear_predictor,
                    sample_weight=None):
        """
        The information at a linear predictor: half the Hessian of the
        deviance, the negative Hessian of the log partial likelihood, as a
        matrix-free ``scipy.sparse.linalg.LinearOperator``.

        A product ``I @ v`` costs about as much as a gradient, ``O(n)``, and
        ``X.T @ (I @ X)`` is the information of the coefficients of a design
        ``X``.

        Parameters
        ----------
        linear_predictor : np.ndarray
            The linear predictor, one entry per observation.
        sample_weight : np.ndarray, optional
            Defaults to one.

        Returns
        -------
        CoxInformation
            An ``(n, n)`` linear operator.
        """
        result = self(linear_predictor,
                      sample_weight)
        return CoxInformation(result=result,
                              coxdev=self)

    def information_derivative(self,
                               linear_predictor,
                               direction,
                               sample_weight=None):
        """
        The derivative of the information at ``linear_predictor`` in the
        direction ``direction`` of the linear predictor,
        ``d/dt information(linear_predictor + t * direction)`` at ``t = 0``,
        as a matrix-free ``scipy.sparse.linalg.LinearOperator``.

        It is symmetric, and a product costs ``O(n)``, a few times a product
        with the information. ``X.T @ (dI @ X)`` is the derivative of the
        information of the coefficients of a design ``X``, as needed for the
        derivative of a log determinant (e.g. in REML or a Laplace
        approximation), where ``direction`` is the derivative of the linear
        predictor. The derivative of the Hessian of the deviance is twice it.

        Parameters
        ----------
        linear_predictor : np.ndarray
            The linear predictor, one entry per observation.
        direction : np.ndarray
            The direction of the derivative, one entry per observation.
        sample_weight : np.ndarray, optional
            Defaults to one.

        Returns
        -------
        CoxInformationDerivative
            An ``(n, n)`` linear operator.
        """
        result = self(linear_predictor,
                      sample_weight)
        direction = np.asarray(direction, float).reshape(-1)
        if direction.shape != result.linear_predictor.shape:
            raise ValueError('direction should have one entry per observation')
        return CoxInformationDerivative(result=result,
                                        coxdev=self,
                                        direction=direction)

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
            2**64). The same seed gives the same draws, here and from R: the
            same random numbers on every platform, and draws equal to
            rounding (the last bits of ``exp``, ``log``, ``sin`` and ``cos``
            can differ between math libraries). If None, a seed is drawn from
            numpy's random entropy.

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
        return self._by_column(self._coxc.score_residuals,
                               linear_predictor, X, sample_weight)

    def schoenfeld_residuals(self,
                             linear_predictor,
                             X,
                             sample_weight=None):
        """
        The Schoenfeld residuals of a design ``X`` at ``linear_predictor``.

        Row ``i``, for an observation that fails (with positive weight), is
        ``x_i - xbar``, with ``xbar`` the mean of the covariates over its
        risk set with weights ``w exp(eta)``. With ``tie_breaking='efron'``,
        ``xbar`` is the average over the failures tied with ``i`` of their
        Efron means, as in R's ``survival``. The other rows are zero.
        Weighted by the sample weights, the rows sum to the score:
        ``sample_weight @ R`` is ``X.T @ (-gradient / 2)``.

        ``residuals(fit, type = "schoenfeld")`` of a ``coxph`` fit in R has
        the rows of the failures only, in order of failure time:
        ``R[status == 1]`` here, in native order.

        Plotted against time, they check the proportional hazards
        assumption (Grambsch and Therneau, 1994).

        The residuals are built a column at a time, each in ``O(n)``.

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

        References
        ----------
        Schoenfeld, D. (1982). Partial residuals for the proportional hazards
        regression model. *Biometrika*, 69(1), 239-241.

        Grambsch, P. M. and Therneau, T. M. (1994). Proportional hazards
        tests and diagnostics based on weighted residuals. *Biometrika*,
        81(3), 515-526.
        """
        return self._by_column(self._coxc.schoenfeld_residuals,
                               linear_predictor, X, sample_weight)

    def risk_set_variance(self,
                          linear_predictor,
                          Z,
                          sample_weight=None):
        """
        The risk-set variances of the columns of ``Z`` at ``linear_predictor``.

        Row ``i``, for an observation that fails (with positive weight), is
        the variance of each covariate over its risk set, with weights
        ``w exp(eta)``: the ``V_z`` by which Grambsch and Therneau (1994)
        scale the Schoenfeld residuals, ``beta + r / V_z`` for one
        covariate. With ``tie_breaking='efron'``, it is the average over the
        failures tied with ``i`` of their Efron variances, as for
        ``schoenfeld_residuals``. The other rows are zero. Weighted by the
        sample weights, the rows sum to the diagonal of the information:
        ``sample_weight @ V`` is ``diag(Z.T @ I @ Z)``.

        The covariance of two covariates ``y`` and ``z`` follows by
        polarization, ``(V(y + z) - V(y - z)) / 4``, which gives the full
        risk-set covariance matrices a pair of columns at a time.

        The variances are computed a column at a time, each in ``O(n)``.

        Parameters
        ----------
        linear_predictor : np.ndarray
            The linear predictor, e.g. of a fitted model.
        Z : np.ndarray or scipy.sparse matrix
            The covariates, one row per observation; a sparse ``Z`` is made
            dense a column at a time.
        sample_weight : np.ndarray, optional
            Defaults to one.

        Returns
        -------
        np.ndarray
            ``(n, p)``, or ``(n,)`` for a one-dimensional ``Z``.

        References
        ----------
        Grambsch, P. M. and Therneau, T. M. (1994). Proportional hazards
        tests and diagnostics based on weighted residuals. *Biometrika*,
        81(3), 515-526.
        """
        return self._by_column(self._coxc.risk_set_variance,
                               linear_predictor, Z, sample_weight)

    def _by_column(self, residuals, linear_predictor, X, sample_weight):
        # the C++ takes one covariate at a time
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
        U = np.empty(X2.shape)
        for j in range(X2.shape[1]):
            U[:, j] = residuals(column(j))
        return U[:, 0] if one_dim else U

    def expected_events(self,
                        linear_predictor,
                        sample_weight=None):
        """
        The expected number of events of each observation at
        ``linear_predictor``: ``exp(linear_predictor[i])`` times the increase
        of the baseline cumulative hazard (see ``baseline_hazard``) over
        ``(start[i], event[i]]``. With ``tie_breaking='efron'``, a failure
        gets Efron's share of the hazard at its own failure time.

        This is ``predict(fit, type = "expected")`` of a ``coxph`` fit in R:
        ``status - expected_events`` is the martingale residual (Barlow and
        Prentice, 1988; Therneau, Grambsch and Fleming, 1990).

        Parameters
        ----------
        linear_predictor : np.ndarray
            The linear predictor, e.g. of a fitted model.
        sample_weight : np.ndarray, optional
            Defaults to one.

        Returns
        -------
        np.ndarray
            ``(n,)``, in native order.

        References
        ----------
        Barlow, W. E. and Prentice, R. L. (1988). Residuals for relative risk
        regression. *Biometrika*, 75(1), 65-74.

        Therneau, T. M., Grambsch, P. M. and Fleming, T. R. (1990).
        Martingale-based residuals for survival models. *Biometrika*, 77(1),
        147-160.
        """
        self(linear_predictor, sample_weight)
        return np.asarray(self._coxc.expected_events())

    def survival_at_exit(self,
                         linear_predictor,
                         sample_weight=None):
        """
        The expected survival probability of each observation at its exit:
        the probability of surviving to ``event[i]``, given survival to
        ``start[i]`` (to 0 without start times), at ``linear_predictor``.
        It is ``exp(-expected_events)``, R's
        ``predict(fit, type = "survival")`` for a ``coxph`` fit.

        Parameters
        ----------
        linear_predictor : np.ndarray
            The linear predictor, e.g. of a fitted model.
        sample_weight : np.ndarray, optional
            Defaults to one.

        Returns
        -------
        np.ndarray
            ``(n,)``, in native order.

        References
        ----------
        Therneau, T. M. and Grambsch, P. M. (2000). *Modeling Survival Data:
        Extending the Cox Model*, Chapter 4. Springer, New York.
        """
        return np.exp(-self.expected_events(linear_predictor, sample_weight))

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
        stratum = np.asarray(stratum)
        if self._strata_labels is not None:
            stratum = self._strata_labels[stratum]
        return BaselineHazard(time=np.asarray(time),
                              hazard=np.asarray(hazard),
                              cumulative_hazard=np.asarray(cumulative_hazard),
                              stratum=stratum)

    def survfit(self,
                linear_predictor,
                sample_weight=None,
                new_linear_predictor=None,
                new_strata=None,
                center=None):
        """
        Survival curves at a fixed linear predictor: R's ``survfit.coxph``
        (``se.fit = FALSE``) of ``coxph(y ~ offset(linear_predictor))``, with
        this object's ties. This is how R's ``survfit.coxnet`` computes the
        curves of a fitted model.

        Parameters
        ----------
        linear_predictor : np.ndarray
            The linear predictor of these data, e.g. of a fitted model.
        sample_weight : np.ndarray, optional
            Defaults to one.
        new_linear_predictor : np.ndarray, optional
            The linear predictors of the subjects to give curves of. If None,
            one curve (in each stratum) at ``center``, as R does without
            ``newdata``.
        new_strata : np.ndarray, optional
            The stratum labels of the new subjects, needed with
            ``new_linear_predictor`` when the data have strata.
        center : float, optional
            The linear predictor of the curve when ``new_linear_predictor``
            is None. Defaults to the weighted mean of ``linear_predictor``
            (R's ``coxph`` means for right-censored data; for (start, stop]
            data R averages without the weights).

        Returns
        -------
        SurvivalCurves
        """
        lp = np.asarray(linear_predictor, float)
        w = np.ones_like(lp) if sample_weight is None else np.asarray(sample_weight, float)
        self(lp, w)
        time, n_risk, n_event, n_censor, H, stratum = (np.asarray(v) for v in
                                                         self._coxc.survival_table())
        if center is None:
            center = np.sum(w * lp) / np.sum(w)
        if new_linear_predictor is None:
            new_lp = np.array([center], float)
        else:
            new_lp = np.atleast_1d(np.asarray(new_linear_predictor, float))

        # the hazard is for lp as given: H exp(new_lp), formed as
        # (H exp(center)) exp(new_lp - center) to keep both factors moderate
        cumulative_hazard = (H * np.exp(center))[:, None] * np.exp(new_lp - center)[None, :]
        if self._strata_labels is not None and new_linear_predictor is not None:
            if new_strata is None:
                raise ValueError('new_strata is needed for the curves of a stratified model')
            new_codes = np.searchsorted(self._strata_labels, np.atleast_1d(np.asarray(new_strata)))
            new_codes = np.minimum(new_codes, len(self._strata_labels) - 1)
            if not np.all(self._strata_labels[new_codes] == np.atleast_1d(np.asarray(new_strata))):
                raise ValueError('new_strata contains labels not seen in the training data')
            if new_codes.shape[0] != new_lp.shape[0]:
                raise ValueError('new_strata and new_linear_predictor have different lengths')
            cumulative_hazard[stratum[:, None] != new_codes[None, :]] = np.nan
        if self._strata_labels is not None:
            stratum = self._strata_labels[stratum]
        return SurvivalCurves(time=time,
                              stratum=stratum,
                              n_risk=n_risk,
                              n_event=n_event,
                              n_censor=n_censor,
                              cumulative_hazard=cumulative_hazard)

    def martingale_residuals(self,
                             linear_predictor,
                             sample_weight=None):
        """
        The martingale residuals at ``linear_predictor``, ``status -
        expected_events``: R's ``residuals(fit, type = "martingale")`` of a
        ``coxph`` fit (Barlow and Prentice, 1988; Therneau, Grambsch and
        Fleming, 1990). See ``expected_events``.

        Parameters
        ----------
        linear_predictor : np.ndarray
            The linear predictor, e.g. of a fitted model.
        sample_weight : np.ndarray, optional
            Defaults to one.

        Returns
        -------
        np.ndarray
            ``(n,)``, in native order.
        """
        return self._raw_status - self.expected_events(linear_predictor, sample_weight)

    def deviance_residuals(self,
                           linear_predictor,
                           sample_weight=None):
        """
        The deviance residuals at ``linear_predictor``, ``sign(M) sqrt(-2 (M +
        status log(status - M)))`` of the martingale residuals ``M``: R's
        ``residuals(fit, type = "deviance")`` of a ``coxph`` fit (Therneau,
        Grambsch and Fleming, 1990).

        Parameters
        ----------
        linear_predictor : np.ndarray
            The linear predictor, e.g. of a fitted model.
        sample_weight : np.ndarray, optional
            Defaults to one.

        Returns
        -------
        np.ndarray
            ``(n,)``, in native order.
        """
        M = self.martingale_residuals(linear_predictor, sample_weight)
        status = self._raw_status
        with np.errstate(divide='ignore', invalid='ignore'):
            d = -2 * (M + np.where(status > 0, status * np.log(status - M), 0))
        return np.sign(M) * np.sqrt(np.maximum(d, 0))


@dataclass
class CoxInformation(LinearOperator):
    """
    The information of a Cox model at a linear predictor, as a matrix-free
    ``(n, n)`` linear operator, from :meth:`CoxDeviance.information`.

    ``I @ v`` (or ``I.matvec(v)``) and ``I @ V`` for a matrix ``V`` give
    products with the information, each column in ``O(n)``. It is
    symmetric, so ``I.T`` is ``I``. The matrix is never formed: as a dense
    array it would take ``8 n**2`` bytes.
    """

    coxdev: CoxDeviance
    result: CoxDevianceResult

    def __post_init__(self):
        """Initialize the linear operator dimensions."""
        n = self.result.linear_predictor.shape[0]
        # sets shape and dtype (and, in SciPy >= 1.18, the array namespace used by @)
        LinearOperator.__init__(self, dtype=float, shape=(n, n))
        
    def _prepare(self):
        # the C++ object holds the state of its last linear predictor: bring
        # it back to this one (free if it is still there, the result being
        # cached), so that products stay right after other calls
        self.coxdev(self.result.linear_predictor, self.result.sample_weight)

    def _matvec(self, arg):
        """
        Compute matrix-vector product with the information matrix.
        """
        # arg is in native order
        self._prepare()
        arg_arr = np.asarray(arg).reshape(-1).astype(float)
        return self.coxdev._coxc.compute_hessian_matvec(arg_arr)

    def _matmat(self, arg):
        """
        Compute the product with a matrix, all its columns in one C++ call.
        """
        self._prepare()
        arg_arr = np.asarray(arg, float).reshape((self.shape[0], -1))
        return self.coxdev._coxc.compute_hessian_matmat(np.asfortranarray(arg_arr))

    # it is symmetric
    _rmatvec = _matvec
    _rmatmat = _matmat

    def _adjoint(self):
        return self


@dataclass
class CoxInformationDerivative(LinearOperator):
    """
    The derivative of the information of a Cox model at a linear predictor
    in a direction of the linear predictor, as a matrix-free ``(n, n)``
    linear operator, from :meth:`CoxDeviance.information_derivative`.

    ``dI @ v`` and ``dI @ V`` give products with
    ``d/dt I(linear_predictor + t * direction)``, each column in ``O(n)``. It
    is symmetric, so ``dI.T`` is ``dI``.
    """

    coxdev: CoxDeviance
    result: CoxDevianceResult
    direction: np.ndarray

    def __post_init__(self):
        n = self.result.linear_predictor.shape[0]
        LinearOperator.__init__(self, dtype=float, shape=(n, n))

    def _prepare(self):
        # as for CoxInformation
        self.coxdev(self.result.linear_predictor, self.result.sample_weight)

    def _matvec(self, arg):
        self._prepare()
        arg_arr = np.asarray(arg).reshape(-1).astype(float)
        return self.coxdev._coxc.compute_information_derivative_matvec(self.direction, arg_arr)

    def _matmat(self, arg):
        self._prepare()
        arg_arr = np.asarray(arg, float).reshape((self.shape[0], -1))
        return self.coxdev._coxc.compute_information_derivative_matmat(self.direction,
                                                                       np.asfortranarray(arg_arr))

    # it is symmetric
    _rmatvec = _matvec
    _rmatmat = _matmat

    def _adjoint(self):
        return self
