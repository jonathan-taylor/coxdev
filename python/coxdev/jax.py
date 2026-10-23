"""
The Cox deviance in JAX, for automatic differentiation.

The same deviance as :class:`coxdev.CoxDeviance` (Efron's or Breslow's ties,
(start, stop] data, weights, possibly zero, and strata), written with JAX so
that ``jax.grad``, ``jax.jvp`` and ``jax.hessian`` give its derivatives. The
preprocessing (the sort and the tie structure) is in numpy, a port of
``setup_preprocess`` in ``cpp/src/coxdev.cpp``, and depends on which weights
are 0: the weights passed later must have the zeros of those passed at
construction. Everything that depends on the linear predictor is a cumulative
sum or a gather, so a gradient or a Hessian-vector product costs ``O(n)``.

The derivatives need double precision: enable it with
``jax.config.update('jax_enable_x64', True)`` before creating arrays.

Unlike the C++ code, the centered linear predictor is not capped at 30
before exponentiating, so the derivatives stay exact for large values.
"""

import warnings

import numpy as np

import jax
import jax.numpy as jnp


def preprocess(event, status, start=None, sample_weight=None, strata=None):
    """
    The sort and the tie structure of (start, event] data, as the C++
    ``setup_preprocess``.

    Returns a dict of numpy arrays: ``event_order`` and ``start_order``
    (native indices in order of event and start times), and, in event order,
    ``status``, ``first`` and ``last`` (the first and last positions of the
    tied failures of each failure, itself for the others), ``scaling``
    (Efron's ``m / K`` within a set of ``K`` tied failures), ``event_map``
    (the number of starts at or before each event time) and ``start_map``
    (the number of event times before each start), and the flag
    ``have_start_times``.

    With ``strata``, the strata are stacked, in the order of their labels:
    each is sorted on its own and its positions shifted by the number of
    observations before it, as if the times of each stratum were moved into
    a window of their own. The risk sums then need the start times (of -inf
    within a stratum without them), which take out the later strata: one pass
    over all the strata (see the Computations notes, "Strata").
    """
    if strata is None:
        return _preprocess_one(event, status, start, sample_weight)

    event = np.asarray(event, float).reshape(-1)
    n = event.shape[0]
    status = np.asarray(status).reshape(-1)
    start = -np.inf * np.ones(n) if start is None else np.asarray(start, float).reshape(-1)
    weight = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float).reshape(-1)
    strata = np.asarray(strata).reshape(-1)

    parts, offset = [], 0
    for label in np.unique(strata):
        idx = np.nonzero(strata == label)[0]
        prep = _preprocess_one(event[idx], status[idx], start[idx], weight[idx])
        prep['event_order'] = idx[prep['event_order']]       # native indices
        prep['start_order'] = idx[prep['start_order']]
        for key in ['first', 'last', 'event_map', 'start_map']:
            prep[key] = prep[key] + offset                    # positions
        parts.append(prep)
        offset += idx.shape[0]

    stacked = {key: np.concatenate([prep[key] for prep in parts])
               for key in ['event_order', 'start_order', 'status', 'first', 'last',
                           'scaling', 'event_map', 'start_map']}
    stacked['have_start_times'] = True
    return stacked


def _preprocess_one(event, status, start=None, sample_weight=None):
    # one stratum: the port of setup_preprocess
    event = np.asarray(event, float).reshape(-1)
    status = np.asarray(status).reshape(-1).astype(int)
    n = event.shape[0]
    have_start_times = start is not None
    if start is None:
        start = -np.inf * np.ones(n)
    start = np.asarray(start, float).reshape(-1)
    if sample_weight is None:
        sample_weight = np.ones(n)
    weight = np.asarray(sample_weight, float).reshape(-1)

    # starts and events stacked: at a tied time the failures come first,
    # those with positive weight before those with weight 0, then the
    # censored and the starts; ties are broken by position, a total order
    time = np.hstack([start, event])
    status_c = np.hstack([np.ones(n, int), 1 - status])
    is_start = np.hstack([np.ones(n, int), np.zeros(n, int)])
    neg_weight = np.hstack([-weight, -weight])
    index = np.hstack([np.arange(n), np.arange(n)])

    order = np.lexsort((np.arange(2 * n), is_start, neg_weight, status_c, time))
    time, status_s, is_start = time[order], 1 - status_c[order], is_start[order].astype(bool)
    index, weight_s = index[order], -neg_weight[order]

    starts_before = np.cumsum(is_start) - is_start      # starts before each row
    events_before = np.cumsum(~is_start) - ~is_start    # events before each row

    event_order = index[~is_start]
    start_order = index[is_start]
    event_map = starts_before[~is_start]

    start_map_native = np.empty(n, int)
    start_map_native[start_order] = events_before[is_start]
    start_map = start_map_native[event_order]

    # a failure with positive weight joins the tied failures before it when
    # the row before it (a failure, as they sort first) has the same time;
    # every other row starts its own set
    prev_time = np.hstack([np.nan, time[:-1]])
    joins = (status_s == 1) & (weight_s != 0) & (time == prev_time)
    new_set = ~joins[~is_start]
    new_set[0] = True
    set_id = np.cumsum(new_set) - 1
    set_first = np.nonzero(new_set)[0]
    set_last = np.hstack([set_first[1:], n]) - 1
    first = set_first[set_id]
    last = set_last[set_id]
    scaling = (np.arange(n) - first) / (last + 1. - first)

    if not np.all(first[start_map] == start_map):
        raise ValueError('first_start disagrees with start_map')

    return dict(event_order=event_order,
                start_order=start_order,
                status=status[event_order],
                first=first,
                last=last,
                scaling=scaling,
                event_map=event_map,
                start_map=start_map,
                have_start_times=have_start_times)


def _forward_cumsum(x):
    return jnp.concatenate([jnp.zeros(1, x.dtype), jnp.cumsum(x)])


def _reverse_cumsum(x):
    return jnp.concatenate([jnp.cumsum(x[::-1])[::-1], jnp.zeros(1, x.dtype)])


def _loglik_sat(sample_weight, prep, efron):
    first, last, status = prep['first'], prep['last'], prep['status']
    w_event = sample_weight[prep['event_order']]
    W = _forward_cumsum(w_event * status)
    W_C = W[last + 1] - W[first]
    K = last + 1 - first
    # once per set of tied failures
    use = (np.arange(first.shape[0]) == first) & (W_C > 0) & (w_event > 0)
    W_safe = jnp.where(use, W_C, 1.)
    # Efron's term: the log of 1 - m / K over the failures with positive weight
    failed = (status == 1) & (w_event > 0)
    P = _forward_cumsum(jnp.where(failed, jnp.log(1 - efron * prep['scaling']), 0.))
    penalty = P[last + 1] - P[first]
    return -jnp.sum(jnp.where(use, W_safe * jnp.log(W_safe) + W_safe / K * penalty, 0.))


def _loglik(eta, sample_weight, prep, efron):
    first, last, status = prep['first'], prep['last'], prep['status']
    event_order, start_order = prep['event_order'], prep['start_order']

    eta = eta - jnp.mean(eta)
    exp_w = sample_weight * jnp.exp(eta)

    event_cumsum = _reverse_cumsum(exp_w[event_order])
    risk_sums = event_cumsum[first]
    if prep['have_start_times']:
        start_cumsum = _reverse_cumsum(exp_w[start_order])
        risk_sums = risk_sums - start_cumsum[prep['event_map']]
    if efron:
        risk_sums = risk_sums - (event_cumsum[first] - event_cumsum[last + 1]) * prep['scaling']

    w_event = sample_weight[event_order]
    W = _forward_cumsum(w_event * status)
    w_avg = jnp.where(status == 1, (W[last + 1] - W[first]) / (last + 1 - first), 0.)

    # the failures with weight 0 (and the censored) contribute nothing; their
    # risk sums may be 0, so they are kept out of the log (and its derivatives)
    use = w_avg > 0
    risk_safe = jnp.where(use, risk_sums, 1.)
    return (jnp.sum(w_event * eta[event_order] * status)
            - jnp.sum(jnp.where(use, w_avg * jnp.log(risk_safe), 0.)))


class CoxDevianceJAX(object):
    """
    The Cox deviance as a JAX function of the linear predictor.

    Parameters
    ----------
    event : np.ndarray
        Event times.
    status : np.ndarray
        1 for a failure, 0 for censored.
    start : np.ndarray, optional
        Start (entry) times of (start, event] data.
    strata : np.ndarray, optional
        Stratum labels.
    sample_weight : np.ndarray, optional
        Defaults to one. Later weights must be 0 where these are.
    tie_breaking : {'efron', 'breslow'}, default='efron'

    Examples
    --------
    >>> jax.config.update('jax_enable_x64', True)
    >>> cox = CoxDevianceJAX(event, status)
    >>> g = jax.grad(cox.deviance)(eta)
    >>> Iu = cox.information_matvec(eta, u)
    >>> dIu = cox.information_derivative_matvec(eta, v, u)
    """

    def __init__(self,
                 event,
                 status,
                 start=None,
                 strata=None,
                 sample_weight=None,
                 tie_breaking='efron'):

        if not jax.config.jax_enable_x64:
            warnings.warn('jax_enable_x64 is off: the deviance and its derivatives '
                          'will be computed in single precision')
        if tie_breaking not in ['efron', 'breslow']:
            raise ValueError("tie_breaking should be 'efron' or 'breslow'")
        self.efron = tie_breaking == 'efron'

        event = np.asarray(event, float).reshape(-1)
        n = event.shape[0]
        status = np.asarray(status).reshape(-1)
        start = None if start is None else np.asarray(start, float).reshape(-1)
        self.sample_weight = (np.ones(n) if sample_weight is None
                              else np.asarray(sample_weight, float).reshape(-1))

        # one pass over all the strata: see preprocess
        self._prep = preprocess(event, status, start, self.sample_weight, strata)

    def _weight(self, sample_weight):
        return self.sample_weight if sample_weight is None else sample_weight

    def loglik(self, linear_predictor, sample_weight=None):
        """The log partial likelihood, the sum over the strata."""
        return _loglik(linear_predictor, self._weight(sample_weight), self._prep, self.efron)

    def loglik_sat(self, sample_weight=None):
        """The log likelihood of the saturated model."""
        return _loglik_sat(self._weight(sample_weight), self._prep, self.efron)

    def deviance(self, linear_predictor, sample_weight=None):
        """``2 * (loglik_sat - loglik)``, as :class:`coxdev.CoxDeviance`."""
        return 2 * (self.loglik_sat(sample_weight)
                    - self.loglik(linear_predictor, sample_weight))

    def gradient(self, linear_predictor, sample_weight=None):
        """The gradient of the deviance."""
        return jax.grad(self.deviance)(linear_predictor, sample_weight)

    def information_matvec(self, linear_predictor, arg, sample_weight=None):
        """The information (minus the Hessian of loglik) times ``arg``."""
        neg_loglik = lambda eta: -self.loglik(eta, sample_weight)
        return jax.jvp(jax.grad(neg_loglik), (linear_predictor,), (arg,))[1]

    def information_derivative_matvec(self, linear_predictor, direction, arg, sample_weight=None):
        """
        ``d/dt I(linear_predictor + t * direction) @ arg`` at ``t = 0``, as
        :meth:`coxdev.CoxDeviance.information_derivative`.
        """
        info = lambda eta: self.information_matvec(eta, arg, sample_weight)
        return jax.jvp(info, (linear_predictor,), (direction,))[1]
