"""
Tests of CoxDeviance.sample_score and CoxDeviance.wild_score.
"""

import numpy as np
import pytest

from coxdev import CoxDeviance

METHODS = ['multinomial', 'bernoulli', 'fisher', 'wild']


def _data(n=150, p=3, start=False, ties=True, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.linspace(0.5, -0.5, p)
    T = rng.exponential(1 / np.exp(X @ beta))
    C = rng.exponential(1.5, n)
    event = np.minimum(T, C)
    status = (T <= C).astype(int)
    if ties:
        event = np.ceil(event * 4) / 4          # a coarse grid: many ties
    entry = None
    if start:
        entry = rng.uniform(0, 0.5, n) * event  # delayed entry, before the event
        if ties:
            entry = np.floor(entry * 8) / 8
    return X, event, status, entry


def reference_residuals(X, eta, event, status, entry, weight, efron):
    """
    Score residuals by direct computation: U_i = sum over the failure terms t of
    (w_i 1{i in D_t} / d - wbar p_ti) (x_i - xbar_t), with Efron's weights for ties.
    """
    n = len(event)
    start = -np.inf * np.ones(n) if entry is None else entry
    e = weight * np.exp(eta - eta.max())
    U = np.zeros_like(X)
    for t in np.unique(event[(status == 1) & (weight > 0)]):
        D = (event == t) & (status == 1) & (weight > 0)
        R = (event >= t) & (start < t) & (weight > 0)
        d = D.sum()
        wbar = weight[D].mean()
        for r in range(d):
            s = r / d if efron else 0.
            w_r = e * (R - s * D)
            p_r = w_r / w_r.sum()
            xbar = p_r @ X
            U += ((weight * D / d) - wbar * p_r)[:, None] * (X - xbar)
    return U


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('start', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_wild_score_matches_residuals(tie_breaking, start, weighted):
    X, event, status, entry = _data(start=start)
    n = len(event)
    rng = np.random.default_rng(1)
    weight = rng.uniform(0.5, 2, n) if weighted else np.ones(n)
    eta = X @ np.array([0.3, -0.2, 0.1])
    cox = CoxDeviance(event=event, status=status, start=entry, tie_breaking=tie_breaking,
                      sample_weight=weight)

    U = reference_residuals(X, eta, event, status, entry, weight, tie_breaking == 'efron')
    # with multipliers 1 the draw is the score
    ones = cox.wild_score(eta, np.ones(n), sample_weight=weight)
    score = -cox(eta, weight).gradient / 2
    np.testing.assert_allclose(ones, score, atol=1e-10)
    np.testing.assert_allclose(X.T @ ones, U.sum(0), atol=1e-8)
    # with other multipliers, sum_i G_i U_i
    G = rng.standard_normal(n)
    draw = cox.wild_score(eta, G, sample_weight=weight)
    np.testing.assert_allclose(X.T @ draw, G @ U, atol=1e-8)


@pytest.mark.parametrize('start', [False, True])
@pytest.mark.parametrize('method', METHODS)
def test_draws_sum_to_zero_and_reproducible(method, start):
    # every draw is centered within the risk sets, so it sums to 0
    X, event, status, entry = _data(start=start)
    n = len(event)
    eta = X @ np.array([0.3, -0.2, 0.1])
    cox = CoxDeviance(event=event, status=status, start=entry)
    draws = cox.sample_score(eta, method=method, n_draws=20, seed=3)
    assert draws.shape == (n, 20)
    np.testing.assert_allclose(draws.sum(0), 0, atol=1e-9)
    np.testing.assert_array_equal(draws, cox.sample_score(eta, method=method, n_draws=20, seed=3))
    assert not np.allclose(draws, cox.sample_score(eta, method=method, n_draws=20, seed=4))
    one = cox.sample_score(eta, method=method, seed=3)
    assert one.shape == (n,)
    # without a seed, a fresh one each time
    assert not np.allclose(cox.sample_score(eta, method=method), cox.sample_score(eta, method=method))


def _breslow_information(X, eta, event, status, entry, weight):
    cox = CoxDeviance(event=event, status=status, start=entry, tie_breaking='breslow',
                      sample_weight=weight)
    I = cox.information(eta, weight)
    return X.T @ np.column_stack([I @ X[:, j] for j in range(X.shape[1])])


@pytest.mark.parametrize('start', [False, True])
def test_multinomial_covariance_is_breslow_information(start):
    X, event, status, entry = _data(n=200, start=start)
    n = len(event)
    weight = np.ones(n) / n      # equal weights, as glmstar uses
    eta = X @ np.array([0.3, -0.2, 0.1])
    cox = CoxDeviance(event=event, status=status, start=entry, sample_weight=weight)
    draws = X.T @ cox.sample_score(eta, sample_weight=weight, method='multinomial',
                                     n_draws=6000, seed=1)
    information = _breslow_information(X, eta, event, status, entry, weight)
    # the score of the weighted loss has covariance w0^2 / w0 = w0 times its Hessian
    target = information / n
    np.testing.assert_allclose(draws.mean(1), 0, atol=4 * np.sqrt(np.diag(target) / 6000).max())
    np.testing.assert_allclose(np.cov(draws), target, rtol=0.08, atol=0.08 * np.diag(target).max())


def test_bernoulli_has_less_variance():
    # heavy ties relative to the risk sets: independent failures remove variance
    X, event, status, entry = _data(n=200)
    eta = X @ np.array([0.3, -0.2, 0.1])
    cox = CoxDeviance(event=event, status=status)
    draws = X.T @ cox.sample_score(eta, method='bernoulli', n_draws=6000, seed=2)
    information = _breslow_information(X, eta, event, status, None, np.ones(len(event)))
    ratio = np.diag(np.cov(draws)) / np.diag(information)
    assert np.all(ratio < 1) and np.all(ratio > 0.5)


def _log_esp(loge, d):
    # log elementary symmetric polynomials E_0..E_d of exp(loge)
    L = np.full(d + 1, -np.inf)
    L[0] = 0.
    for v in loge:
        L[1:] = np.logaddexp(L[1:], v + L[:-1])
    return L


def _exact_loglik_normalizer(eta, event, status, entry):
    # sum over failure times of log E_d(exp(eta) over the risk set): the
    # log-normalizer of the exact discrete partial likelihood
    total = 0.
    for t in np.unique(event[status == 1]):
        d = ((event == t) & (status == 1)).sum()
        risk = (event >= t) & ((entry < t) if entry is not None else True)
        total += _log_esp(eta[risk], d)[d]
    return total


@pytest.mark.parametrize('start', [False, True])
def test_fisher_moments_are_exact_likelihood(start):
    # the mean and covariance of X^T r are the gradient (centered) and Hessian
    # of the exact log-normalizer sum_k log E_{d_k}(exp(eta + X t)) at t = 0
    X, event, status, entry = _data(n=200, start=start)
    eta = X @ np.array([0.3, -0.2, 0.1])
    cox = CoxDeviance(event=event, status=status, start=entry)
    draws = X.T @ cox.sample_score(eta, method='fisher', n_draws=20000, seed=7)

    f = lambda t: _exact_loglik_normalizer(eta + X @ t, event, status, entry)
    p, h = X.shape[1], 1e-3
    I = np.eye(p) * h
    H = np.array([[(f(I[i] + I[j]) - f(I[i] - I[j]) - f(-I[i] + I[j]) + f(-I[i] - I[j])) / (4 * h**2)
                   for j in range(p)] for i in range(p)])
    np.testing.assert_allclose(draws.mean(1), 0, atol=4 * np.sqrt(np.diag(H) / 20000).max())
    np.testing.assert_allclose(np.cov(draws), H, rtol=0.04, atol=0.04 * np.diag(H).max())
    # less than the Breslow information when ties are heavy
    breslow = _breslow_information(X, eta, event, status, entry, np.ones(len(event)))
    assert np.all(np.diag(H) < np.diag(breslow))


def test_fisher_exact_distribution():
    # one failure time, d = 2 of m = 5 at risk: P(A) is proportional to the
    # product of exp(eta) over A
    from itertools import combinations
    from scipy.stats import chisquare
    eta = np.array([0.8, -0.3, 0.2, 1.1, -1.0])
    event = np.ones(5)
    status = np.array([1, 1, 0, 0, 0])
    cox = CoxDeviance(event=event, status=status)
    draws = cox.sample_score(eta, method='fisher', n_draws=100000, seed=11)
    sets = list(combinations(range(5), 2))
    prob = np.array([np.exp(eta[list(A)].sum()) for A in sets])
    prob /= prob.sum()
    # the exact centering: E delta_j is the probability j is in A
    mean = np.array([prob[[j in A for A in sets]].sum() for j in range(5)])
    delta = draws + mean[:, None]
    np.testing.assert_allclose(delta, np.round(delta), atol=1e-10)
    np.testing.assert_allclose(delta.sum(0), 2)
    index = {A: i for i, A in enumerate(sets)}
    drawn = [index[tuple(np.flatnonzero(np.round(col)))] for col in delta.T]
    counts = np.bincount(drawn, minlength=len(sets))
    assert chisquare(counts, prob * draws.shape[1]).pvalue > 1e-3


def test_fisher_all_at_risk_fail():
    # every subject at risk fails: the failure set is fixed and the draws are 0
    cox = CoxDeviance(event=np.ones(3), status=np.ones(3, int))
    draws = cox.sample_score(np.array([0.5, -0.2, 0.1]), method='fisher', n_draws=5, seed=0)
    np.testing.assert_allclose(draws, 0, atol=1e-12)


@pytest.mark.parametrize('method', ['multinomial', 'bernoulli', 'fisher'])
def test_no_ties_same_mean_and_covariance(method):
    # with one failure at each event time the model-based methods draw the same law
    X, event, status, entry = _data(n=150, ties=False)
    eta = X @ np.array([0.3, -0.2, 0.1])
    cox = CoxDeviance(event=event, status=status)
    draws = X.T @ cox.sample_score(eta, method=method, n_draws=6000, seed=3)
    information = _breslow_information(X, eta, event, status, None, np.ones(len(event)))
    np.testing.assert_allclose(np.cov(draws), information, rtol=0.08, atol=0.08 * np.diag(information).max())


def test_wild_covariance_is_sandwich():
    X, event, status, entry = _data(n=200)
    eta = X @ np.array([0.3, -0.2, 0.1])
    cox = CoxDeviance(event=event, status=status)
    draws = X.T @ cox.sample_score(eta, method='wild', n_draws=6000, seed=5)
    U = reference_residuals(X, eta, event, status, None, np.ones(len(event)), True)
    sandwich = U.T @ U
    np.testing.assert_allclose(np.cov(draws), sandwich, rtol=0.08, atol=0.08 * np.diag(sandwich).max())


def test_not_at_risk_is_zero():
    # a subject who enters after the last failure time is never at risk
    X, event, status, entry = _data(start=True)
    last = event[status == 1].max()
    event = np.r_[event, last + 2]
    status = np.r_[status, 0]
    entry = np.r_[entry, last + 1]
    eta = np.zeros(len(event))
    cox = CoxDeviance(event=event, status=status, start=entry)
    for method in METHODS:
        draws = cox.sample_score(eta, method=method, n_draws=50, seed=0)
        np.testing.assert_array_equal(draws[-1], 0)


def test_unequal_weights():
    X, event, status, entry = _data()
    n = len(event)
    weight = np.random.default_rng(0).uniform(0.5, 2, n)
    eta = X @ np.array([0.3, -0.2, 0.1])
    cox = CoxDeviance(event=event, status=status, sample_weight=weight)
    with pytest.raises(RuntimeError, match='equal positive weights'):
        cox.sample_score(eta, sample_weight=weight, method='multinomial')
    cox.sample_score(eta, sample_weight=weight, method='wild')
    # weights 0 drop out
    weight = np.ones(n)
    weight[:10] = 0
    cox = CoxDeviance(event=event, status=status, sample_weight=weight)
    draws = cox.sample_score(eta, sample_weight=weight, method='multinomial', n_draws=20)
    np.testing.assert_array_equal(draws[:10], 0)


def test_stratified():
    X, event, status, entry = _data(n=200)
    n = len(event)
    strata = np.arange(n) % 3
    eta = X @ np.array([0.3, -0.2, 0.1])
    cox = CoxDeviance(event=event, status=status, strata=strata)
    np.testing.assert_allclose(cox.wild_score(eta, np.ones(n)), -cox(eta).gradient / 2, atol=1e-10)
    for method in METHODS:
        draws = cox.sample_score(eta, method=method, n_draws=10, seed=0)
        for s in range(3):
            np.testing.assert_allclose(draws[strata == s].sum(0), 0, atol=1e-9)
    # each stratum is sampled within itself, with its own risk sets
    single = CoxDeviance(event=event[strata == 0], status=status[strata == 0])
    U = reference_residuals(X[strata == 0], eta[strata == 0], event[strata == 0],
                            status[strata == 0], None, np.ones((strata == 0).sum()), True)
    G = np.random.default_rng(0).standard_normal(n)
    np.testing.assert_allclose(X[strata == 0].T @ cox.wild_score(eta, G)[strata == 0],
                               G[strata == 0] @ U, atol=1e-8)
