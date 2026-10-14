"""
Tests of CoxDeviance.sample_score and CoxDeviance.wild_score.
"""

import numpy as np
import pytest

from coxdev import CoxDeviance

METHODS = ['multinomial', 'multinomial_centered', 'bernoulli', 'wallenius', 'wild']


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
@pytest.mark.parametrize('method', ['multinomial', 'multinomial_centered'])
def test_multinomial_covariance_is_breslow_information(method, start):
    X, event, status, entry = _data(n=200, start=start)
    n = len(event)
    weight = np.ones(n) / n      # equal weights, as glmstar uses
    eta = X @ np.array([0.3, -0.2, 0.1])
    cox = CoxDeviance(event=event, status=status, start=entry, sample_weight=weight)
    draws = X.T @ cox.sample_score(eta, sample_weight=weight, method=method,
                                     n_draws=6000, seed=1)
    information = _breslow_information(X, eta, event, status, entry, weight)
    # the score of the weighted loss has covariance w0^2 / w0 = w0 times its Hessian
    target = information / n
    np.testing.assert_allclose(draws.mean(1), 0, atol=4 * np.sqrt(np.diag(target) / 6000).max())
    np.testing.assert_allclose(np.cov(draws), target, rtol=0.08, atol=0.08 * np.diag(target).max())


@pytest.mark.parametrize('method', ['bernoulli', 'wallenius'])
def test_without_replacement_has_less_variance(method):
    # heavy ties relative to the risk sets: drawing without replacement removes variance
    X, event, status, entry = _data(n=200)
    eta = X @ np.array([0.3, -0.2, 0.1])
    cox = CoxDeviance(event=event, status=status)
    draws = X.T @ cox.sample_score(eta, method=method, n_draws=6000, seed=2)
    information = _breslow_information(X, eta, event, status, None, np.ones(len(event)))
    ratio = np.diag(np.cov(draws)) / np.diag(information)
    assert np.all(ratio < 1) and np.all(ratio > 0.5)
    np.testing.assert_allclose(draws.mean(1), 0, atol=4 * np.sqrt(np.diag(information) / 6000).max())


def test_wild_covariance_is_sandwich():
    X, event, status, entry = _data(n=200)
    eta = X @ np.array([0.3, -0.2, 0.1])
    cox = CoxDeviance(event=event, status=status)
    draws = X.T @ cox.sample_score(eta, method='wild', n_draws=6000, seed=5)
    U = reference_residuals(X, eta, event, status, None, np.ones(len(event)), True)
    sandwich = U.T @ U
    np.testing.assert_allclose(np.cov(draws), sandwich, rtol=0.08, atol=0.08 * np.diag(sandwich).max())


def test_wallenius_draws_each_failure_once():
    # one failure time with d failures: each draw picks d distinct subjects
    rng = np.random.default_rng(0)
    n, d = 30, 12
    event = np.ones(n)
    status = np.r_[np.ones(d), np.zeros(n - d)].astype(int)
    eta = rng.standard_normal(n)
    cox = CoxDeviance(event=event, status=status)
    draws = cox.sample_score(eta, method='wallenius', n_draws=200, seed=0) * np.sqrt(2)
    assert set(np.round(np.unique(draws), 12)) <= {-1., 0., 1.}
    assert np.all((draws > 0).sum(0) <= d)
    # with replacement, a subject can be drawn more than once
    multinomial = cox.sample_score(eta, method='multinomial', n_draws=200, seed=0) * np.sqrt(2)
    assert np.abs(multinomial).max() > 1


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
