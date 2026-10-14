"""
Weights that are a positive multiple c of those of a CoxDeviance are computed
with its own weights, the results multiplied by c (and the saturated
log-likelihood shifted), without rebuilding the object.
"""

import numpy as np
import pytest

from coxdev import CoxDeviance


def _data(seed, start, stratified, weights):
    rng = np.random.default_rng(seed)
    n = 60
    event = np.ceil(rng.exponential(1, n) * 3) / 3      # a coarse grid: many ties
    status = rng.binomial(1, 0.7, n)
    eta = rng.standard_normal(n) * 0.3
    w = np.ones(n) if weights == 'ones' else rng.uniform(0.5, 2, n)
    if weights == 'zeros':
        w[:5] = 0
    entry = np.floor(rng.uniform(0, 0.5, n) * event * 6) / 6 if start else None
    strata = np.arange(n) % 2 if stratified else None
    return dict(event=event, status=status, start=entry, strata=strata), eta, w


def _check_same(scaled, rebuilt, eta, w):
    """scaled at the weights w, against a CoxDeviance built with w."""
    n = eta.shape[0]
    a, b = scaled(eta, w), rebuilt(eta, w)
    np.testing.assert_allclose(a.deviance, b.deviance, rtol=1e-10)
    np.testing.assert_allclose(a.loglik_sat, b.loglik_sat, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(a.gradient, b.gradient, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a.diag_hessian, b.diag_hessian, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a.sample_weight, w, rtol=1e-12)
    v = np.linspace(-1, 1, n)
    np.testing.assert_allclose(scaled.information(eta, w) @ v, rebuilt.information(eta, w) @ v,
                               rtol=1e-10, atol=1e-12)
    G = np.cos(np.arange(n))
    np.testing.assert_allclose(scaled.wild_score(eta, G, w), rebuilt.wild_score(eta, G, w),
                               rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('start', [False, True])
@pytest.mark.parametrize('weights', ['ones', 'unequal', 'zeros'])
@pytest.mark.parametrize('stratified', [False, True])
@pytest.mark.parametrize('scale', [1 / 60, 3.7])
def test_scaled_weights(tie_breaking, start, weights, stratified, scale):
    data, eta, w = _data(0, start, stratified, weights)
    scaled = CoxDeviance(tie_breaking=tie_breaking, sample_weight=w, **data)
    rebuilt = CoxDeviance(tie_breaking=tie_breaking, sample_weight=scale * w, **data)
    _check_same(scaled, rebuilt, eta, scale * w)
    # and back at its own weights
    _check_same(scaled, CoxDeviance(tie_breaking=tie_breaking, sample_weight=w, **data), eta, w)


@pytest.mark.parametrize('method', ['multinomial', 'bernoulli', 'fisher', 'wild'])
@pytest.mark.parametrize('scale', [1 / 60, 3.7])
def test_scaled_sample_score(method, scale):
    # the model-based methods need equal weights
    data, eta, w = _data(1, True, False, 'ones')
    scaled = CoxDeviance(sample_weight=w, **data)
    rebuilt = CoxDeviance(sample_weight=scale * w, **data)
    np.testing.assert_allclose(scaled.sample_score(eta, scale * w, method=method, n_draws=3, seed=4),
                               rebuilt.sample_score(eta, scale * w, method=method, n_draws=3, seed=4),
                               rtol=1e-10, atol=1e-12)


def test_no_rebuild(monkeypatch):
    data, eta, w = _data(2, True, True, 'zeros')
    other = w.copy()
    other[10] *= 2        # not a multiple of w
    zeros = other.copy()
    zeros[11] = 0         # and with another zero weight
    # alternating between the weights and their normalization, at two linear predictors
    calls = [(eta, w), (eta, w / w.sum()), (2 * eta, w / w.sum()), (eta, w), (2 * eta, 7 * w)]
    expected = [CoxDeviance(sample_weight=weights, **data)(e, weights) for e, weights in calls]
    expected_other = [CoxDeviance(sample_weight=weights, **data)(eta, weights)
                      for weights in [other, zeros]]

    cox = CoxDeviance(sample_weight=w, **data)
    rebuilds = []
    reconstruct = CoxDeviance._reconstruct_coxc

    def counting(self, sample_weight):
        rebuilds.append(1)
        return reconstruct(self, sample_weight)

    monkeypatch.setattr(CoxDeviance, '_reconstruct_coxc', counting)

    for (e, weights), b in zip(calls, expected):
        a = cox(e, weights)
        np.testing.assert_allclose(a.deviance, b.deviance, rtol=1e-10)
        np.testing.assert_allclose(a.gradient, b.gradient, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(a.diag_hessian, b.diag_hessian, rtol=1e-10, atol=1e-12)
    assert not rebuilds

    # weights that are not a multiple rebuild, as do weights with other zeros
    for k, (weights, b) in enumerate(zip([other, zeros], expected_other)):
        np.testing.assert_allclose(cox(eta, weights).deviance, b.deviance, rtol=1e-10)
        assert len(rebuilds) == k + 1


def test_default_weights():
    # None means weights of one, whatever the weights at construction
    data, eta, w = _data(3, False, False, 'unequal')
    cox = CoxDeviance(sample_weight=w, **data)
    np.testing.assert_allclose(cox(eta).deviance, CoxDeviance(**data)(eta).deviance, rtol=1e-10)
    np.testing.assert_allclose(cox(eta, w).deviance,
                               CoxDeviance(sample_weight=w, **data)(eta, w).deviance, rtol=1e-10)
