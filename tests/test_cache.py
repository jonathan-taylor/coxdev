"""
CoxDeviance rebuilds its C++ object only for new weights, and reuses its
result at the linear predictor of the last call, comparing the arrays.
"""

import numpy as np
import pytest

from coxdev import CoxDeviance


class _Counting:
    """The C++ object, counting the calls of compute_deviance."""

    def __init__(self, coxc):
        self._coxc = coxc
        self.computes = 0

    def compute_deviance(self, *args):
        self.computes += 1
        return self._coxc.compute_deviance(*args)

    def __getattr__(self, name):
        return getattr(self._coxc, name)


def _data(seed=0, stratified=False):
    rng = np.random.default_rng(seed)
    n = 50
    event = np.ceil(rng.exponential(1, n) * 3) / 3      # a coarse grid: many ties
    status = rng.binomial(1, 0.7, n)
    entry = np.floor(rng.uniform(0, 0.5, n) * event * 6) / 6
    strata = np.arange(n) % 2 if stratified else None
    w = rng.uniform(0.5, 2, n)
    w[:4] = 0
    return dict(event=event, status=status, start=entry, strata=strata), rng.standard_normal(n) * 0.3, w


@pytest.fixture
def rebuilds(monkeypatch):
    count = [0]
    reconstruct = CoxDeviance._reconstruct_coxc

    def counting(self, sample_weight):
        count[0] += 1
        reconstruct(self, sample_weight)
        self._coxc = _Counting(self._coxc)

    monkeypatch.setattr(CoxDeviance, '_reconstruct_coxc', counting)
    return count


@pytest.mark.parametrize('stratified', [False, True])
def test_reuse(rebuilds, stratified):
    data, eta, w = _data(stratified=stratified)
    cox = CoxDeviance(sample_weight=w, **data)
    assert rebuilds[0] == 1

    first = cox(eta, w).deviance
    # equal arrays, not the same objects: no rebuild, and the result is reused
    assert cox(eta.copy(), w.copy()).deviance == first
    assert rebuilds[0] == 1 and cox._coxc.computes == 1

    # a new linear predictor is computed
    cox(2 * eta, w)
    assert rebuilds[0] == 1 and cox._coxc.computes == 2


@pytest.mark.parametrize('stratified', [False, True])
def test_changes(rebuilds, stratified):
    data, eta, w = _data(seed=1, stratified=stratified)
    fresh = lambda weights: CoxDeviance(sample_weight=weights, **data)
    other = w.copy()
    other[10] *= 2
    calls = [(eta, w), (eta, other), (eta, w / w.sum()), (2 * eta, w / w.sum()), (eta, None)]
    expected = [fresh(weights)(e, weights) for e, weights in calls]

    cox = fresh(w)
    rebuilds[0] = 0
    for (e, weights), b in zip(calls, expected):
        a = cox(e, weights)
        np.testing.assert_allclose(a.deviance, b.deviance, rtol=1e-12)
        np.testing.assert_allclose(a.gradient, b.gradient, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(a.diag_hessian, b.diag_hessian, rtol=1e-12, atol=1e-14)
    # new weights at each call but the fourth
    assert rebuilds[0] == 3


def test_modified_in_place():
    # the caller's arrays are compared against copies: changing them in place is seen
    data, eta, w = _data(seed=2)
    cox = CoxDeviance(sample_weight=w, **data)
    weights = w.copy()
    e = eta.copy()
    cox(e, weights)
    e *= 2
    np.testing.assert_allclose(cox(e, weights).deviance,
                               CoxDeviance(sample_weight=w, **data)(2 * eta, w).deviance, rtol=1e-12)
    weights[10] = 0
    np.testing.assert_allclose(cox(e, weights).deviance,
                               CoxDeviance(sample_weight=weights, **data)(2 * eta, weights).deviance,
                               rtol=1e-12)
