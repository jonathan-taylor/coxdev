"""
The R package (R/coxdev) against the Python package: both run the same C++
code, so the deviance, its derivatives, the information and the draws of the
score (with the same seed) agree to rounding.

Needs rpy2 and the R package installed (R CMD INSTALL R/coxdev); skipped
otherwise, unless COXDEV_REQUIRE_R is set (as in CI), when they fail.
"""

import os

import numpy as np
import pytest

from coxdev import CoxDeviance


def _skip(reason):
    if os.environ.get('COXDEV_REQUIRE_R'):
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope='module')
def R():
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr, PackageNotInstalledError
    except ImportError:
        _skip('rpy2 is not importable')
    try:
        importr('coxdev')
    except PackageNotInstalledError:
        _skip('the R package coxdev is not installed (R CMD INSTALL R/coxdev)')
    return ro


def _to_R(ro, x):
    if x is None:
        return ro.NULL
    x = np.asarray(x)
    if x.ndim == 2:
        return ro.r.matrix(ro.FloatVector(x.ravel(order='F')), nrow=x.shape[0], ncol=x.shape[1])
    if np.issubdtype(x.dtype, np.integer):
        return ro.IntVector(x.tolist())
    return ro.FloatVector(x.astype(float).tolist())


def _to_numpy(value):
    arr = np.array(list(value), float)
    dim = getattr(value, 'dim', None)
    try:
        dim = tuple(dim)
    except TypeError:
        dim = None
    if dim is not None and len(dim) == 2:
        arr = arr.reshape(dim, order='F')
    return arr


class RCox:
    """A CoxDeviance of the R package, called through rpy2."""

    def __init__(self, ro, event, status, start=None, strata=None, sample_weight=None,
                 tie_breaking='efron'):
        self.ro = ro
        ro.globalenv['cox_'] = ro.r('coxdev::CoxDeviance$new')(
            event=_to_R(ro, event), status=_to_R(ro, status), start=_to_R(ro, start),
            strata=_to_R(ro, strata), sample_weight=_to_R(ro, sample_weight),
            tie_breaking=tie_breaking)
        self.obj = ro.globalenv['cox_']

    def _method(self, name):
        return self.ro.r('function(obj, name) obj[[name]]')(self.obj, name)

    def compute(self, eta, w=None):
        res = self._method('compute')(_to_R(self.ro, eta), _to_R(self.ro, w))
        get = lambda k: _to_numpy(res.rx2(k))
        return {'deviance': get('deviance')[0], 'loglik_sat': get('loglik_sat')[0],
                'gradient': get('gradient'), 'diag_hessian': get('diag_hessian')}

    def information_matmat(self, eta, w, V):
        info = self._method('information')(_to_R(self.ro, eta), _to_R(self.ro, w))
        multiply = self.ro.r('function(I, v) I$multiply(v)')
        return _to_numpy(multiply(info, _to_R(self.ro, V)))

    def sample_score(self, eta, w, method, n_draws, seed):
        out = self._method('sample_score')(_to_R(self.ro, eta), _to_R(self.ro, w),
                                           method=method, n_draws=n_draws,
                                           seed=str(seed))
        return _to_numpy(out)

    def wild_score(self, eta, G, w=None):
        return _to_numpy(self._method('wild_score')(_to_R(self.ro, eta), _to_R(self.ro, G),
                                                    _to_R(self.ro, w)))


def _data(n, start, ties, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    eta = X @ np.array([0.5, -0.5, 0.25])
    event = rng.exponential(1 / np.exp(eta))
    status = rng.binomial(1, 0.7, n)
    if ties:
        event = np.ceil(event * 4) / 4
    entry = None
    if start:
        entry = rng.uniform(0, 0.5, n) * event
        if ties:
            entry = np.floor(entry * 8) / 8
    return rng, X, eta, event, status, entry


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('start', [False, True])
@pytest.mark.parametrize('ties', [False, True])
@pytest.mark.parametrize('weights', ['ones', 'unequal', 'zeros'])
@pytest.mark.parametrize('stratified', [False, True])
def test_deviance_and_information(R, tie_breaking, start, ties, weights, stratified):
    rng, X, eta, event, status, entry = _data(80, start, ties, seed=0)
    n = len(event)
    w = np.ones(n) if weights == 'ones' else rng.uniform(0.5, 2, n)
    if weights == 'zeros':
        w[:8] = 0
    strata = np.arange(n) % 3 if stratified else None

    py = CoxDeviance(event=event, status=status, start=entry, strata=strata,
                     sample_weight=w, tie_breaking=tie_breaking)
    r = RCox(R, event, status, start=entry, strata=strata, sample_weight=w,
             tie_breaking=tie_breaking)

    P = py(eta, w)
    Q = r.compute(eta, w)
    np.testing.assert_allclose(Q['deviance'], P.deviance, rtol=1e-12)
    np.testing.assert_allclose(Q['loglik_sat'], P.loglik_sat, rtol=1e-12)
    np.testing.assert_allclose(Q['gradient'], P.gradient, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(Q['diag_hessian'], P.diag_hessian, rtol=1e-12, atol=1e-14)

    V = np.column_stack([X, rng.standard_normal(n)])
    I = py.information(eta, w)
    np.testing.assert_allclose(r.information_matmat(eta, w, V), I @ V, rtol=1e-12, atol=1e-13)

    G = rng.standard_normal(n)
    np.testing.assert_allclose(r.wild_score(eta, G, w), py.wild_score(eta, G, sample_weight=w),
                               rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize('method', ['multinomial', 'bernoulli', 'fisher', 'wild'])
@pytest.mark.parametrize('start', [False, True])
@pytest.mark.parametrize('stratified', [False, True])
@pytest.mark.parametrize('seed', [0, 2024, 2**64 - 1])
def test_sample_score_same_draws(R, method, start, stratified, seed):
    rng, X, eta, event, status, entry = _data(100, start, True, seed=1)
    n = len(event)
    strata = np.arange(n) % 3 if stratified else None
    # the model-based methods need equal weights; zero weights drop out
    w = np.ones(n)
    w[:5] = 0

    py = CoxDeviance(event=event, status=status, start=entry, strata=strata, sample_weight=w)
    r = RCox(R, event, status, start=entry, strata=strata, sample_weight=w)
    draws_py = py.sample_score(eta, sample_weight=w, method=method, n_draws=7, seed=seed)
    draws_r = r.sample_score(eta, w, method, 7, seed)
    assert draws_r.shape == (n, 7)
    np.testing.assert_allclose(draws_r, draws_py, rtol=1e-12, atol=1e-13)
