"""
The C API (cpp/include/coxdev_c_api.h) against the Python package: the same
C++ underneath, so the results agree to rounding and the draws exactly.

Needs the shared library: build it with cpp/capi/build_local.sh, or point
COXDEV_C_LIB at it. Skipped if it is not found.
"""
import ctypes as C
import os
from glob import glob
from itertools import product

import numpy as np
import pytest

from coxdev import CoxDeviance

HERE = os.path.dirname(__file__)


def _find_lib():
    path = os.environ.get('COXDEV_C_LIB')
    if path:
        return path
    found = glob(os.path.join(HERE, '..', 'cpp', 'capi', 'build', 'libcoxdev_c.*'))
    return found[0] if found else None


LIBPATH = _find_lib()
pytestmark = pytest.mark.skipif(LIBPATH is None, reason='libcoxdev_c not built (cpp/capi/build_local.sh)')

dp = C.POINTER(C.c_double)
ip = C.POINTER(C.c_int)


def _lib():
    lib = C.CDLL(LIBPATH)
    lib.coxdev_last_error.restype = C.c_char_p
    lib.coxdev_new.argtypes = [C.c_size_t, dp, dp, ip, ip, dp, C.c_int, C.POINTER(C.c_void_p)]
    lib.coxdev_free.argtypes = [C.c_void_p]
    lib.coxdev_n.argtypes = [C.c_void_p]
    lib.coxdev_n.restype = C.c_size_t
    lib.coxdev_deviance.argtypes = [C.c_void_p, dp, dp, dp, dp, dp, dp]
    lib.coxdev_hessian_matvec.argtypes = [C.c_void_p, dp, dp, dp, dp]
    lib.coxdev_baseline_hazard.argtypes = [C.c_void_p, dp, dp, C.POINTER(C.c_size_t), dp, dp, dp, ip]
    lib.coxdev_handle_concordance.argtypes = [C.c_void_p, dp, dp, dp, dp, dp]
    lib.coxdev_concordance.argtypes = [C.c_size_t, dp, dp, dp, ip, dp, dp, dp, dp]
    lib.coxdev_score_residuals.argtypes = [C.c_void_p, dp, dp, dp, C.c_size_t, dp]
    lib.coxdev_wild_score.argtypes = [C.c_void_p, dp, dp, dp, dp]
    lib.coxdev_sample_score.argtypes = [C.c_void_p, dp, dp, C.c_char_p, C.c_size_t, C.c_uint64, dp]
    return lib


LIB = _lib() if LIBPATH else None


def _d(x):
    return None if x is None else np.ascontiguousarray(x, float).ctypes.data_as(dp)


def _i(x):
    return None if x is None else np.ascontiguousarray(x, np.int32).ctypes.data_as(ip)


class Handle:
    """A thin ctypes wrapper of a coxdev_t, keeping its inputs alive."""

    def __init__(self, event, status, start=None, strata=None, weight=None, efron=True):
        self.n = event.shape[0]
        self._keep = [np.ascontiguousarray(a, t) if a is not None else None
                      for a, t in [(event, float), (status, np.int32), (start, float),
                                   (strata, np.int32), (weight, float)]]
        event, status, start, strata, weight = self._keep
        self.h = C.c_void_p()
        check(LIB.coxdev_new(self.n, _d(start), _d(event), _i(status), _i(strata), _d(weight),
                             int(efron), C.byref(self.h)))

    def __del__(self):
        if LIB is not None and getattr(self, 'h', None):
            LIB.coxdev_free(self.h)


def check(rc):
    if rc != 0:
        raise RuntimeError(f'coxdev C API error {rc}: {LIB.coxdev_last_error().decode()}')


def make_data(n, start_times, strata, weights, rng):
    # times on a coarse grid, so that there are ties among the events and
    # between events and starts
    event = rng.integers(1, n // 3 + 2, size=n).astype(float)
    status = rng.binomial(1, 0.7, size=n)
    status[0] = 1
    start = None
    if start_times:
        start = np.floor(event * rng.uniform(0, 1, size=n))
        tie = rng.uniform(size=n) < 0.3
        start[tie] = event[tie] - 1  # starts tied with other event times
        start = np.minimum(start, event - 1)
    strata_ = rng.integers(0, 3, size=n) if strata else None
    if weights == 'unit':
        w = np.ones(n)
    elif weights == 'random':
        w = rng.uniform(0.5, 2, size=n)
    else:  # zeros
        w = rng.uniform(0.5, 2, size=n)
        w[rng.choice(n, n // 5, replace=False)] = 0
    eta = rng.normal(scale=0.5, size=n)
    return event, status, start, strata_, w, eta


CASES = list(product(['efron', 'breslow'], [False, True], [False, True], ['unit', 'random', 'zeros']))


@pytest.mark.parametrize('ties,start_times,strata,weights', CASES)
def test_against_python(ties, start_times, strata, weights):
    rng = np.random.default_rng(abs(hash((ties, start_times, strata, weights))) % 2**32)
    n = 60
    event, status, start, strata_, w, eta = make_data(n, start_times, strata, weights, rng)
    cd = CoxDeviance(event=event, status=status, start=start, strata=strata_,
                     sample_weight=w, tie_breaking=ties)
    h = Handle(event, status, start=start, strata=strata_, weight=w, efron=(ties == 'efron'))
    assert LIB.coxdev_n(h.h) == n
    tol = dict(rtol=1e-13, atol=1e-13)

    # deviance, gradient, diagonal Hessian, saturated log-likelihood
    res = cd(eta, w)
    dev, sat = C.c_double(), C.c_double()
    grad, diag = np.empty(n), np.empty(n)
    check(LIB.coxdev_deviance(h.h, _d(eta), _d(w), C.byref(dev), grad.ctypes.data_as(dp),
                              diag.ctypes.data_as(dp), C.byref(sat)))
    np.testing.assert_allclose(dev.value, res.deviance, **tol)
    np.testing.assert_allclose(sat.value, res.loglik_sat, **tol)
    np.testing.assert_allclose(grad, res.gradient, **tol)
    np.testing.assert_allclose(diag, res.diag_hessian, **tol)

    # information times vectors
    info = cd.information(eta, w)
    for v in [np.ones(n), rng.normal(size=n)]:
        out = np.empty(n)
        check(LIB.coxdev_hessian_matvec(h.h, _d(eta), _d(w), _d(v), out.ctypes.data_as(dp)))
        np.testing.assert_allclose(out, info @ v, **tol)

    # baseline hazard: size query, then fill
    bh = cd.baseline_hazard(eta, w)
    m = C.c_size_t(0)
    check(LIB.coxdev_baseline_hazard(h.h, _d(eta), _d(w), C.byref(m), None, None, None, None))
    t, hz, ch, st = np.empty(m.value), np.empty(m.value), np.empty(m.value), np.empty(m.value, np.int32)
    check(LIB.coxdev_baseline_hazard(h.h, _d(eta), _d(w), C.byref(m), t.ctypes.data_as(dp),
                                     hz.ctypes.data_as(dp), ch.ctypes.data_as(dp), st.ctypes.data_as(ip)))
    np.testing.assert_allclose(t, bh.time, **tol)
    np.testing.assert_allclose(hz, bh.hazard, **tol)
    np.testing.assert_allclose(ch, bh.cumulative_hazard, **tol)
    # the C API reports the strata labels as given, as Python does (0 without strata)
    np.testing.assert_array_equal(st, bh.stratum)

    # concordance: with the handle (within strata), and the free function (no strata)
    num, den, val = C.c_double(), C.c_double(), C.c_double()
    check(LIB.coxdev_handle_concordance(h.h, _d(eta), _d(w), C.byref(num), C.byref(den), C.byref(val)))
    np.testing.assert_allclose(val.value, cd.c_index(eta, w), **tol)
    if strata_ is None:
        status32 = np.ascontiguousarray(status, np.int32)
        check(LIB.coxdev_concordance(n, _d(eta), _d(start), _d(event), status32.ctypes.data_as(ip), _d(w),
                                     C.byref(num), C.byref(den), C.byref(val)))
        np.testing.assert_allclose(val.value, cd.c_index(eta, w), **tol)

    # score residuals of a design (column-major)
    X = np.asfortranarray(rng.normal(size=(n, 3)))
    U = np.empty((n, 3), order='F')
    check(LIB.coxdev_score_residuals(h.h, _d(eta), _d(w), X.ctypes.data_as(dp), 3, U.ctypes.data_as(dp)))
    np.testing.assert_allclose(U, cd.score_residuals(eta, X, w), **tol)

    # wild score
    G = rng.normal(size=n)
    out = np.empty(n)
    check(LIB.coxdev_wild_score(h.h, _d(eta), _d(w), _d(G), out.ctypes.data_as(dp)))
    np.testing.assert_allclose(out, cd.wild_score(eta, G, w), **tol)

    # draws: identical for the same seed (the model-based methods need equal weights)
    methods = ['wild'] + (['multinomial', 'bernoulli', 'fisher'] if weights == 'unit' else [])
    for method in methods:
        draws = np.empty((n, 4), order='F')
        check(LIB.coxdev_sample_score(h.h, _d(eta), _d(w), method.encode(), 4, 2024, draws.ctypes.data_as(dp)))
        np.testing.assert_array_equal(draws, cd.sample_score(eta, w, method=method, n_draws=4, seed=2024))


def test_new_weights_rebuild():
    # a sample_weight different from the handle's rebuilds it, as in Python
    rng = np.random.default_rng(1)
    event, status, start, strata_, w, eta = make_data(40, True, False, 'random', rng)
    h = Handle(event, status, start=start)            # weights one
    cd = CoxDeviance(event=event, status=status, start=start)
    dev = C.c_double()
    check(LIB.coxdev_deviance(h.h, _d(eta), _d(w), C.byref(dev), None, None, None))
    np.testing.assert_allclose(dev.value, cd(eta, w).deviance, rtol=1e-13)


def test_errors():
    event = np.array([1., 2., 3.])
    bad = np.array([1, 2, 0], np.int32)
    h = C.c_void_p()
    rc = LIB.coxdev_new(3, None, _d(event), bad.ctypes.data_as(ip), None, None, 1, C.byref(h))
    assert rc == -1 and not h.value
    assert b'status' in LIB.coxdev_last_error()
    assert LIB.coxdev_deviance(None, _d(event), None, None, None, None, None) == -1
