"""
coxdev.jax against the C++ code: the preprocessing, the deviance, its
gradient and Hessian, the information and its derivative in a direction of
the linear predictor, with Efron's and Breslow's ties, (start, stop] data,
unequal weights, weights 0 and strata. (The C++ derivative of the
information alone is tested in test_information_operator.py.)
"""

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from coxdev import CoxDeviance
from coxdev.jax import CoxDevianceJAX, preprocess

from .simulate import (simulate_df,
                       all_combos,
                       sample_weights,
                       sample_weights_zeros)


def _weights(kind, n, rng):
    if kind == 'ones':
        return np.ones(n)
    if kind == 'unequal':
        return sample_weights(n)
    if kind == 'zeros':
        return sample_weights_zeros(n)
    if kind == 'zeros_unequal':
        w = rng.uniform(0.5, 2, n)
        w[rng.choice(n, n // 4, replace=False)] = 0
        return w


def _grid_data(rng, n=60, start=True):
    # a coarse grid: many ties among the events, the starts, and between them
    event = np.ceil(rng.exponential(1, n) * 3) / 3
    status = rng.binomial(1, 0.7, n)
    entry = np.floor(rng.uniform(0, 0.9, n) * event * 3) / 3 if start else None
    return event, status, entry


def _both(event, status, entry, w, tie_breaking, strata=None):
    cox = CoxDeviance(event=event, status=status, start=entry, strata=strata,
                      sample_weight=w, tie_breaking=tie_breaking)
    cox_jax = CoxDevianceJAX(event, status, start=entry, strata=strata,
                             sample_weight=w, tie_breaking=tie_breaking)
    return cox, cox_jax


def _check_preprocess(event, status, entry, w, tie_breaking):
    cox = CoxDeviance(event=event, status=status, start=entry,
                      sample_weight=w, tie_breaking=tie_breaking)
    prep = preprocess(event, status, entry, w)
    for key, cpp in [('event_order', cox.event_order),
                     ('start_order', cox.start_order),
                     ('status', cox._status),
                     ('first', cox._first),
                     ('last', cox._last),
                     ('event_map', cox._event_map),
                     ('start_map', cox._start_map)]:
        np.testing.assert_array_equal(prep[key], cpp, err_msg=key)
    np.testing.assert_allclose(prep['scaling'], cox._scaling, rtol=0, atol=1e-15)


def _check_against_cpp(cox, cox_jax, w, rng, hessian=True, tol=1e-10):
    n = w.shape[0]
    eta = rng.standard_normal(n) * 0.5
    u, v = rng.standard_normal((2, n))
    eta_j = jnp.asarray(eta)

    result = cox(eta, w)
    deviance, gradient = result.deviance, result.gradient.copy()
    diag_hessian = result.diag_hessian.copy()

    np.testing.assert_allclose(cox_jax.loglik_sat(), result.loglik_sat, rtol=tol, atol=tol)
    np.testing.assert_allclose(cox_jax.deviance(eta_j), deviance, rtol=tol, atol=tol)
    np.testing.assert_allclose(cox_jax.gradient(eta_j), gradient, rtol=tol, atol=tol)

    information = cox.information(eta, w)
    np.testing.assert_allclose(cox_jax.information_matvec(eta_j, jnp.asarray(u)),
                               information @ u, rtol=tol, atol=tol)

    dI = cox.information_derivative(eta, v, w)
    dI_jax = cox_jax.information_derivative_matvec(eta_j, jnp.asarray(v), jnp.asarray(u))
    np.testing.assert_allclose(dI_jax, dI @ u, rtol=tol, atol=tol)

    if hessian:
        # the full Hessian of the deviance: twice the information, with
        # diag_hessian its diagonal
        H = np.asarray(jax.hessian(cox_jax.deviance)(eta_j))
        np.testing.assert_allclose(H, 2 * (information @ np.eye(n)), rtol=tol, atol=tol)
        np.testing.assert_allclose(np.diag(H), diag_hessian, rtol=tol, atol=tol)

    return eta


# every combination of the types of ties of simulate.py

@pytest.mark.parametrize('tie_types', all_combos)
@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('weights', ['ones', 'unequal', 'zeros'])
@pytest.mark.parametrize('have_start_times', [True, False])
def test_jax_all_tie_types(tie_types, tie_breaking, weights, have_start_times):
    rng = np.random.default_rng(hash((tie_types, tie_breaking, weights, have_start_times)) % 2**32)
    data = simulate_df(tie_types, 2, 4, rng=rng, noinfo=True)
    n = data.shape[0]
    w = _weights(weights, n, rng)
    event, status = data['event'].values, data['status'].values.astype(int)
    entry = data['start'].values if have_start_times else None

    _check_preprocess(event, status, entry, w, tie_breaking)
    cox, cox_jax = _both(event, status, entry, w, tie_breaking)
    _check_against_cpp(cox, cox_jax, w, rng, hessian=n <= 80)


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('weights', ['ones', 'unequal', 'zeros_unequal'])
@pytest.mark.parametrize('have_start_times', [True, False])
@pytest.mark.parametrize('stratified', [False, True])
@pytest.mark.parametrize('seed', range(4))
def test_jax_grid(tie_breaking, weights, have_start_times, stratified, seed):
    rng = np.random.default_rng(seed)
    event, status, entry = _grid_data(rng, start=have_start_times)
    n = event.shape[0]
    w = _weights(weights, n, rng)
    strata = rng.choice(3, n) if stratified else None
    if not stratified:
        _check_preprocess(event, status, entry, w, tie_breaking)
    cox, cox_jax = _both(event, status, entry, w, tie_breaking, strata=strata)
    _check_against_cpp(cox, cox_jax, w, rng)


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('have_start_times', [True, False])
@pytest.mark.parametrize('seed', range(3))
def test_jax_zero_weights_drop(tie_breaking, have_start_times, seed):
    # weight 0 is the same as dropping the observation: the deviance, and the
    # derivatives on the others; the derivatives on those dropped are 0
    rng = np.random.default_rng(seed)
    event, status, entry = _grid_data(rng, start=have_start_times)
    n = event.shape[0]
    w = _weights('zeros_unequal', n, rng)
    keep = w > 0

    cox_jax = CoxDevianceJAX(event, status, start=entry, sample_weight=w,
                             tie_breaking=tie_breaking)
    kept = CoxDevianceJAX(event[keep], status[keep],
                          start=None if entry is None else entry[keep],
                          sample_weight=w[keep], tie_breaking=tie_breaking)

    eta = jnp.asarray(rng.standard_normal(n) * 0.5)
    u, v = (jnp.asarray(x) for x in rng.standard_normal((2, n)))

    np.testing.assert_allclose(cox_jax.deviance(eta), kept.deviance(eta[keep]), rtol=1e-10)
    for full, dropped in [(cox_jax.gradient(eta), kept.gradient(eta[keep])),
                          (cox_jax.information_matvec(eta, u),
                           kept.information_matvec(eta[keep], u[keep])),
                          (cox_jax.information_derivative_matvec(eta, v, u),
                           kept.information_derivative_matvec(eta[keep], v[keep], u[keep]))]:
        full = np.asarray(full)
        np.testing.assert_allclose(full[keep], dropped, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(full[~keep], 0, atol=1e-12)


def test_jax_later_weights():
    # weights other than those at construction, with the same zeros
    rng = np.random.default_rng(3)
    event, status, entry = _grid_data(rng)
    n = event.shape[0]
    w0 = _weights('zeros_unequal', n, rng)
    w = np.where(w0 > 0, rng.uniform(0.2, 3, n), 0)
    cox = CoxDeviance(event=event, status=status, start=entry, sample_weight=w)
    cox_jax = CoxDevianceJAX(event, status, start=entry, sample_weight=w0)
    eta = rng.standard_normal(n) * 0.5
    result = cox(eta, w)
    np.testing.assert_allclose(cox_jax.deviance(jnp.asarray(eta), jnp.asarray(w)),
                               result.deviance, rtol=1e-10)
    np.testing.assert_allclose(cox_jax.gradient(jnp.asarray(eta), jnp.asarray(w)),
                               result.gradient, rtol=1e-10, atol=1e-12)


def test_jax_jit():
    rng = np.random.default_rng(4)
    event, status, entry = _grid_data(rng)
    n = event.shape[0]
    w = _weights('zeros_unequal', n, rng)
    cox, cox_jax = _both(event, status, entry, w, 'efron')
    eta = rng.standard_normal(n) * 0.5
    u, v = rng.standard_normal((2, n))
    dI = jax.jit(cox_jax.information_derivative_matvec)
    np.testing.assert_allclose(dI(jnp.asarray(eta), jnp.asarray(v), jnp.asarray(u)),
                               cox.information_derivative(eta, v, w) @ u, rtol=1e-10, atol=1e-12)
