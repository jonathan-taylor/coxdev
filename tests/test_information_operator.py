"""
CoxDevianceResult and CoxInformation: results are not changed by later
calls, an information operator stays at its own linear predictor, products
with a matrix agree with column-by-column products, and the operator is
symmetric (adjoint, transpose, rmatvec). The same for the derivative of the
information in a direction, also against finite differences of the
information.
"""

import numpy as np
import pytest

from coxdev import CoxDeviance

from .test_sample_score import _data


@pytest.mark.parametrize('strata', [False, True])
@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
def test_information_operator(strata, tie_breaking):
    X, event, status, entry = _data(n=120, start=True)
    n = len(event)
    rng = np.random.default_rng(4)
    w = rng.uniform(0.5, 2, n)
    cox = CoxDeviance(event=event, status=status, start=entry, sample_weight=w,
                      strata=(np.arange(n) % 3) if strata else None,
                      tie_breaking=tie_breaking)
    eta1, eta2 = X @ [0.3, -0.2, 0.1], X @ [-0.5, 0.4, 0.2]
    V = rng.standard_normal((n, 4))

    r1 = cox(eta1, w)
    I1 = cox.information(eta1, w)
    P1 = np.column_stack([I1 @ V[:, j] for j in range(4)])

    # another linear predictor in between changes neither the result nor I1
    r2 = cox(eta2, w)
    np.testing.assert_array_equal(r1.linear_predictor, eta1)
    np.testing.assert_array_equal(r1.sample_weight, w)
    assert r1.deviance != r2.deviance
    np.testing.assert_allclose(I1 @ V[:, 0], P1[:, 0], rtol=1e-12, atol=1e-12)

    # a matrix in one call, as a matrix and through X.T @ (I @ X)
    np.testing.assert_allclose(I1 @ V, P1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(I1.matmat(V), P1, rtol=1e-12, atol=1e-12)
    I2 = cox.information(eta2, w)
    np.testing.assert_allclose(I2 @ V, np.column_stack([I2 @ V[:, j] for j in range(4)]),
                               rtol=1e-12, atol=1e-12)

    # symmetric
    np.testing.assert_allclose(I1.T @ V[:, 1], P1[:, 1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(I1.H @ V, P1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(I1.rmatvec(V[:, 2]), P1[:, 2], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(V[:, 3] @ (I1 @ V[:, 2]), V[:, 2] @ (I1 @ V[:, 3]), rtol=1e-10)

    # the diagonal of the Hessian of the deviance is twice that of I
    E = np.eye(n)[:, :10]
    np.testing.assert_allclose(np.diag((I1 @ E)[:10]), r1.diag_hessian[:10] / 2, rtol=1e-10)


# CoxInformationDerivative: the derivative of the information in a direction

def _tied_data(rng, n=60, start=True):
    # a coarse grid: many ties among the events, the starts, and between them
    event = np.ceil(rng.exponential(1, n) * 3) / 3
    status = rng.binomial(1, 0.7, n)
    entry = np.floor(rng.uniform(0, 0.9, n) * event * 3) / 3 if start else None
    return event, status, entry


def _zero_weights(n, rng):
    w = rng.uniform(0.5, 2, n)
    w[rng.choice(n, n // 4, replace=False)] = 0
    return w


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('have_start_times', [True, False])
@pytest.mark.parametrize('stratified', [False, True])
def test_information_derivative_finite_differences(tie_breaking, have_start_times, stratified):
    # the C++ derivative of the information, without JAX: against central
    # differences of the information, symmetric in the direction and the
    # argument, and as a matrix (matmat) symmetric
    rng = np.random.default_rng(5)
    event, status, entry = _tied_data(rng, start=have_start_times)
    n = event.shape[0]
    w = _zero_weights(n, rng)
    strata = rng.choice(3, n) if stratified else None
    cox = CoxDeviance(event=event, status=status, start=entry, strata=strata,
                      sample_weight=w, tie_breaking=tie_breaking)
    eta = rng.standard_normal(n) * 0.5
    u, v = rng.standard_normal((2, n))

    h = 1e-5
    fd = (cox.information(eta + h * v, w) @ u - cox.information(eta - h * v, w) @ u) / (2 * h)
    dI_v = cox.information_derivative(eta, v, w)
    np.testing.assert_allclose(dI_v @ u, fd, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(dI_v @ u, cox.information_derivative(eta, u, w) @ v,
                               rtol=1e-10, atol=1e-12)

    D = dI_v @ np.eye(n)
    np.testing.assert_allclose(D, D.T, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(D[:, 3], dI_v @ np.eye(n)[:, 3], rtol=1e-12, atol=1e-14)
    # the information derivative is linear in the direction
    dI_2v = cox.information_derivative(eta, 2 * v + u, w) @ u
    np.testing.assert_allclose(dI_2v, 2 * (dI_v @ u) + cox.information_derivative(eta, u, w) @ u,
                               rtol=1e-10, atol=1e-12)


def test_information_derivative_after_other_calls():
    # as CoxInformation, the operator brings the C++ object back to its
    # linear predictor
    rng = np.random.default_rng(6)
    event, status, entry = _tied_data(rng)
    n = event.shape[0]
    cox = CoxDeviance(event=event, status=status, start=entry)
    eta, other, u, v = rng.standard_normal((4, n))
    dI = cox.information_derivative(eta, v)
    before = dI @ u
    cox(other)
    np.testing.assert_allclose(dI @ u, before, rtol=1e-12)
