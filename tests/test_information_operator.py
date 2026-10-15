"""
CoxDevianceResult and CoxInformation: results are not changed by later
calls, an information operator stays at its own linear predictor, products
with a matrix agree with column-by-column products, and the operator is
symmetric (adjoint, transpose, rmatvec).
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
