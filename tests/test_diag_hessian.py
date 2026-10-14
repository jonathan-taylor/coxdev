"""
diag_hessian is the diagonal of the Hessian of the deviance with respect to
the linear predictor: twice the diagonal of the information operator.
"""

import numpy as np
import pytest

from coxdev import CoxDeviance


def _diag(operator, n):
    return np.array([(operator @ np.eye(n)[:, i])[i] for i in range(n)])


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('start', [False, True])
@pytest.mark.parametrize('weights', ['ones', 'unequal', 'zeros'])
@pytest.mark.parametrize('stratified', [False, True])
@pytest.mark.parametrize('seed', range(5))
def test_diag_hessian_is_diagonal_of_information(tie_breaking, start, weights, stratified, seed):
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

    cox = CoxDeviance(event=event, status=status, start=entry, strata=strata,
                      tie_breaking=tie_breaking, sample_weight=w)
    diag_hessian = cox(eta, w).diag_hessian.copy()
    information = cox.information(eta, w)
    np.testing.assert_allclose(diag_hessian, 2 * _diag(information, n), rtol=1e-10, atol=1e-12)


def test_diag_hessian_is_derivative_of_gradient():
    # the same, against finite differences of the gradient
    rng = np.random.default_rng(1)
    n = 40
    event = np.ceil(rng.exponential(1, n) * 3) / 3
    status = rng.binomial(1, 0.7, n)
    eta = rng.standard_normal(n) * 0.3
    entry = np.floor(rng.uniform(0, 0.5, n) * event * 6) / 6
    cox = CoxDeviance(event=event, status=status, start=entry, tie_breaking='efron')
    diag_hessian = cox(eta).diag_hessian.copy()
    h = 1e-6
    fd = np.array([(cox(eta + h * np.eye(n)[i]).gradient[i] - cox(eta - h * np.eye(n)[i]).gradient[i]) / (2 * h)
                   for i in range(n)])
    np.testing.assert_allclose(diag_hessian, fd, rtol=1e-6, atol=1e-8)
