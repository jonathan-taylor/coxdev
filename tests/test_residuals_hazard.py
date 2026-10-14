"""
Tests of CoxDeviance.score_residuals and CoxDeviance.baseline_hazard: against
direct computations, the identities with the score and wild_score, and R's
survival package (residuals(type='score') and basehaz).
"""

import numpy as np
import pytest

from coxdev import CoxDeviance, BaselineHazard

from .test_sample_score import _data, reference_residuals


def _setting(start, weights, seed=0):
    X, event, status, entry = _data(n=120, start=start, seed=seed)
    n = len(event)
    rng = np.random.default_rng(seed + 1)
    w = np.ones(n) if weights == 'ones' else rng.uniform(0.5, 2, n)
    if weights == 'zeros':
        w[rng.choice(n, 15, replace=False)] = 0
    eta = X @ np.array([0.3, -0.2, 0.1]) + 2       # far from mean zero
    return X, event, status, entry, w, eta


def reference_hazard(eta, event, status, entry, weight, efron):
    """Direct computation: at each failure time, the sum over the Efron terms
    of wbar / (sum of w exp(eta) over the risk set, less s of the failures)."""
    start = -np.inf * np.ones(len(event)) if entry is None else entry
    e = weight * np.exp(eta)
    times, hazard = [], []
    for t in np.unique(event[(status == 1) & (weight > 0)]):
        D = (event == t) & (status == 1) & (weight > 0)
        R = (event >= t) & (start < t) & (weight > 0)
        d = D.sum()
        wbar = weight[D].mean()
        h = 0.
        for r in range(d):
            s = r / d if efron else 0.
            h += wbar / (e * (R - s * D)).sum()
        times.append(t)
        hazard.append(h)
    return np.array(times), np.array(hazard)


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('start', [False, True])
@pytest.mark.parametrize('weights', ['ones', 'unequal', 'zeros'])
def test_score_residuals(tie_breaking, start, weights):
    X, event, status, entry, w, eta = _setting(start, weights)
    cox = CoxDeviance(event=event, status=status, start=entry, sample_weight=w,
                      tie_breaking=tie_breaking)
    U = cox.score_residuals(eta, X, sample_weight=w)
    assert U.shape == X.shape
    keep = w > 0
    U_ref = np.zeros_like(X)
    U_ref[keep] = reference_residuals(X[keep], eta[keep], event[keep], status[keep],
                                      None if entry is None else entry[keep], w[keep],
                                      tie_breaking == 'efron')
    np.testing.assert_allclose(U, U_ref, atol=1e-10)
    # the rows sum to the score; G @ U is X^T wild_score(G)
    score = -cox(eta, w).gradient / 2
    np.testing.assert_allclose(U.sum(0), X.T @ score, atol=1e-10)
    G = np.random.default_rng(3).standard_normal(len(eta))
    np.testing.assert_allclose(G @ U, X.T @ cox.wild_score(eta, G, sample_weight=w), atol=1e-10)
    # a vector design gives a vector
    np.testing.assert_allclose(cox.score_residuals(eta, X[:, 0], sample_weight=w), U[:, 0])


def test_score_residuals_stratified():
    X, event, status, entry, w, eta = _setting(True, 'unequal')
    n = len(event)
    strata = np.arange(n) % 3
    cox = CoxDeviance(event=event, status=status, start=entry, strata=strata, sample_weight=w)
    U = cox.score_residuals(eta, X, sample_weight=w)
    for s in range(3):
        k = strata == s
        single = CoxDeviance(event=event[k], status=status[k], start=entry[k], sample_weight=w[k])
        np.testing.assert_allclose(U[k], single.score_residuals(eta[k], X[k], sample_weight=w[k]),
                                   atol=1e-12)
    with pytest.raises(ValueError, match='one row per observation'):
        cox.score_residuals(eta, X[:-1])


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('start', [False, True])
@pytest.mark.parametrize('weights', ['ones', 'unequal', 'zeros'])
def test_baseline_hazard(tie_breaking, start, weights):
    X, event, status, entry, w, eta = _setting(start, weights)
    cox = CoxDeviance(event=event, status=status, start=entry, sample_weight=w,
                      tie_breaking=tie_breaking)
    H = cox.baseline_hazard(eta, sample_weight=w)
    assert isinstance(H, BaselineHazard)
    times, hazard = reference_hazard(eta, event, status, entry, w, tie_breaking == 'efron')
    np.testing.assert_array_equal(H.time, times)
    np.testing.assert_allclose(H.hazard, hazard, rtol=1e-10)
    np.testing.assert_allclose(H.cumulative_hazard, np.cumsum(hazard), rtol=1e-10)
    np.testing.assert_array_equal(H.stratum, 0)
    # for the linear predictor as given: a shift by c scales it by exp(-c)
    H2 = cox.baseline_hazard(eta + 1.5, sample_weight=w)
    np.testing.assert_allclose(H2.cumulative_hazard, H.cumulative_hazard * np.exp(-1.5), rtol=1e-10)


def test_baseline_hazard_stratified():
    X, event, status, entry, w, eta = _setting(True, 'ones')
    n = len(event)
    strata = np.where(np.arange(n) % 2 == 0, 7, 3)
    cox = CoxDeviance(event=event, status=status, start=entry, strata=strata)
    H = cox.baseline_hazard(eta)
    assert list(np.unique(H.stratum)) == [3, 7]
    for s in [3, 7]:
        k = strata == s
        single = CoxDeviance(event=event[k], status=status[k], start=entry[k]).baseline_hazard(eta[k])
        np.testing.assert_allclose(H.time[H.stratum == s], single.time)
        np.testing.assert_allclose(H.cumulative_hazard[H.stratum == s], single.cumulative_hazard,
                                   rtol=1e-12)


# R's survival package

def _coxph_R(Rinfo, X, beta, event, status, entry, weight, ties, strata=None):
    rpy = Rinfo['rpy']
    ro = rpy
    ro.globalenv['X'] = ro.r.matrix(ro.FloatVector(X.ravel(order='F')), nrow=X.shape[0])
    ro.globalenv['beta'] = ro.FloatVector(beta)
    ro.globalenv['event'] = ro.FloatVector(event)
    ro.globalenv['status'] = ro.FloatVector(status)
    ro.globalenv['weight'] = ro.FloatVector(weight)
    ro.globalenv['ties'] = ties
    if entry is None:
        ro.r('y <- survival::Surv(event, status)')
    else:
        ro.globalenv['entry'] = ro.FloatVector(entry)
        ro.r('y <- survival::Surv(entry, event, status)')
    if strata is None:
        ro.r('F <- survival::coxph(y ~ X, init = beta, weights = weight, ties = ties, '
             'control = survival::coxph.control(iter.max = 0, timefix = FALSE))')
    else:
        ro.globalenv['stratum'] = ro.IntVector(strata)
        ro.r('F <- survival::coxph(y ~ X + survival::strata(stratum), init = beta, '
             'weights = weight, ties = ties, '
             'control = survival::coxph.control(iter.max = 0, timefix = FALSE))')
    U = np.array(ro.r('as.vector(residuals(F, type = "score"))')).reshape(X.shape, order='F')
    bh = ro.r('survival::basehaz(F, centered = FALSE)')
    return U, np.array(bh.rx2('time')), np.array(bh.rx2('hazard'))


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('start', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_against_survival(Rinfo, tie_breaking, start, weighted):
    X, event, status, entry, w, eta = _setting(start, 'unequal' if weighted else 'ones')
    beta = np.array([0.3, -0.2, 0.1])
    eta = X @ beta
    cox = CoxDeviance(event=event, status=status, start=entry, sample_weight=w,
                      tie_breaking=tie_breaking)
    U_R, time_R, cumhaz_R = _coxph_R(Rinfo, X, beta, event, status, entry, w, tie_breaking)

    # survival's score residuals are per unit weight
    U = cox.score_residuals(eta, X, sample_weight=w)
    np.testing.assert_allclose(U, w[:, None] * U_R, atol=1e-10)

    # basehaz reports every event time; compare at the failure times
    H = cox.baseline_hazard(eta, sample_weight=w)
    at = np.searchsorted(time_R, H.time)
    np.testing.assert_allclose(time_R[at], H.time)
    np.testing.assert_allclose(H.cumulative_hazard, cumhaz_R[at], rtol=1e-10)
