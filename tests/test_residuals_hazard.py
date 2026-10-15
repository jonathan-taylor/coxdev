"""
Tests of CoxDeviance.score_residuals and CoxDeviance.baseline_hazard: against
direct computations, the identities with the score and wild_score, and R's
survival package (residuals(type='score') and basehaz).
"""

import numpy as np
import pytest
import scipy.sparse as sp

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


@pytest.mark.parametrize('fmt', ['csc', 'csr', 'coo'])
def test_score_residuals_sparse(fmt):
    """A sparse X gives the residuals of the dense X; a column with no
    nonzeros at all gives zeros."""
    X, event, status, entry, w, eta = _setting(True, 'unequal')
    X = np.where(np.abs(X) > 1, X, 0)
    X[:, 1] = 0
    cox = CoxDeviance(event=event, status=status, start=entry, sample_weight=w)
    U = cox.score_residuals(eta, X, sample_weight=w)
    U_sp = cox.score_residuals(eta, getattr(sp, f'{fmt}_matrix')(X), sample_weight=w)
    np.testing.assert_allclose(U_sp, U, atol=1e-14)
    np.testing.assert_allclose(U[:, 1], 0, atol=1e-14)


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


# Schoenfeld residuals and expected events

def reference_schoenfeld(X, eta, event, status, entry, weight, efron):
    """Direct computation: for each failure with positive weight, x_i less the
    mean of x over its risk set with weights w exp(eta); with Efron's ties,
    the average over r = 0, ..., d - 1 of the means with the tied failures
    downweighted by r / d. Zero for the other rows."""
    start = -np.inf * np.ones(len(event)) if entry is None else entry
    e = weight * np.exp(eta - eta.max())
    R_sch = np.zeros_like(X)
    for t in np.unique(event[(status == 1) & (weight > 0)]):
        D = (event == t) & (status == 1) & (weight > 0)
        R = (event >= t) & (start < t) & (weight > 0)
        d = D.sum()
        means = []
        for r in range(d):
            s = r / d if efron else 0.
            p = e * (R - s * D)
            means.append(p @ X / p.sum())
        R_sch[D] = X[D] - np.mean(means, 0)
    return R_sch


def reference_expected(eta, event, status, entry, weight, efron):
    """Direct computation: exp(eta_i) times the sum over the failure times in
    (start_i, event_i] and the Efron terms r of wbar / S_r, with i's own
    failure time downweighted by r / d if efron."""
    n = len(event)
    start = -np.inf * np.ones(n) if entry is None else entry
    e = weight * np.exp(eta)
    E = np.zeros(n)
    for t in np.unique(event[(status == 1) & (weight > 0)]):
        D = (event == t) & (status == 1) & (weight > 0)
        R = (event >= t) & (start < t)              # i at risk, whatever its weight
        d = D.sum()
        wbar = weight[D].mean()
        for r in range(d):
            s = r / d if efron else 0.
            S = (e * ((R & (weight > 0)) - s * D)).sum()
            E += np.exp(eta) * (R - s * D) * wbar / S
    return E


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('start', [False, True])
@pytest.mark.parametrize('weights', ['ones', 'unequal', 'zeros'])
def test_schoenfeld_residuals(tie_breaking, start, weights):
    X, event, status, entry, w, eta = _setting(start, weights)
    cox = CoxDeviance(event=event, status=status, start=entry, sample_weight=w,
                      tie_breaking=tie_breaking)
    R_sch = cox.schoenfeld_residuals(eta, X, sample_weight=w)
    assert R_sch.shape == X.shape
    R_ref = reference_schoenfeld(X, eta, event, status, entry, w, tie_breaking == 'efron')
    np.testing.assert_allclose(R_sch, R_ref, atol=1e-10)
    assert np.all(R_sch[(status == 0) | (w == 0)] == 0)
    # weighted, the rows sum to the score
    score = -cox(eta, w).gradient / 2
    np.testing.assert_allclose(w @ R_sch, X.T @ score, atol=1e-10)
    # a constant covariate has no residuals; a vector design gives a vector
    np.testing.assert_allclose(cox.schoenfeld_residuals(eta, np.ones(len(eta)), sample_weight=w),
                               0, atol=1e-12)
    np.testing.assert_allclose(cox.schoenfeld_residuals(eta, X[:, 0], sample_weight=w),
                               R_sch[:, 0])
    # the Hessian matvec in between leaves them unchanged
    cox.information(eta, w) @ X[:, 1]
    np.testing.assert_allclose(cox.schoenfeld_residuals(eta, X, sample_weight=w), R_sch,
                               atol=1e-14)


def test_schoenfeld_residuals_sparse_stratified():
    X, event, status, entry, w, eta = _setting(True, 'unequal')
    n = len(event)
    X = np.where(np.abs(X) > 1, X, 0)
    cox = CoxDeviance(event=event, status=status, start=entry, sample_weight=w)
    np.testing.assert_allclose(cox.schoenfeld_residuals(eta, sp.csr_matrix(X), sample_weight=w),
                               cox.schoenfeld_residuals(eta, X, sample_weight=w), atol=1e-14)
    strata = np.arange(n) % 3
    cox = CoxDeviance(event=event, status=status, start=entry, strata=strata, sample_weight=w)
    R_sch = cox.schoenfeld_residuals(eta, X, sample_weight=w)
    for s in range(3):
        k = strata == s
        single = CoxDeviance(event=event[k], status=status[k], start=entry[k], sample_weight=w[k])
        np.testing.assert_allclose(R_sch[k], single.schoenfeld_residuals(eta[k], X[k],
                                                                         sample_weight=w[k]),
                                   atol=1e-12)
    with pytest.raises(ValueError, match='one row per observation'):
        cox.schoenfeld_residuals(eta, X[:-1])


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('start', [False, True])
@pytest.mark.parametrize('weights', ['ones', 'unequal', 'zeros'])
def test_expected_events(tie_breaking, start, weights):
    X, event, status, entry, w, eta = _setting(start, weights)
    cox = CoxDeviance(event=event, status=status, start=entry, sample_weight=w,
                      tie_breaking=tie_breaking)
    E = cox.expected_events(eta, sample_weight=w)
    E_ref = reference_expected(eta, event, status, entry, w, tie_breaking == 'efron')
    np.testing.assert_allclose(E, E_ref, rtol=1e-10)
    np.testing.assert_allclose(cox.survival_at_exit(eta, sample_weight=w), np.exp(-E_ref),
                               rtol=1e-10)
    # for positive weights, w (status - E) is the score -gradient / 2
    keep = w > 0
    score = -cox(eta, w).gradient / 2
    np.testing.assert_allclose((w * (status - E))[keep], score[keep], atol=1e-10)
    # with the baseline hazard: without start times, the cumulative hazard
    # at the exit time, but for Efron's share at a tied failure time
    if not start and tie_breaking == 'breslow':
        H = cox.baseline_hazard(eta, sample_weight=w)
        at = np.searchsorted(H.time, event, side='right') - 1
        cumhaz = np.where(at >= 0, H.cumulative_hazard[np.maximum(at, 0)], 0)
        np.testing.assert_allclose(E, np.exp(eta) * cumhaz, rtol=1e-10)


def test_expected_events_stratified():
    X, event, status, entry, w, eta = _setting(True, 'unequal')
    n = len(event)
    strata = np.where(np.arange(n) % 2 == 0, 7, 3)
    cox = CoxDeviance(event=event, status=status, start=entry, strata=strata, sample_weight=w)
    E = cox.expected_events(eta, sample_weight=w)
    for s in [3, 7]:
        k = strata == s
        single = CoxDeviance(event=event[k], status=status[k], start=entry[k], sample_weight=w[k])
        np.testing.assert_allclose(E[k], single.expected_events(eta[k], sample_weight=w[k]),
                                   rtol=1e-12)


def _schoenfeld_expected_R(Rinfo, X, beta, event, status, entry, weight, ties):
    ro = Rinfo['rpy']
    _coxph_R(Rinfo, X, beta, event, status, entry, weight, ties)
    sch = np.array(ro.r('as.matrix(residuals(F, type = "schoenfeld"))'))
    expected = np.array(ro.r('as.vector(predict(F, type = "expected"))'))
    surv = np.array(ro.r('as.vector(predict(F, type = "survival"))'))
    return sch, expected, surv


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('start', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_schoenfeld_expected_against_survival(Rinfo, tie_breaking, start, weighted):
    X, event, status, entry, w, eta = _setting(start, 'unequal' if weighted else 'ones')
    beta = np.array([0.3, -0.2, 0.1])
    eta = X @ beta
    cox = CoxDeviance(event=event, status=status, start=entry, sample_weight=w,
                      tie_breaking=tie_breaking)
    sch_R, expected_R, surv_R = _schoenfeld_expected_R(Rinfo, X, beta, event, status, entry,
                                                      w, tie_breaking)
    # survival has the rows of the failures, in order of failure time; tied
    # failures have the same mean, so compare the sorted rows at each time
    R_sch = cox.schoenfeld_residuals(eta, X, sample_weight=w)
    fail = np.flatnonzero(status == 1)
    order = np.lexsort((R_sch[fail, 0], event[fail]))
    order_R = np.lexsort((sch_R[:, 0], np.sort(event[fail])))
    np.testing.assert_allclose(R_sch[fail][order], sch_R[order_R], atol=1e-10)
    np.testing.assert_allclose(cox.expected_events(eta, sample_weight=w), expected_R, rtol=1e-10)
    np.testing.assert_allclose(cox.survival_at_exit(eta, sample_weight=w), surv_R, rtol=1e-10)
