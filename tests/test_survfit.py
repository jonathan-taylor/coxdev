"""
Tests of CoxDeviance.survfit (the survival table and curves), the martingale
and deviance residuals and the log partial likelihood: against direct
computations, the single strata, and R's survival package (survfit.coxph,
residuals.coxph and coxph's loglik).
"""

import numpy as np
import pytest

from coxdev import CoxDeviance, SurvivalCurves

from .test_residuals_hazard import _setting, _coxph_R, reference_hazard


def reference_table(event, status, entry, weight):
    """Direct computation: at each distinct event time t, the weight at risk
    (start < t <= event) and the weights of the failures and censorings."""
    start = -np.inf * np.ones(len(event)) if entry is None else entry
    times = np.unique(event)
    n_risk = np.array([weight[(start < t) & (event >= t)].sum() for t in times])
    n_event = np.array([weight[(event == t) & (status == 1)].sum() for t in times])
    n_censor = np.array([weight[(event == t) & (status == 0)].sum() for t in times])
    return times, n_risk, n_event, n_censor


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('start', [False, True])
@pytest.mark.parametrize('weights', ['ones', 'unequal', 'zeros'])
def test_survfit(tie_breaking, start, weights):
    X, event, status, entry, w, eta = _setting(start, weights)
    cox = CoxDeviance(event=event, status=status, start=entry, sample_weight=w,
                      tie_breaking=tie_breaking)
    new_eta = np.array([-1., 0., 2.5])
    curves = cox.survfit(eta, w, new_linear_predictor=new_eta)
    assert isinstance(curves, SurvivalCurves)

    times, n_risk, n_event, n_censor = reference_table(event, status, entry, w)
    np.testing.assert_allclose(curves.time, times)
    np.testing.assert_allclose(curves.n_risk, n_risk, atol=1e-10)
    np.testing.assert_allclose(curves.n_event, n_event, atol=1e-10)
    np.testing.assert_allclose(curves.n_censor, n_censor, atol=1e-10)
    assert np.all(curves.stratum == 0)

    # the baseline cumulative hazard, constant between failure times, times
    # exp(new_eta)
    t_ref, h_ref = reference_hazard(eta, event, status, entry, w, tie_breaking == 'efron')
    pos = np.searchsorted(t_ref, times, side='right') - 1
    H = np.where(pos >= 0, np.cumsum(h_ref)[np.maximum(pos, 0)], 0)
    np.testing.assert_allclose(curves.cumulative_hazard, H[:, None] * np.exp(new_eta)[None, :],
                               rtol=1e-10)
    np.testing.assert_allclose(curves.survival, np.exp(-curves.cumulative_hazard))

    # the weights default to the object's (it was built with w)
    np.testing.assert_allclose(cox.survfit(eta, new_linear_predictor=new_eta).cumulative_hazard,
                               curves.cumulative_hazard)
    if weights != 'ones':
        ones = CoxDeviance(event=event, status=status, start=entry, tie_breaking=tie_breaking)
        assert not np.allclose(ones.survfit(eta, new_linear_predictor=new_eta).cumulative_hazard,
                               curves.cumulative_hazard)

    # without new subjects, one curve at the weighted mean
    default = cox.survfit(eta, w)
    center = np.sum(w * eta) / np.sum(w)
    np.testing.assert_allclose(default.cumulative_hazard[:, 0], H * np.exp(center), rtol=1e-10)
    other = cox.survfit(eta, w, center=0.)
    np.testing.assert_allclose(other.cumulative_hazard[:, 0], H, rtol=1e-10)


def test_survfit_predict():
    X, event, status, entry, w, eta = _setting(False, 'unequal')
    curves = CoxDeviance(event=event, status=status, sample_weight=w).survfit(eta, w)
    t = curves.time
    P = curves.predict([t[0] - 1, t[2], (t[2] + t[3]) / 2, t[-1] + 1])
    np.testing.assert_allclose(P[0], 1)
    np.testing.assert_allclose(P[1], curves.survival[2])
    np.testing.assert_allclose(P[2], curves.survival[2])
    np.testing.assert_allclose(P[3], curves.survival[-1])


def test_survfit_stratified():
    X, event, status, entry, w, eta = _setting(True, 'unequal')
    n = len(event)
    strata = np.array(['b', 'a'])[np.random.default_rng(4).integers(0, 2, n)]
    cox = CoxDeviance(event=event, status=status, start=entry, strata=strata, sample_weight=w)
    new_eta, new_strata = np.array([0.5, -0.5, 1.]), np.array(['a', 'b', 'a'])
    curves = cox.survfit(eta, w, new_linear_predictor=new_eta, new_strata=new_strata)
    assert list(np.unique(curves.stratum)) == ['a', 'b']
    center = np.sum(w * eta) / np.sum(w)
    for s in ['a', 'b']:
        k, rows = strata == s, curves.stratum == s
        single = CoxDeviance(event=event[k], status=status[k], start=entry[k],
                             sample_weight=w[k]).survfit(eta[k], w[k], center=center)
        np.testing.assert_allclose(curves.time[rows], single.time)
        np.testing.assert_allclose(curves.n_risk[rows], single.n_risk)
        for j in range(3):
            if new_strata[j] == s:
                np.testing.assert_allclose(curves.cumulative_hazard[rows, j],
                                           single.cumulative_hazard[:, 0] * np.exp(new_eta[j] - center),
                                           rtol=1e-10)
            else:
                assert np.all(np.isnan(curves.cumulative_hazard[rows, j]))
    # each curve is a step function of its own stratum's rows
    P = curves.predict([curves.time.max() + 1])
    for j in range(3):
        rows = curves.stratum == new_strata[j]
        np.testing.assert_allclose(P[0, j], curves.survival[rows, j][-1])

    with pytest.raises(ValueError, match='new_strata is needed'):
        cox.survfit(eta, w, new_linear_predictor=new_eta)
    with pytest.raises(ValueError, match='not seen'):
        cox.survfit(eta, w, new_linear_predictor=new_eta, new_strata=['a', 'z', 'a'])
    with pytest.raises(ValueError, match='different lengths'):
        cox.survfit(eta, w, new_linear_predictor=new_eta, new_strata=['a', 'b'])


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('weights', ['ones', 'unequal', 'zeros'])
def test_residuals_loglik(tie_breaking, weights):
    X, event, status, entry, w, eta = _setting(True, weights)
    cox = CoxDeviance(event=event, status=status, start=entry, sample_weight=w,
                      tie_breaking=tie_breaking)
    M = cox.martingale_residuals(eta, w)
    np.testing.assert_allclose(M, status - cox.expected_events(eta, w))
    # the gradient of the deviance is -2 w M
    np.testing.assert_allclose(cox(eta, w).gradient, -2 * w * M, atol=1e-10)
    D = cox.deviance_residuals(eta, w)
    np.testing.assert_allclose(np.sign(D), np.sign(M))
    with np.errstate(divide='ignore'):
        d = -2 * (M + np.where(status > 0, np.log(np.where(status > 0, 1 - M, 1)), 0))
    np.testing.assert_allclose(D**2, d, atol=1e-12)
    result = cox(eta, w)
    np.testing.assert_allclose(result.loglik, result.loglik_sat - result.deviance / 2)


# R's survival package

def _survfit_R(Rinfo, eta, new_eta, event, status, entry, weight, ties, strata=None,
               new_strata=None):
    ro = Rinfo['rpy']
    X = eta[:, None]
    _coxph_R(Rinfo, X, np.array([1.]), event, status, entry, weight, ties, strata=strata)
    ro.globalenv['new_eta'] = ro.FloatVector(new_eta)
    if strata is None:
        ro.r('S <- survival::survfit(F, newdata = list(X = matrix(new_eta)), se.fit = FALSE)')
        ro.r('S0 <- survival::survfit(F, se.fit = FALSE)')
    else:
        ro.globalenv['new_stratum'] = ro.IntVector(new_strata)
        ro.r('S <- survival::survfit(F, newdata = list(X = matrix(new_eta), stratum = new_stratum), '
             'se.fit = FALSE)')
        ro.r('S0 <- NULL')
    out = {k: np.array(ro.r(f'as.numeric(S${k})'))
           for k in ['time', 'n.risk', 'n.event', 'n.censor', 'cumhaz']}
    out['cumhaz0'] = (np.array(ro.r('as.numeric(S0$cumhaz)')) if strata is None else None)
    out['martingale'] = np.array(ro.r('as.vector(residuals(F, type = "martingale"))'))
    out['deviance'] = np.array(ro.r('as.vector(residuals(F, type = "deviance"))'))
    out['loglik'] = float(np.array(ro.r('F$loglik[2]'))[0])
    return out


@pytest.mark.parametrize('tie_breaking', ['efron', 'breslow'])
@pytest.mark.parametrize('start', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_against_survival(Rinfo, tie_breaking, start, weighted):
    X, event, status, entry, w, eta = _setting(start, 'unequal' if weighted else 'ones')
    eta = X @ np.array([0.3, -0.2, 0.1])
    new_eta = np.array([-1., 0., 0.7])
    cox = CoxDeviance(event=event, status=status, start=entry, sample_weight=w,
                      tie_breaking=tie_breaking)
    R = _survfit_R(Rinfo, eta, new_eta, event, status, entry, w, tie_breaking)

    # R's newdata curves are at X = new_eta less nothing: coxph's linear
    # predictor of new data is centered at its means, but the curves are not
    curves = cox.survfit(eta, w, new_linear_predictor=new_eta)
    np.testing.assert_allclose(curves.time, R['time'])
    np.testing.assert_allclose(curves.n_risk, R['n.risk'], rtol=1e-10)
    np.testing.assert_allclose(curves.n_event, R['n.event'], rtol=1e-10)
    np.testing.assert_allclose(curves.n_censor, R['n.censor'], rtol=1e-10)
    # R stores the cumhaz matrix column-major
    np.testing.assert_allclose(curves.cumulative_hazard.T.reshape(-1), R['cumhaz'],
                               rtol=1e-8, atol=1e-12)
    # R's default curve is at coxph's means: weighted for right-censored
    # data, unweighted for (start, stop] data
    center = eta.mean() if start else np.sum(w * eta) / np.sum(w)
    default = cox.survfit(eta, w, center=center)
    np.testing.assert_allclose(default.cumulative_hazard[:, 0], R['cumhaz0'], rtol=1e-8, atol=1e-12)

    # survival's residuals and log partial likelihood (its residuals are per
    # observation, unweighted)
    np.testing.assert_allclose(cox.martingale_residuals(eta, w), R['martingale'], atol=1e-10)
    np.testing.assert_allclose(cox.deviance_residuals(eta, w), R['deviance'], atol=1e-10)
    np.testing.assert_allclose(cox(eta, w).loglik, R['loglik'], rtol=1e-10)


def test_against_survival_stratified(Rinfo):
    X, event, status, entry, w, eta = _setting(True, 'unequal')
    eta = X @ np.array([0.3, -0.2, 0.1])
    strata = np.random.default_rng(4).integers(1, 3, len(event))
    new_eta, new_strata = np.array([-1., 0.7]), np.array([2, 1])
    cox = CoxDeviance(event=event, status=status, start=entry, strata=strata, sample_weight=w)
    R = _survfit_R(Rinfo, eta, new_eta, event, status, entry, w, 'efron', strata=strata,
                   new_strata=new_strata)
    curves = cox.survfit(eta, w, new_linear_predictor=new_eta, new_strata=new_strata)
    # R gives each new subject's curve on the times of its stratum only
    flat = np.concatenate([curves.cumulative_hazard[curves.stratum == s, j]
                           for j, s in enumerate(new_strata)])
    np.testing.assert_allclose(flat, R['cumhaz'], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(cox.martingale_residuals(eta, w), R['martingale'], atol=1e-10)
    np.testing.assert_allclose(cox(eta, w).loglik, R['loglik'], rtol=1e-10)
