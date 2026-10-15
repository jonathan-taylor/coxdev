"""
CoxDeviance.c_index, Harrell's concordance index: against the O(n^2) count
from the definition, and R's survival::concordance and glmnet::Cindex.
"""

import numpy as np
import pytest

from coxdev import CoxDeviance


def _brute_force(pred, time, status, start, strata, w):
    """For each failure i at t, each j at risk at t that is not a failure at
    t, in the same stratum: weight w_i w_j, concordant if pred_i > pred_j,
    half if tied."""
    num = den = 0.
    for i in np.flatnonzero(status == 1):
        t = time[i]
        risk = (start < t) & (t <= time) & ~((status == 1) & (time == t)) & (strata == strata[i])
        ww = w[i] * w[risk]
        num += (ww * (1 + np.sign(pred[i] - pred[risk])) / 2).sum()
        den += ww.sum()
    return num / den if den > 0 else np.nan


def _data(rng, n):
    # few distinct times and rounded predictions, so that both have ties
    time = rng.integers(1, 10, n).astype(float)
    status = rng.binomial(1, 0.6, n)
    start = np.minimum(np.floor(time * rng.uniform(0, 1, n)), time - 1)
    strata = rng.integers(0, 3, n)
    w = rng.uniform(0, 2, n) * (rng.uniform(size=n) > 0.1)
    pred = np.round(rng.standard_normal(n), 1)
    return time, status, start, strata, w, pred


@pytest.mark.parametrize('seed', range(30))
def test_c_index_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 80))
    time, status, start, strata, w, pred = _data(rng, n)
    for S, G, W in [(None, None, None), (start, None, w), (None, strata, w), (start, strata, w)]:
        cox = CoxDeviance(event=time, status=status, start=S, strata=G)
        C = cox.c_index(pred, sample_weight=W)
        C_bf = _brute_force(pred, time, status,
                            np.full(n, -np.inf) if S is None else S,
                            np.zeros(n) if G is None else G,
                            np.ones(n) if W is None else W)
        np.testing.assert_allclose(C, C_bf, rtol=0, atol=1e-12, equal_nan=True)


def test_c_index_properties():
    rng = np.random.default_rng(1)
    time, status, start, strata, w, pred = _data(rng, 100)
    cox = CoxDeviance(event=time, status=status, start=start)
    # columns, a reversed predictor and a constant one
    P = np.column_stack([pred, -pred, np.zeros(100)])
    C = cox.c_index(P, sample_weight=w)
    assert C.shape == (3,)
    np.testing.assert_allclose(C, [cox.c_index(P[:, j], sample_weight=w) for j in range(3)])
    np.testing.assert_allclose(C[0] + C[1], 1, atol=1e-12)
    assert C[2] == 0.5
    # it needs no compute, and a compute does not change it
    cox(pred * 0.3)
    assert cox.c_index(pred, sample_weight=w) == C[0]
    # no comparable pair
    assert np.isnan(CoxDeviance(event=time, status=np.zeros(100, int)).c_index(pred))
    with pytest.raises(ValueError, match='one row per observation'):
        cox.c_index(pred[:-1])


@pytest.mark.parametrize('start', [False, True])
@pytest.mark.parametrize('stratified', [False, True])
def test_c_index_against_R(Rinfo, start, stratified):
    ro = Rinfo['rpy']
    rng = np.random.default_rng(2)
    n = 150
    time = np.ceil(rng.exponential(size=n) * 4) / 4 + 0.25
    status = rng.binomial(1, 0.7, n)
    entry = np.floor(rng.uniform(0, 0.5, n) * time * 8) / 8 if start else None
    strata = rng.integers(0, 3, n) if stratified else None
    w = rng.uniform(0.2, 3, n)
    pred = np.round(rng.standard_normal(n), 1)

    ro.globalenv['time'] = ro.FloatVector(time)
    ro.globalenv['status'] = ro.FloatVector(status)
    ro.globalenv['pred'] = ro.FloatVector(pred)
    ro.globalenv['w'] = ro.FloatVector(w)
    if start:
        ro.globalenv['entry'] = ro.FloatVector(entry)
        ro.r('y <- survival::Surv(entry, time, status)')
    else:
        ro.r('y <- survival::Surv(time, status)')
    if stratified:
        ro.globalenv['stratum'] = ro.IntVector(strata)
        formula = 'y ~ I(-pred) + survival::strata(stratum)'
    else:
        formula = 'y ~ I(-pred)'
    C_R = ro.r(f'survival::concordance({formula}, weights = w, timefix = FALSE)$concordance')[0]

    cox = CoxDeviance(event=time, status=status, start=entry, strata=strata)
    np.testing.assert_allclose(cox.c_index(pred, sample_weight=w), C_R, rtol=0, atol=1e-12)
    if not stratified:
        C_glmnet = ro.r('glmnet::Cindex(pred, y, weights = w)')[0]
        np.testing.assert_allclose(cox.c_index(pred, sample_weight=w), C_glmnet, rtol=0, atol=1e-12)
