"""
Fixtures for the Julia wrapper's tests (julia/CoxDev.jl/test/fixtures/):
the inputs and the Python package's outputs on a grid of cases, as JSON.

Grid: Efron/Breslow x right-censored/(start, event] x no strata/strata x
unit/random/zero weights, with tied times. For each case: the deviance, its
gradient and diagonal Hessian, information products, the baseline hazard,
the C-index, score residuals, wild draws and sample_score draws (fixed seed).

Usage: python tools/make_julia_fixtures.py [outdir]
"""
import json
import os
import sys
from itertools import product

import numpy as np

from coxdev import CoxDeviance

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'julia', 'CoxDev.jl', 'test', 'fixtures')

SEED = 20261017
N, P, NDRAWS = 60, 3, 4


def _list(x):
    x = np.asarray(x)
    if x.ndim == 2:
        # column-major, as Julia reads a vector of columns
        return [_list(x[:, j]) for j in range(x.shape[1])]
    return [None if (isinstance(v, float) and np.isnan(v)) else v for v in x.tolist()]


def case(ties, start_stop, strata, weights, rng):
    # tied integer times
    event = rng.integers(1, 15, N).astype(float)
    status = rng.binomial(1, 0.7, N)
    start = None
    if start_stop:
        start = np.maximum(event - rng.integers(1, 8, N), 0.).astype(float)
        start[rng.random(N) < 0.3] = 0.
    labels = None
    if strata == 'int':
        labels = rng.choice([3, 7, 11], N)          # non-contiguous integer labels
    elif strata == 'str':
        labels = rng.choice(['b', 'a', 'c'], N)     # string labels, unsorted
    if weights == 'unit':
        w = None
    elif weights == 'random':
        w = rng.uniform(0.5, 2., N)
    else:
        w = rng.uniform(0.5, 2., N)
        w[rng.random(N) < 0.2] = 0.
    eta = 0.5 * rng.standard_normal(N)
    V = rng.standard_normal((N, 2))
    X = rng.standard_normal((N, P))
    G = rng.standard_normal(N)
    D = rng.standard_normal(N)          # a direction, for the information derivative

    cox = CoxDeviance(event=event, status=status, start=start, strata=labels,
                      sample_weight=w, tie_breaking=ties)
    res = cox(eta, w)
    info = cox.information(eta, w)
    bh = cox.baseline_hazard(eta, w)
    out = dict(deviance=res.deviance, gradient=_list(res.gradient), diag_hessian=_list(res.diag_hessian),
               loglik_sat=res.loglik_sat,
               information_V=_list(np.column_stack([info @ V[:, j] for j in range(V.shape[1])])),
               information_derivative_V=_list(cox.information_derivative(eta, D, w) @ V),
               bh_time=_list(bh.time), bh_hazard=_list(bh.hazard),
               bh_cumulative_hazard=_list(bh.cumulative_hazard), bh_stratum=_list(bh.stratum),
               c_index=float(cox.c_index(eta, w)),
               score_residuals=_list(cox.score_residuals(eta, X, w)),
               wild_score=_list(cox.wild_score(eta, G, w)))
    # weight=None means ones, whatever the weights given at construction
    out['deviance_default_weight'] = cox(eta).deviance
    # model-based draws need equal positive weights; wild works with any
    methods = ['wild'] + (['multinomial', 'bernoulli', 'fisher'] if weights == 'unit' else [])
    out['sample_score'] = {m: _list(cox.sample_score(eta, w, method=m, n_draws=NDRAWS, seed=SEED))
                           for m in methods}
    inputs = dict(event=_list(event), status=_list(status), start=None if start is None else _list(start),
                  strata=None if labels is None else _list(labels), weight=None if w is None else _list(w),
                  ties=ties, eta=_list(eta), V=_list(V), X=_list(X), G=_list(G), D=_list(D), seed=SEED, n_draws=NDRAWS)
    name = f'{ties}_{"startstop" if start_stop else "right"}_{strata or "nostrata"}_{weights}'
    return name, dict(name=name, inputs=inputs, outputs=out)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    names = []
    for ties, ss, strata, weights in product(['efron', 'breslow'], [False, True],
                                             [None, 'int', 'str'], ['unit', 'random', 'zero']):
        name, data = case(ties, ss, strata, weights, rng)
        with open(os.path.join(OUT, name + '.json'), 'w') as f:
            json.dump(data, f)
        names.append(name)
    print(f'wrote {len(names)} fixtures to {os.path.abspath(OUT)}')


if __name__ == '__main__':
    main()
