
import pytest

import numpy as np
import pandas as pd
from coxdev import CoxDeviance
#from coxdev.base import _reverse_cumsums
from coxdev.coxc import reverse_cumsums as _reverse_cumsums

from simulate import (simulate_df, 
                      all_combos,
                      rng)

rng = np.random.default_rng(0)

@pytest.mark.parametrize('tie_types', all_combos)
@pytest.mark.parametrize('have_start_times', [True, False])
def test_rev_cumsum(tie_types,
                    have_start_times,
                    nrep=5,
                    size=5,
                    tol=1e-10,
                    nsim=5):

    for _ in range(nsim):
        data = simulate_df(all_combos[-1],
                           nrep=3,
                           size=4,
                           rng=rng)
        data = data.reset_index().drop(columns='index')

        if have_start_times:
            cox = CoxDeviance(event=data['event'],
                              start=data['start'],
                              status=data['status'],
                              tie_breaking='efron')
        else:
            cox = CoxDeviance(event=data['event'],
                              start=None,
                              status=data['status'],
                              tie_breaking='efron')
            
        X = rng.standard_normal(data.shape[0])

        X_event = np.zeros(X.shape[0]+1)
        X_start = np.zeros(X.shape[0]+1)        

        _reverse_cumsums(X, 
                         X_event,
                         X_start,
                         cox._event_order.astype(np.int32),
                         cox._start_order.astype(np.int32),
                         True,  ## do_event = True
                         True)  ## do_start = True

        tmp = X_event[cox._first] - X_start[cox._event_map]
        cumsum_diff = np.zeros_like(tmp)
        cumsum_diff[cox._event_order] = tmp

        by_hand = []
        by_hand2 = []
        for i in range(data.shape[0]):
            if have_start_times:
                val = X[(data['event'] >= data['event'].iloc[i]) & (data['start'] < data['event'].iloc[i])].sum()
                val2 = X[(data['event'] >= data['event'].iloc[i])].sum() - X[(data['start'] >= data['event'].iloc[i])].sum()
            else:
                val = X[(data['event'] >= data['event'].iloc[i])].sum()
                val2 = X[(data['event'] >= data['event'].iloc[i])].sum()
            by_hand.append(val)
            by_hand2.append(val2)
        by_hand = np.array(by_hand)
        by_hand2 = np.array(by_hand2)
        assert np.allclose(by_hand * np.array(data['status']), cumsum_diff * np.array(data['status']))
        assert np.allclose(by_hand2 * np.array(data['status']), cumsum_diff * np.array(data['status']))


@pytest.mark.parametrize('tie_types', all_combos)
def test_event_start_maps(tie_types,
                          nrep=5,
                          size=5,
                          tol=1e-10,
                          nsim=1):

    for _ in range(nsim):
        data = simulate_df(all_combos[-1],
                           nrep=3,
                           size=4,
                           rng=rng)
        data = data.reset_index().drop(columns='index')

        cox = CoxDeviance(event=data['event'],
                          start=data['start'],
                          status=data['status'],
                          tie_breaking='efron')

        _status = np.asarray(data['status'])[cox._event_order]
        _event = np.asarray(data['event'])[cox._event_order]
        _start = np.asarray(data['start'])[cox._event_order]        

        _start_check = []
        _event_check = []
        by_hand = []

        X = rng.standard_normal(data.shape[0]) * data['status']

        for k in range(data.shape[0]):
            _event_check.append((data['start'] < _event[k]).sum())
            _start_check.append((data['event'] <= _start[k]).sum())
            by_hand.append(X[data['event'] <= data['event'].iloc[k]].sum() -
                           X[data['event'] <= data['start'].iloc[k]].sum())

        assert np.allclose(np.asarray(_event_check), cox._preproc['event_map'])
        assert np.allclose(np.asarray(_start_check), cox._preproc['start_map'])

        _X = X[cox._event_order]
        _cumsumX = np.cumsum(np.hstack([0, _X]))
        last = np.asarray(cox._preproc['last'])
        start_map = np.asarray(cox._preproc['start_map'])
        first_start = np.asarray(cox._first_start)

        assert np.allclose(first_start, start_map)
        tmp = (_cumsumX[last+1] -
               _cumsumX[start_map])
        cumsum_diff = np.zeros_like(tmp)
        cumsum_diff[cox._event_order] = tmp

        tmp2 = (_cumsumX[last+1] -
               _cumsumX[first_start])
        cumsum_diff2 = np.zeros_like(tmp2)
        cumsum_diff2[cox._event_order] = tmp2

        assert np.allclose(cumsum_diff, by_hand)
        assert np.allclose(cumsum_diff, cumsum_diff2)
    
