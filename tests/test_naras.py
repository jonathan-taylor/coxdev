
import pytest

import numpy as np
import pandas as pd
from coxdev import CoxDeviance
from coxdev.base import _reverse_cumsums as rt
from coxc import reverse_cumsums as _reverse_cumsums

from simulate import (simulate_df, 
                      all_combos,
                      rng)

rng = np.random.default_rng(0)

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
        print(f"X is {X.dtype}");
        print(f"X_event is {X_event.dtype}");
        print(f"X_start is {X_start.dtype}");
        print(f"event_order is {cox._event_order.dtype}");
        print(f"start_order is {cox._start_order.dtype}");
        
        _reverse_cumsums(X, 
                         X_event,
                         X_start,
                         cox._event_order.astype(np.int32),
                         cox._start_order.astype(np.int32),
                         True,
                         True)
        
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


test_rev_cumsum(tie_types = all_combos[100], have_start_times = True, nsim = 1)
