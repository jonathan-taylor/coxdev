
import pytest

import numpy as np
import pandas as pd
from coxdev import (CoxDeviance,
                    _reversed_cumsums)

from simulate import (simulate_df, 
                      all_combos,
                      rng)

rng = np.random.default_rng(0)

@pytest.mark.parametrize('tie_types', all_combos)
@pytest.mark.parametrize('have_start_times', [True, False])
def test_coxph(tie_types,
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

        (X_event, 
         X_start) = _reversed_cumsums(X, 
                                      event_order=cox._event_order,
                                      start_order=cox._start_order)

        tmp = X_event[cox._first] - X_start[cox._event_map]
        cumsum_diff = np.zeros_like(tmp)
        cumsum_diff[cox._event_order] = tmp
        # -

        by_hand = []
        for i in range(data.shape[0]):
            if have_start_times:
                val = X[(data['event'] >= data['event'].iloc[i]) & (data['start'] < data['event'].iloc[i])].sum()
            else:
                val = X[(data['event'] >= data['event'].iloc[i])].sum()
            by_hand.append(val)
        by_hand = np.array(by_hand)
        assert np.allclose(by_hand * np.array(data['status']), cumsum_diff * np.array(data['status']))


