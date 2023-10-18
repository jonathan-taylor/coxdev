# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
from coxdev import (CoxDeviance,
                    _reversed_cumsums)

from simulate import (simulate_df, 
                      all_combos,
                      rng)
# -

data = simulate_df(all_combos[-1], nrep=3, size=4)
data = data.reset_index().drop(columns='index')
data

cox = CoxDeviance(event=data['event'],
                  start=data['start'],
                  status=data['status'],
                  tie_breaking='efron')
(data['event'] < data['start']).sum()

# +
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
    val = X[(data['event'] >= data['event'].iloc[i]) & (data['start'] < data['event'].iloc[i])].sum()
    by_hand.append(val)
by_hand = np.array(by_hand)
assert np.allclose(by_hand * np.array(data['status']), cumsum_diff * np.array(data['status']))


