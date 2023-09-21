# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd

from coxdev import _cox_dev
from coxdev import _preprocess
from coxdev import CoxDeviance
                       

# -

# # A small dataset (with ties)
#
# - Event times (or "stop" times) $t_i$.
#
# - Start times: $s_i$.

data_df = pd.read_csv('dataset.csv', index_col='index')[:20]
data_df
n = data_df.shape[0]

# ## Key sequences
#
# We will jointly sort `start` and `event` by stacking the times into
# a data frame with 60 rows instead of 30.
#
# We will sort this on `(time, status, is_start)` with `status==1` occurring before
# `status==0`.

stacked_df = pd.DataFrame({'time':np.hstack([data_df['start'], data_df['event']]),
                           'status':np.hstack([np.zeros_like(data_df['start']), 
                                               data_df['status']]),
                           'is_start':np.hstack([np.ones(n, int), np.zeros(n, int)]),
                           'index':np.hstack([np.arange(n), np.arange(n)])})
stacked_df

# From this joint sort we will define several key sequences of length 30.
#
# - `event_order`: sorts event times into increasing order (according to this joint sort)
# - `start_order`: sorts start times into increasing order (according to this joint sort)
# - `first`: for a failure time, which may have ties, this is the first entry of `event_order` in this set of ties -- the entries of `first` that are not failure times are not used, but they are defined consistently in that no ties of non-failure times are material to the computations.
# - `last`: analogous to `first` but the last time
# - `event_map`: for the $i$-th event time, this is the number of start times stricly less than this event time: $\text{event\_map}[i] = \# \{j: s_j < t_i\}$.
# - `start_map`: for the $i$-th start time, this is the number of event times less than or equal to $s_i$:
#   $\text{start\_map}[i] = \# \{j: t_j \leq s_i\}$.
#
# **These sequences are computed once.**

# ### Computing the key sequences
#
# We sort the stacked data frame, placing "failures" (`status==1`) in front, and `start` times following `event` times.
#
# - This sorted data frame makes it easy to compute `start_order` and `event_order`.
# - Computing `first` requires keeping track of successive `event` times that have `status==1`, continuing to append a current beginning of the tie sequence until a change is detected.
# - Computing `event_map` and `start_map` requires keeping track of the current number of events (or starts) and appending this to `start_map` or `event_map`. This will yield `start_map` (in start order) and `event_map` (in event order). These can reordered to the original index. Later, we will set everything to event order.

sorted_df = stacked_df.sort_values(by=['time', 'status', 'is_start'], ascending=[True,False,True])
sorted_df


def sort_start_event(original_df, sorted_df):
    event_count, start_count = 0, 0
    event_order, start_order = [], []
    start_map, event_map = [], []
    first = []
    event_idx = []
    last_row = None
    which_event = -1
    first_event = -1
    num_successive_event = 1
    ties = {}    
    for _r in range(sorted_df.shape[0]):
        row = sorted_df.iloc[_r]
        if row['is_start'] == 1: # a start time
            start_order.append(row['index'])
            start_map.append(event_count)
            start_count += 1
        else: # an event / stop time
            if row['status'] == 1:
                # if it's an event and the time is same as last row 
                # it is the same event
                # else it's the next "which_event"
                
                if (last_row is not None and 
                    row['time'] != last_row['time']): # index of next `status==1`
                    first_event += num_successive_event
                    num_successive_event = 1
                    which_event += 1
                else:
                    num_successive_event += 1
                    
                first.append(first_event)
            else:
                first_event += num_successive_event
                num_successive_event = 1
                first.append(first_event) # this event time was not an failure time

            event_map.append(start_count)
            event_order.append(row['index'])
            event_count += 1
        last_row = row

    first = np.array(first)
    start_order = np.array(start_order).astype(int)
    event_order = np.array(event_order).astype(int)
    start_map = np.array(start_map)
    event_map = np.array(event_map)

    # reset start_map to original order
    start_map_cp = start_map.copy()
    start_map[start_order] = start_map_cp

    status_event = np.asarray(original_df['status'])[event_order]
    preprocessed_df = pd.DataFrame({'status':status_event,
                                    'first':first,
                                    'start_map':start_map[event_order].astype(int), 
                                    'event_map':event_map.astype(int) # already in event order
                                    }, index=event_order)
    print(preprocessed_df.index)

    # compute `last`
    
    last = []
    last_event = n-1
    for i, f in enumerate(preprocessed_df['first'][::-1]):
        last.append(last_event)
        # immediately following a last event, `first` will agree with np.arange
        if f - (n - 1 - i) == 0:
            last_event = f - 1        

    preprocessed_df.insert(2, 'last', last[::-1])
    

    return preprocessed_df, event_order, start_order

(preproc, 
 event_order, 
 start_order) = sort_start_event(data_df, sorted_df)
preproc # note that preproc.index == event_order

# Let's add the sorted `event`, `start` and `status` so we can go through some of these
# calculations.

preproc['event'] = data_df['event'][event_order]
preproc['start'] = data_df['start'][event_order]
preproc

# Let's look at row with index `11` above. The value of `7` for `start_map` means that this individual's `start` time was
# such that they were not in the risk set for the first 7 event times. Indeed, the `start` time is `4` and there are 7 event times less than or equal to 4.
#
# The value of `event_map` is 20. This means there were 20 start times less than the event time which is `6`.
#

(data_df['start'] < 6).sum()

# # Evaluation
#
# ## Saturated log-likelihood
#
# First, let's compute the saturated log-likelihood. This is also a pre-processing step.

sat_df = preproc[['first', 'status']].copy() # in event order
sat_df['sample_weight'] = data_df['weight'][event_order]
loglik_sat = 0
for _, df in sat_df[['first', 'status', 'sample_weight']].groupby('first'):
    if df['status'].sum() > 0:
        W = (df['sample_weight'] * df['status']).sum()
        loglik_sat -= W * np.log(W)


# ## Efron's tie breaking method
#
# For this, we need yet another array of shape `(n,)`. This vector can be
# computed without `eta`. When there are no ties, this vector is identically 0.
#

den = preproc['last'] + 1 - preproc['first']
preproc['scaling'] = (np.arange(n) - preproc['first']) / den

# The columns in `preproc` are:

preproc.columns

# The data frame `preproc` can be computed without reference to `eta`. Its columns
# will be the arguments to the core `C` computation.

# With all of the above preprocessing done, we can now define the relevant
# reversed cumsums in the risk sum: `E` for event, `S` for start and `M` for (scaled) mean:
# $$
# \begin{aligned}
# E_i(\eta) &= \sum_{j:j \geq i} w_j e^{\eta_j} \\
# &= \sum_{j:t_j \geq t_i} w_j e^{\eta_j} \\
# S_i(\eta) &= \sum_{j: s_j > t_i} w_j e^{\eta_j} \\
# M_i(\eta) &= \text{scaling}(i)  \cdot \left(E_{\text{first}(i)}(\eta) - E_{\text{last}(i)+1}(\eta)\right) \\
# &\overset{\text{def}}{=} \sigma_E(i)  \cdot \left(E_{\text{first}(i)}(\eta) - E_{\text{last}(i)+1}(\eta)\right).
# \end{aligned}
# $$
# The risk sums for start / event data with Efron's tie-breaking are
# $$
# R_i(\eta) = E_{\text{first}(i)}(\eta) - S_i(\eta) - M_i(\eta).
# $$
#
# Introducing indicators $\delta_S$ for whether start times are present (otherwise equal to $-\infty)$ and 
# $\delta_E$ for whether we use Efron's tie correction, we can include all 4 cases with the risk sum
# $$
# R_i(\eta) = E_i(\eta) - \delta_S \cdot S_i(\eta) - \delta_E \cdot M_i(\eta).
# $$
#
# The log-likelihood (in all four cases) is defined to be
# $$
# \ell(\eta) = \sum_i w_i d_i \left(\eta_i - \log(R_i(\eta))\right)
# $$

# The following derivatives are useful for computing the gradient and Hessian of the log-likelihood:
# $$
# \begin{aligned}
# \frac{\partial}{\partial \eta_k} E_i(\eta) &= w_k e^{\eta_k} \cdot 1_{\{k \geq \text{first}(i)\}} \\
# &= w_k e^{\eta_k} \cdot 1_{\{i \leq \text{last}(k)\}} \\
# \frac{\partial}{\partial \eta_k} S_i(\eta) &= w_k e^{\eta_k}  \cdot 1_{\{s_k > t_i\}} \\
# &= w_k e^{\eta_k} \cdot 1_{\{i < \text{start\_map}(k)\}} \\
# &= w_k e^{\eta_k} \cdot 1_{\{i \leq \text{start\_map}(k)-1\}} \\
# \frac{\partial}{\partial \eta_k} M_i(\eta) &= w_k e^{\eta_k} \cdot \text{scaling}_i \cdot 1_{[\text{first}(i), \text{last}(i)]}(k) \\
# &\overset{\text{def}}{=} w_k e^{\eta_k} \cdot \sigma_E(i) \cdot \left( 1_{\{i \leq  \text{last}(k)\}} - 1_{\{i \leq \text{first}(k)-1\}}\right)
# \end{aligned}
# $$
#

# ### Gradient
#
# Let's compute the derivative 
# $$
# \begin{aligned}
# \frac{\partial}{\partial \eta_k} \ell(\eta) &= 
# w_k d_k - w_k e^{\eta_k} \cdot \sum_{i=1}^n w_i d_i \cdot \frac{1_{\{i \leq \text{last}(k)\}} - \delta_S \cdot 1_{\{i \leq \text{start\_map}(k)-1 \}}
# - \delta_E \cdot \sigma_E(i) \cdot \left( 1_{\{i \leq \text{last}(k)\}} - 1_{\{i \leq \text{first}(k)-1\}}\right)}{R_i(\eta)}
# \end{aligned}
# $$
#
# We see this can be expressed cumsums of the sequences
# $$
# \frac{w_i d_i}{R_i(\eta)}, \qquad \frac{w_i d_i \sigma_E(i)}{R_i(\eta)}.
# $$
#
# Define
# $$
# {\cal C}_{rs}(\eta)[i] = \sum_{j=1}^i \frac{w_i d_i \sigma_E(i)^s}{R_i(\eta)^r}.
# $$

# Our derivative above is
# $$
# w_k d_k - w_k e^{\eta_k} \left({\cal C}_{10}(\eta)[\text{last}(k)] - \delta_S \cdot {\cal C}_{10}(\eta)[\text{start\_map}(k)-1] - \delta_E \cdot \left({\cal C}_{11}(\eta)[\text{last}(k)] - {\cal C}_{11}(\eta)[\text{first}(k)-1] \right) \right)
# $$

# ### Hessian
#
# Let's compute the second
# derivative
# $$
# \frac{\partial^2}{\partial \eta_l \eta_k} \ell(\eta)
# $$
# consists of two terms. The first is the diagonal
# $$
# T_{1,kl}(\eta) = - \delta_{lk} w_ke^{\eta_k} \left({\cal C}_{10}(\eta)[\text{last}(k)] - \delta_S \cdot {\cal C}_{10}(\eta)[\text{start\_map}(k)-1] - \delta_E \cdot \left({\cal C}_{11}(\eta)[\text{last}(k)] - {\cal C}_{11}(\eta)[\text{first}(k)-1] \right) \right)
# $$

# The second is
# $$
# \begin{aligned}
# T_{2,kl}(\eta) &= \sum_{i=1}^n w_i d_i \biggl[\frac{\left(1_{\{i \leq \text{last}(k)\}} - \delta_S \cdot 1_{\{i \leq \text{start\_map}(k) -1\}}
# - \delta_E \cdot \sigma_E(i) \cdot \left( 1_{\{i \leq \text{last}(k)\}} - 1_{\{i \leq \text{first}(k)-1\}}\right)\right) }{R_i(\eta)} \\
#  & \qquad \qquad \times \frac{\left(1_{\{i \leq \text{last}(l)\}} - \delta_S \cdot 1_{\{i \leq \text{start\_map}(l) -1\}}
# - \delta_E \cdot \sigma_E(i) \cdot \left( 1_{\{i \leq \text{last}(l)\}} - 1_{\{i \leq \text{first}(l)-1\}}\right)\right) }{R_i(\eta)}
#   \biggr]
# \end{aligned}
# $$

# Without belaboring the expansion just yet, it is clear that this can be expressed in terms of
# $$
# \left({\cal C}_{rs}(\eta)\right)_{1 \leq r \leq 2, 1 \leq s \leq r}.
# $$
#
# ### Diagonal of Hessian
#
# The diagonal terms $T_{2,kk}(\eta)$ are
# $$
# \begin{aligned}
# T_{2,kk}(\eta) &= \sum_{i=1}^n w_i d_i \frac{\left(1_{\{i \leq \text{last}(k)\}} - \delta_S \cdot 1_{\{i \leq \text{start\_map}(k)-1 \}}
# - \delta_E \cdot \sigma_E(i) \cdot \left( 1_{\{i \leq \text{last}(k)\}} - 1_{\{i \leq \text{first}(k)-1\}}\right)\right)^2 }{R_i(\eta)^2} 
# \end{aligned}
# $$
#
#
#

# #### Expansion
#
# Let's expand the numerator in $T_{2,kl}(\eta)$ which will tell us which indices of the
# relevant cumsums to use.
# $$
# \begin{aligned}
# N_{kl} &= \left(1_{\{i \leq \text{last}(k)\}} - \delta_S \cdot 1_{\{i \leq \text{start\_map}(k)-1 \}}
# - \delta_E \cdot \sigma_E(i) \cdot \left( 1_{\{i \leq \text{last}(k)\}} - 1_{\{i \leq \text{first}(k)-1\}}\right)\right) \\
# & \qquad \times \left(1_{\{i \leq \text{last}(l)\}} - \delta_S \cdot 1_{\{i \leq \text{start\_map}(l)-1 \}}
# - \delta_E \cdot \sigma_E(i) \cdot \left( 1_{\{i \leq \text{last}(l)\}} - 1_{\{i \leq \text{first}(l)-1\}}\right)\right)
# \end{aligned}
# $$ 
#
# It will be handy to use a symbolic manipulation tool for this as there are many terms.

# +
from sympy import Symbol, Function, simplify, expand

last_ = Function('last')
start_ = Function('start')
first_ = Function('first')
k = Symbol('k')
l = Symbol('l')
s_E = Symbol('sigma_E')
d_E = Symbol('delta_E')
d_S = Symbol('delta_S')

E_k = last_(k) - d_S * start_(k-1) - d_E * s_E * (last_(k) - first_(k-1))
# -

diag_prod = expand(E_k * E_k)
diag_prod.as_ordered_terms()

# This list of terms can be used to deduce which ${\cal C}_{rs}(\eta)$ are needed
# to compute the diagonal. Of course we use the logic that `delta_E**2=delta_E`,
# `delta_S**2=delta_S` and whenever we see terms like $\text{last}(k)*\text{first}(k-1)$ or
# similar this becomes
# $$
# 1_{\{i \leq \min(\text{last}(k), \text{first}(k)-1)\}} = 1_{\{i \leq \text{first}(k)-1\}}.
# $$
#
# If we were clever, we could try to take the above symbolic representation and turn it into 
# correct LaTeX but it is not too painful to do by hand.

# #### Simplification: right censored with Breslow's method
#
# By setting $\delta_S$ or $\delta_E$ to 0, these expressions symplify considerably.
# Let's try setting $\delta_S=\delta_E=0$. This is right-censored data with Breslow's tie-breaking.
# This has no $\sigma_E$ in it, so we know we only need ${\cal C}_{10}$ and ${\cal C}_{20}$ to compute it.

diag_prod.subs(d_E,0).subs(d_S,0).as_ordered_terms()

# From this we deduce that the diagonal entries for right-censored data with
# Breslow's tie breaking are:
# $$
# T_{2,kk}(\eta) = {\cal C}_{20}(\eta)[\text{last}(k)]
# $$

# #### Simplification: start times with Breslow's method
#
# Let's do Breslow's tie breaking with start times.

diag_prod.subs(d_E,0).subs(d_S, 1).as_ordered_terms()

# As $\text{start}(k) \leq \text{last}(k)$ we see that, in this case
# $$
# T_{2,kk}(\eta) = C_{20}(\eta)[\text{last}(k)] - C_{20}(\eta)[\text{start}(k)-1].
# $$

# #### Simplification: right-censored with Efron's tie-breaking:

diag_prod.subs(d_S,0).subs(d_E, 1).as_ordered_terms()

# This has powers $\sigma_E^{\{0,1,2\}}$ so we will need all 5 ${\cal C}_{rs}$ cumsums to evaluate
# it and the gradient.
# For this case,
# $$
# \begin{aligned}
# T_{2,kk}(\eta) &= {\cal C}_{22}(\eta)[\text{first}(k)-1] \\
# & -2 \cdot {\cal C}_{22}(\eta)[\text{first}(k)-1] \\
# & + {\cal C}_{22}(\eta)[\text{last}(k)] \\
# & + 2 \cdot {\cal C}_{21}(\eta)[\text{first}(k)-1] \\
# & -2 \cdot {\cal C}_{21}(\eta)[\text{last}(k)] \\
# & + {\cal C}_{20}(\eta)[\text{last}(k)] \\
# &= ({\cal C}_{22}(\eta)[\text{last}(k)] - {\cal C}_{22}(\eta)[\text{first}(k)-1]) \\
# & - 2 \cdot ({\cal C}_{21}(\eta)[\text{last}(k)] - {\cal C}_{21}(\eta)[\text{first}(k)-1]) \\
# & + {\cal C}_{20}(\eta)[\text{last}(k)] \\
# \end{aligned}
# $$

# #### General case

diag_prod.subs(d_S,1).subs(d_E, 1).as_ordered_terms()

# This will have a few more terms than previous. Let's just see which they are.
#

full = set(diag_prod.subs(d_S,1).subs(d_E, 1).as_ordered_terms())
partial = set(diag_prod.subs(d_S,0).subs(d_E, 1).as_ordered_terms())
assert partial.issubset(full)

full.difference(partial)

# By appending these terms to those above,  and we note  that $\text{start}(k)-1 \leq \text{first}(k)-1$ (Why? Because $s_k$ is strictly less than $t_k$ and $t_{\text{first}(k)-1}$ is the event time immediately preceeding $t_k$.)
# Therefore
# $$
# \begin{aligned}
# T_{2,kk}(\eta)  &= ({\cal C}_{22}(\eta)[\text{last}(k)] - {\cal C}_{22}(\eta)[\text{first}(k)-1]) \\
# & - 2 \cdot ({\cal C}_{21}(\eta)[\text{last}(k)] - {\cal C}_{21}(\eta)[\text{first}(k)-1]) \\
# & + {\cal C}_{20}(\eta)[\text{last}(k)] \\
# & - 2 \cdot {\cal C}_{20}(\eta)[\text{start}(k)-1] \\
# & - 2 \cdot {\cal C}_{21}(\eta)[\text{start}(k)-1] \\
# & + 2 \cdot {\cal C}_{21}(\eta)[\text{start}(k)-1] \\
# & + {\cal C}_{20}(\eta)[\text{start}(k)-1] \\
# &=  ({\cal C}_{22}(\eta)[\text{last}(k)] - {\cal C}_{22}(\eta)[\text{first}(k)-1]) \\
# & - 2 \cdot ({\cal C}_{21}(\eta)[\text{last}(k)] - {\cal C}_{21}(\eta)[\text{first}(k)-1]) \\
# & + {\cal C}_{20}(\eta)[\text{last}(k)] - {\cal C}_{20}(\eta)[\text{start}(k)-1]
# \end{aligned}
# $$

assert np.all(preproc['start'] <= preproc['first'])

# #### Off-diagonal Hessian entries
#
# We can similarly deduce how to evaluate off-diagonal entries of the Hessian, though
# we have not implemented all of these yet.

E_l = last_(l) - d_S * start_(l) - d_E * s_E * (last_(l) - first_(l-1))
prod = expand(E_k * E_l)
prod.as_ordered_terms()


# ## Evaluation
# Below is the evaluation in `python` code that is similar to what the `C` code will look like.

# +

def log_like(eta,           # eta is in native order 
             sample_weight, # sample_weight is in native order
             event_order,   
             start_order,
             status,        # everything below in event order
             event,
             start,
             first,
             last,
             scaling,
             event_map,
             start_map,
             loglik_sat,
             have_start_times=True,
             efron=False):

    # be sure they're arrays so that no weird pandas indexing is used
    eta = np.asarray(eta)
    sample_weight = np.asarray(sample_weight)
    event_order = np.asarray(event_order)   
    start_order = np.asarray(start_order)
    status = np.asarray(status)
    event = np.asarray(event)
    start = np.asarray(start)
    first = np.asarray(first)
    last = np.asarray(last)
    scaling = np.asarray(scaling)
    event_map = np.asarray(event_map)
    start_map = np.asarray(start_map)
    
    _status = (status==1)
    
    eta = eta - eta.mean()
    
    # compute the event ordered reversed cumsum
    eta_event = eta[event_order]
    w_event = sample_weight[event_order]
    exp_eta_w_event = w_event * np.exp(eta_event)
    event_cumsum = np.hstack([np.cumsum(exp_eta_w_event[::-1])[::-1], 0]) # length=n+1 for when last=n-1

    # compute the start oredered reversed cumsum, if necessary
    # then compute the cumsums (or difference of cumsums) for Breslow approximation
    
    if have_start_times:
        exp_eta_w_start = np.hstack([(sample_weight * np.exp(eta))[start_order], 0]) # length=n+1
        start_cumsum = np.cumsum(exp_eta_w_start[::-1])[::-1]  # length=n+1
        risk_sums = event_cumsum[first] - start_cumsum[event_map]
    else:
        risk_sums = event_cumsum[first]
        
    # compute the Efron correction, adjusting risk_sum if necessary
    
    if efron == True:
        # XXXXX is last term handled correctly?
        n = eta.shape[0]
        num = (event_cumsum[first] - 
               event_cumsum[last+1])
        risk_sums -= num * scaling
    
    log_terms = np.log(np.array(risk_sums)) * w_event * _status
    loglik = (w_event * eta_event * _status).sum() - np.sum(log_terms)

    # cumsums for gradient and Hessian
    
    # length of cumsums is n+1
    # 0 is prepended for first(k)-1, start(k)-1 lookups
    # a 1 is added to all indices

    A_10 = _status * w_event / risk_sums
    C_10 = np.hstack([0, np.cumsum(A_10)]) 
    
    A_20 = _status * w_event / risk_sums**2
    C_20 = np.hstack([0, np.cumsum(A_20)]) # length=n+1

    # if there are no ties, scaling should be identically 0
    # don't bother with cumsums below 

    if not efron:
        if have_start_times:
            T_1_term = C_10[last+1] - C_10[start_map]   # +1 for start_map? depends on how  
                                                         # a tie between a start time and an event time
                                                         # if that means the start individual is excluded
                                                         # we should add +1, otherwise there should be
                                                         # no +1 in the [start_map+1] above
            T_2_term = C_20[last+1] - C_20[start_map]
        else:
            T_1_term = C_10[last+1]
            T_2_term = C_20[last+1]
    else:
        # compute the other necessary cumsums
        
        A_11 = _status * w_event * scaling / risk_sums
        C_11 = np.hstack([0, np.cumsum(A_11)]) # length=n+1

        A_21 = _status * w_event * scaling / risk_sums
        C_21 = np.hstack([0, np.cumsum(A_21)]) # length=n+1

        A_22 = _status * w_event * scaling / risk_sums
        C_22 = np.hstack([0, np.cumsum(A_22)]) # length=n+1

        T_1_term = (C_10[last+1] - 
                    (C_11[last+1] - C_11[first]))
        T_2_term = ((C_22[last+1] - C_22[first]) 
                    - 2 * (C_21[last+1] - C_21[first]) + 
                    C_20[last+1])
        if have_start_times:
            T_1_term -= C_10[start_map]
            T_2_term -= C_20[first]
    
    grad = w_event * _status - exp_eta_w_event * T_1_term
    grad_cp = grad.copy()
    grad[event_order] = grad_cp

    # now the diagonal of the Hessian

    diag_hess = exp_eta_w_event**2 * T_2_term - exp_eta_w_event * T_1_term
    diag_hess_cp = diag_hess.copy()
    diag_hess[event_order] = diag_hess_cp

    deviance = 2 * (loglik_sat - loglik)
    return deviance, -2 * grad, -2 * diag_hess


# -

eta = data_df['eta'] # in native order
print(loglik_sat)
_, dev, G, H = _cox_dev(eta,
                        data_df['weight'],
                        event_order,
                        start_order,
                        preproc['status'],
                        preproc['event'],
                        preproc['start'],
                        preproc['first'],
                        preproc['last'],
                        preproc['scaling'],
                        preproc['event_map'],
                        preproc['start_map'],
                        loglik_sat,
                        efron=False,
                        have_start_times=True)

import rpy2
# %load_ext rpy2.ipython
start = data_df['start'].copy()
event = data_df['event'].copy()
status = data_df['status'].copy()
weight = data_df['weight'].copy()
# %R -i start,event,status,eta,weight

# + magic_args="-o G_R,H_R,D_R" language="R"
# library(survival)
# library(glmnet)
# Y = Surv(start, event, status)
# D_R = glmnet:::coxnet.deviance3(pred=eta, y=Y, weight=weight, std.weights=FALSE)
# # glmnet computes grad and hessian of the log-likelihood, not deviance
# # need to multiply by -2 to get grad and hessian of deviance
# G_R = glmnet:::coxgrad3(eta, Y, weight, std.weights=FALSE, diag.hessian=TRUE)
# H_R = attr(G_R, 'diag_hessian')
# G_R = -2 * G_R
# H_R = -2 * H_R
# -

np.fabs(dev - D_R)

np.linalg.norm(G-G_R)/ np.linalg.norm(G)

np.linalg.norm(H-H_R) / np.linalg.norm(H)

# ## Using `CoxDeviance`

coxdev_ = CoxDeviance(event,
                      status,
                      start=start,
                      tie_breaking='breslow')
_, dev_, G_, H_ = coxdev_(eta, 
                          weight)

np.fabs(dev - dev_)

np.linalg.norm(G-G_)/ np.linalg.norm(G)

np.linalg.norm(H-H_) / np.linalg.norm(H)

# ## Larger data sets

ties = True
n = 100000
rng = np.random.default_rng(0)
start = rng.integers(0, 10, size=n) 
event = start + rng.integers(0, 10, size=n) + ties + (1 - ties) * rng.standard_exponential(n) * 0.01
status = rng.choice([0,1], size=n, replace=True)
weight = rng.uniform(1, 2, size=n)
eta = rng.standard_normal(n)

# + magic_args="-i event,status,start,eta,weight -o G_R,H_R,D_R " language="R"
# Y = Surv(start, event, status)
# eta = as.numeric(eta)
# weight = as.numeric(weight)
# print(system.time(for (i in 1:400) {c(glmnet:::coxnet.deviance3(pred=eta, y=Y, weight=weight, std.weights=FALSE),
#                     glmnet:::coxgrad3(eta, Y, weight, std.weights=FALSE, diag.hessian=TRUE))}))
# D_R = glmnet:::coxnet.deviance3(pred=eta, y=Y, weight=weight, std.weights=FALSE)
# G_R = glmnet:::coxgrad3(eta, Y, weight, std.weights=FALSE, diag.hessian=TRUE)
# H_R = attr(G_R, 'diag_hessian')
# G_R = -2 * G_R
# H_R = -2 * H_R
# -

# %%timeit
coxdev_ = CoxDeviance(event,
                      status,
                      start=start,
                      tie_breaking='breslow')
loglik_sat, dev, G, H = coxdev_(eta, weight)
[coxdev_(eta, weight, loglik_sat=loglik_sat) for _ in range(400)]

coxdev_ = CoxDeviance(event,
                      status,
                      start=start,
                      tie_breaking='breslow')
_, dev_, G_, H_ = coxdev_(eta, 
                          weight)

np.fabs(dev_ - D_R)

np.linalg.norm(G_-G_R) / np.linalg.norm(G)

np.linalg.norm(H_-H_R) / np.linalg.norm(H)

# ## Right censored 

# %%timeit
coxdev_ = CoxDeviance(event,
                      status,
                      tie_breaking='breslow')
loglik_sat, dev, G, H = coxdev_(eta, weight)
[coxdev_(eta, weight, loglik_sat=loglik_sat) for _ in range(400)]

# + magic_args="-i event,status,start,eta,weight -o G_R,H_R,D_R " language="R"
# Y = Surv(event, status)
# eta = as.numeric(eta)
# weight = as.numeric(weight)
# print(system.time(for (i in 1:400) {c(glmnet:::coxnet.deviance2(pred=eta, y=Y, weight=weight, std.weights=FALSE),
#                     glmnet:::coxgrad2(eta, Y, weight, std.weights=FALSE, diag.hessian=TRUE))}))
# D_R = glmnet:::coxnet.deviance2(pred=eta, y=Y, weight=weight, std.weights=FALSE)
# G_R = glmnet:::coxgrad2(eta, Y, weight, std.weights=FALSE, diag.hessian=TRUE)
# H_R = attr(G_R, 'diag_hessian')
# G_R = -2 * G_R
# H_R = -2 * H_R
# -

# ## Efron's tie breaking
#

# %%timeit
coxdev_ = CoxDeviance(event,
                      status,
                      tie_breaking='efron')
loglik_sat, dev, G, H = coxdev_(eta, weight)
[coxdev_(eta, weight, loglik_sat=loglik_sat) for _ in range(400)]

# ### With start time

# %%timeit
coxdev_ = CoxDeviance(event,
                      status,
                      start=start,
                      tie_breaking='efron')
loglik_sat, dev, G, H = coxdev_(eta, weight)
[coxdev_(eta, weight, loglik_sat=loglik_sat) for _ in range(400)]


