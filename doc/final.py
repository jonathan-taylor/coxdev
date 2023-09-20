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

rng = np.random.default_rng(0)
# -

# ## Since `stop` is when the status is recorded, I've renamed `stop` as `event` here
#
# ## `status` refers to the status of the individual at the `event` / `stop` time

n = 30
ties = True
start = rng.integers(0, 6, size=n) 
event = (start + rng.integers(2, 6, size=n) + (1 - ties) * rng.standard_exponential(n) * 0.01).astype(int)
status = rng.choice([0,1], size=n, replace=True)
data = pd.DataFrame({'start':start,
                     'event':event,
                     'status':status})
data.to_csv('dataset.csv')
data

order = False
if order:
    event_order = np.argsort(event)
    event = event[event_order]
    start = start[event_order]
    status = status[event_order]

df = pd.DataFrame({'start':start, 'event':event, 'status':status})

df.iloc[:20]

stacked_df = pd.DataFrame({'time':np.hstack([start, event]),
                           'status':np.hstack([np.zeros_like(start), status]),
                           'start?':np.hstack([np.ones(n), np.zeros(n)]),
                           'idx':np.hstack([np.arange(n), np.arange(n)])})
stacked_df[:20]

stacked_df.iloc[(n-10):(n+10)]

sorted_df = stacked_df.sort_values(by=['time', 'status', 'start?'], ascending=[True,False,True])
sorted_df[:20]


# +
def sort_start_event(sorted_df, sample_weight=None):
    event_count, start_count = 0, 0
    event_order, start_order = [], []
    start_cum_count, event_cum_count = [], []
    event_first = []
    event_idx = []
    last_row = None
    which_event = -1
    first_event = -1
    num_successive_event = 1
    ties = {}    
    for _r in range(sorted_df.shape[0]):
        row = sorted_df.iloc[_r]
        _r += 1
        if row['start?'] == 1: # a start time
            start_order.append(row['idx'])
            start_cum_count.append(event_count)
            start_count += 1
        else: # an event / stop time
            if row['status']:
                # if it's an event and the time is same as last row 
                # it is the same event
                # else it's the next "which_event"
                
                if (last_row is not None and 
                    row['time'] != last_row['time']): # index of next death
                    first_event += num_successive_event
                    num_successive_event = 1
                    which_event += 1
                else:
                    num_successive_event += 1
                    
                event_first.append(first_event)
            else:
                first_event += num_successive_event
                num_successive_event = 1
                event_first.append(first_event) # this event time was not an failure time

            event_idx.append(which_event)
            event_cum_count.append(start_count)
            event_order.append(row['idx'])
            event_count += 1
        last_row = row

    event_first = np.array(event_first)
    start_order = np.array(start_order).astype(int)
    event_order = np.array(event_order).astype(int)
    start_cum_count = np.array(start_cum_count)
    event_cum_count = np.array(event_cum_count)
    event_idx = np.array(event_idx)

    start_cum_count_cp = start_cum_count.copy()
    start_cum_count[start_order] = start_cum_count_cp
    start_cum_count # original ordering

    event_cum_count_cp = event_cum_count.copy()
    event_cum_count[event_order] = event_cum_count_cp
    event_cum_count # original ordering

    final = pd.DataFrame({'event':event[event_order],
                          'start':start[event_order],
                          'status':status[event_order],
                          'event_first':event_first,
                          'start_idx':start_cum_count[event_order].astype(int), # this is how many (sorted) event times
                                                            # this individual is not in the risk set 
                          'event_idx':event_cum_count[event_order].astype(int)
                          
                          })

    if sample_weight is None:
        sample_weight = np.ones(final.shape[0])
    
    final['sample_weight'] = sample_weight[event_order]

    # we also need the largest index in a cluster of tied failure times
    
    event_last = []
    last_event = n-1
    for i, f in enumerate(final['event_first'][::-1]):
        event_last.append(last_event)
        
        if f - (n - 1 - i) == 0:
            last_event = f - 1        

    final['event_last'] = event_last[::-1]
    
    loglik_sat = 0
    df = final[['event_first', 'status', 'sample_weight']]
    for v, df in final[['event_first', 'status', 'sample_weight']].groupby('event_first'):
        if df['status'].sum() > 0:
            W = (df['sample_weight'] * df['status']).sum()
            loglik_sat -= W * np.log(W)

    

    return final, event_order, start_order, loglik_sat

final, event_order, start_order, loglik_sat = sort_start_event(sorted_df)
final
# -

final['event_first'] - np.arange(n)

final.iloc[:20]

# Let's look at row `5` above (really the 6-th row of course). The value of `5` for `start_idx` means that this individual's `start` time was
# such that they were not in the risk set for the first 5 event times. Indeed, the `start` time is `6` and the 5-th largest event time is `6`, the 6-th largest (which is this row) is `7`. So there are 5 event times less than or equal to 6.
#

final[lambda df: df['event'] <= final.loc[5, 'start']]

# Let's look at row `16`. The `start_idx` there is `3`. 

final[lambda df: df['event'] <= final.loc[16, 'start']]

# ## `event_idx`
#
# This is how many start times (not counting ties) occured strictly before this event time. 
# Let's look at row 8 now. The value of `22` for `event_idx` indicates there are 22 start times less than 9. Alternatively, there are 3 start times greater than or equal to 9. These must be removed from the risk set. This can be done by subtracting `S[22]` (where `S` is the reversed cumsum with respect to start time of $w_j e^{\eta_j}$).

(final['start'] < final.loc[8,'event']).sum()

# ## `event_first`
#
# This vector indexes the events in such a way that, with tied failure times, the index is the *first* time in this sort that a failure occurs.
#
# Let's look at row `12` for `event_first`. Row `12` is a failure time with a time of `10`. There are three failures at this time, indices `[11,12,13]`. These three rows
# have `event_first=11`. The rows of `event_first` that do not count as failure times will not be important for evaluation (at least for Breslow's method). 
#
# # Evaluation
#
# Let's put in some weights and some linear predictor.

weight = rng.uniform(1, 2, size=n)
eta = linear_predictor = rng.standard_normal(size=n)
eta -= eta.mean()

# # Sort by event time

weight_e = weight[event_order]
eta_e = eta[event_order]
final['exp_eta_w'] = weight_e * np.exp(eta_e)
final['weight'] = weight_e

# +
weight_s = weight[start_order]
eta_s = eta[start_order]

start_cumsum = np.cumsum((weight_s * np.exp(eta_s))[::-1])[::-1]
event_cumsum = np.cumsum((weight_e * np.exp(eta_e))[::-1])[::-1]
final['event_cumsum'] = event_cumsum
start_cumsum[22]
# -

exp_eta_w = np.exp(eta) * weight
np.sum(exp_eta_w[start >= 9])

# The contribution for row 8 is therefore

final['event_cumsum'][final['event_first'][8]] - start_cumsum[final['event_idx'][8]]

np.sum(exp_eta_w[(event >= final['event'][8]) * (start < final['event'][8])])

# ## Check
#
#

by_hand, using_indices = [], []
for _r in final.index:
    row = final.loc[_r]
    v = int(row['event_idx'])
    
    if row['status'] == 1: # only compute at failure times
        if v < n:
            using_indices.append(final['event_cumsum'][row['event_first']] - start_cumsum[v])
        else:
            using_indices.append(final['event_cumsum'][row['event_first']])
        by_hand.append(np.sum(exp_eta_w[(event >= row['event']) * (start < row['event'])]))

np.allclose(np.array(by_hand), np.array(using_indices))

# +
efron = True
def log_like(eta, 
             sample_weight, 
             final_df, 
             event_order, 
             start_order,
             efron=efron):
    eta = eta - eta.mean()
    eta_event = eta[event_order]
    w_event = sample_weight[event_order]

    exp_eta_w_event = w_event * np.exp(eta_event)
    exp_eta_w_start = np.hstack([(sample_weight * np.exp(eta))[start_order], 0])

    event_cumsum = np.hstack([np.cumsum(exp_eta_w_event[::-1])[::-1], 0])
    start_cumsum = np.cumsum(exp_eta_w_start[::-1])[::-1]

    diffs = event_cumsum[final_df['event_first']] - start_cumsum[final_df['event_idx']]
    if efron == True:
        n = final_df.shape[0]
        num = (event_cumsum[np.asarray(final_df['event_first'])[:-1]] - 
               event_cumsum[np.asarray(final_df['event_last'])[:-1]+1])
        den = np.asarray(final_df['event_last'])[:-1] + 1 - np.asarray(final_df['event_first'])[:-1]
        efron_means = num / den
        scaling = np.arange(n-1) - np.asarray(final_df['event_first'])[:-1]
        print(scaling)
        print(den)
        diffs[:-1] -= efron_means * scaling
        print(efron_means)
    log_terms = np.log(np.array(diffs)) * w_event * (final_df['status'] == 1)
    loglik = (w_event * eta_event * (final_df['status']==1)).sum() - np.sum(log_terms)

    recip = np.asarray(final_df['status'] * w_event / diffs)
    recip_cumsum = np.hstack([0, np.cumsum(recip)])
    
    G_term = recip_cumsum[final_df['event_last']+1] - recip_cumsum[final_df['start_idx']]
    
    grad = np.asarray(w_event * (final_df['status'] == 1) - exp_eta_w_event * G_term)
    grad_cp = grad.copy()
    grad[event_order] = grad_cp

    # now the Hessian

    recip2 = np.asarray(final_df['status'] * w_event / diffs**2)
    recip2_cumsum = np.hstack([0, np.cumsum(recip2)])

    G2_term = recip2_cumsum[final_df['event_last']+1] - recip2_cumsum[final_df['start_idx']]
    
    diag_hess = exp_eta_w_event**2 * G2_term - exp_eta_w_event * G_term
    diag_hess_cp = diag_hess.copy()
    diag_hess[event_order] = diag_hess_cp
    
    return loglik, grad, diag_hess

loglik, G, H = log_like(eta, weight, final, event_order, start_order)
# -


final

# +
import jax.numpy as jnp
from jax import grad
def log_like_jax(eta, sample_weight, final_df, event_order, start_order, efron=efron):
    eta = eta - jnp.mean(eta)
    eta_event = eta[event_order]
    w_event = sample_weight[event_order]

    exp_eta_w_event = w_event * jnp.exp(eta_event)
    exp_eta_w_start = jnp.hstack([(sample_weight * jnp.exp(eta))[start_order], 0])

    event_cumsum = jnp.cumsum(exp_eta_w_event[::-1])[::-1]
    start_cumsum = jnp.cumsum(exp_eta_w_start[::-1])[::-1]

    diffs = (event_cumsum[np.asarray(final_df['event_first'])] - 
             start_cumsum[np.asarray(final_df['event_idx'])])
    if efron == True:
        num = (event_cumsum[np.asarray(final_df['event_first'])[:-1]] - 
               event_cumsum[np.asarray(final_df['event_last'])[:-1]+1])
        den = np.asarray(final_df['event_last'])[:-1] + 1 - np.asarray(final_df['event_first'])[:-1]
        efron_means = num / den
        
        diffs = diffs.at[:-1].add(- efron_means)
    log_terms = jnp.log(jnp.array(diffs)) * w_event * np.asarray(final_df['status'] == 1)
    
    #log_terms = (jnp.log(jnp.array(diffs)) * w_event)[np.asarray(final_df['status']) == 1]

    loglik = (w_event * eta_event)[np.asarray(final_df['status'])==1].sum() - np.sum(log_terms)

    return loglik
    
def logL(eta):
    return log_like_jax(eta, weight, final, event_order, start_order)
logL(eta), loglik
# -

grad_logL = grad(logL)
G_jax = grad_logL(eta)

G_jax[:10] - G[:10]

import rpy2
# %load_ext rpy2.ipython
# %R -i start,event,status,weight,eta

# + magic_args="-o G_R" language="R"
# library(survival)
# library(glmnet)
# y_s = Surv(start, event, status)
# G_R = glmnet:::coxgrad3(eta, y_s, as.numeric(weight), std.weights=FALSE, diag.hessian=TRUE)
# -

G_jax

G

np.linalg.norm(G - G_jax) / np.linalg.norm(G), np.linalg.norm(G - G_R) / np.linalg.norm(G)

# # Run with no ties

ties = False
start = rng.integers(0, 10, size=n) 
event = start + rng.integers(0, 10, size=n) + ties + (1 - ties) * rng.standard_exponential(n) * 0.01
status = rng.choice([0,1], size=n, replace=True)

stacked_df = pd.DataFrame({'time':np.hstack([start, event]),
                           'status':np.hstack([np.zeros_like(start), status]),
                           'start?':np.hstack([np.ones(n), np.zeros(n)]),
                           'idx':np.hstack([np.arange(n), np.arange(n)])})
sorted_df = stacked_df.sort_values(by=['time', 'status', 'start?'], ascending=[True,False,True])
sorted_df[:20]
final, event_order, start_order, loglik_sat = sort_start_event(sorted_df, weight)
loglik, G, H = log_like(eta, weight, final, event_order, start_order)
2 * (loglik_sat - loglik)
final

# + magic_args="-i start,event,status,weight,eta -o G_R,H_R" language="R"
# y_s = Surv(start, event, status)
# G_R = glmnet:::coxgrad3(eta, y_s, as.numeric(weight), std.weights=FALSE, diag.hessian=TRUE)
# H_R = attr(G_R, "diag_hessian")
# G_R
# -

np.linalg.norm(G - G_R) / np.linalg.norm(G)

np.linalg.norm(H_R - H) / np.linalg.norm(H)

G_jax = grad_logL(eta)
np.linalg.norm(G - G_jax) / np.linalg.norm(G), np.linalg.norm(G-G_R) / np.linalg.norm(G)

final

# ## Right censored case
#

start = -np.inf * np.ones(n)
stacked_df = pd.DataFrame({'time':np.hstack([start, event]),
                           'status':np.hstack([np.zeros_like(start), status]),
                           'start?':np.hstack([np.ones(n), np.zeros(n)]),
                           'idx':np.hstack([np.arange(n), np.arange(n)])})
sorted_df = stacked_df.sort_values(by=['time', 'status', 'start?'], ascending=[True,False,True])
sorted_df[:20]
final, event_order, start_order, loglik_sat = sort_start_event(sorted_df, weight)
loglik, G, H = log_like(eta, weight, final, event_order, start_order)
2 * (loglik_sat - loglik)

# + magic_args="-i start,event,status,weight,eta -o G_R,H_R" language="R"
# y_s = Surv(event, status)
# print(glmnet:::coxnet.deviance2(pred=as.numeric(eta), y=y_s, weights=weight, std.weights=FALSE))
# G_R = glmnet:::coxgrad2(eta, y_s, as.numeric(weight), std.weights=FALSE, diag.hessian=TRUE)
# H_R = attr(G_R, "diag_hessian")
# -

np.linalg.norm(G - G_R) / np.linalg.norm(G)

np.linalg.norm(H_R - H) / np.linalg.norm(H)

from glmnet import coxdev
rc = coxdev.CoxRightCensored(event, status, sample_weight=weight)
r = rc(eta, compute_gradient=True, compute_diag_hessian=True)
r.deviance
np.linalg.norm(r.grad - 2 *G) / np.linalg.norm(r.grad)

np.linalg.norm(r.diag_hessian - 2 * H) / np.linalg.norm(r.diag_hessian)


