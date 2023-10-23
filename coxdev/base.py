# these should be done in C ideally

import numpy as np

def _compute_sat_loglik(_first,
                        _last,
                        _weight, # in natural order!!!
                        _event_order,
                        _status):
    
    W_status = np.cumsum(np.hstack([0, _weight[_event_order] * _status]))
    sums = W_status[_last+1] - W_status[_first]
    loglik_sat = 0
    prev_first = -1
    for f, s in zip(_first, sums):
        if s > 0 and f != prev_first:
            loglik_sat -= s * np.log(s)
        prev_first = f

    return loglik_sat

def _reversed_cumsums(sequence,
                      event_order=None,
                      start_order=None):
    """
    Compute reversed cumsums of a sequence
    in start and / or event order with a 0 padded at the end.
    """
    
    # pad by 1 at the end length=n+1 for when last=n-1    

    if event_order is not None:
        seq_event = np.hstack([sequence[event_order], 0])
        event_cumsum = np.cumsum(seq_event[::-1])[::-1]
    else:
        event_cumsum = None

    if start_order is not None:
        seq_start = np.hstack([sequence[start_order], 0])
        start_cumsum = np.cumsum(seq_start[::-1])[::-1]  # length=n+1
    else:
        start_cumsum = None

    return event_cumsum, start_cumsum

# Core function

def _cox_dev(eta,           # eta is in native order 
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
             first_start,
             loglik_sat,
             have_start_times=True,
             efron=False):

    eta = eta - eta.mean()
    
    # compute the event ordered reversed cumsum
    exp_w = np.exp(eta) * sample_weight
    
    if have_start_times:
        risk_sums = _sum_over_risk_set(exp_w,
                                       event_order,
                                       start_order,
                                       first,
                                       last,
                                       event_map,
                                       scaling,
                                       efron)
    else:
        risk_sums = _sum_over_risk_set(exp_w,
                                       event_order,
                                       start_order,
                                       first,
                                       last,
                                       None,
                                       scaling,
                                       efron)

    # some ordered terms to complete likelihood
    # calculation

    eta_event = eta[event_order]
    w_event = sample_weight[event_order]
    w_cumsum = np.cumsum(np.hstack([0, sample_weight[event_order]]))
    w_avg = ((w_cumsum[last + 1] - w_cumsum[first]) /
             (last + 1 - first))

    exp_eta_w_event = exp_w[event_order]

    log_terms = np.log(np.array(risk_sums)) * w_avg * status
    loglik = (w_event * eta_event * status).sum() - np.sum(log_terms)

    # forward cumsums for gradient and Hessian
    
    # length of cumsums is n+1
    # 0 is prepended for first(k)-1, start(k)-1 lookups
    # a 1 is added to all indices

    A_10 = status * w_avg / risk_sums
    C_10 = np.hstack([0, np.cumsum(A_10)]) 
    
    A_20 = status * w_avg / risk_sums**2
    C_20 = np.hstack([0, np.cumsum(A_20)]) # length=n+1

    # if there are no ties, scaling should be identically 0
    # don't bother with cumsums below 

    use_first_start = True # JT: don't think this is strictly needed
                           # but haven't found a counterexample

    if not efron:
        if have_start_times:

            # +1 for start_map? depends on how  
            # a tie between a start time and an event time
            # if that means the start individual is excluded
            # we should add +1, otherwise there should be
            # no +1 in the [start_map+1] above

            if use_first_start:
                T_1_term = C_10[last+1] - C_10[first_start]
                T_2_term = C_20[last+1] - C_20[first_start]
            else:
                T_1_term = C_10[last+1] - C_10[start_map]
                T_2_term = C_20[last+1] - C_20[start_map]
        else:
            T_1_term = C_10[last+1]
            T_2_term = C_20[last+1]
    else:
        # compute the other necessary cumsums
        
        A_11 = status * w_avg * scaling / risk_sums
        C_11 = np.hstack([0, np.cumsum(A_11)]) # length=n+1

        A_21 = status * w_avg * scaling**2 / risk_sums
        C_21 = np.hstack([0, np.cumsum(A_21)]) # length=n+1

        A_22 = status * w_avg * scaling**2 / risk_sums**2
        C_22 = np.hstack([0, np.cumsum(A_22)]) # length=n+1

        T_1_term = (C_10[last+1] - 
                    (C_11[last+1] - C_11[first]))
        T_2_term = ((C_22[last+1] - C_22[first]) 
                    - 2 * (C_21[last+1] - C_21[first]) + 
                    C_20[last+1])

        if have_start_times:
            if use_first_start:
                T_1_term -= C_10[first_start]
            else:
                T_1_term -= C_10[start_map]
            T_2_term -= C_20[first]
    
    # could do multiply by exp_w after reorder...
    # save a reorder of w * exp(eta)
    diag_part = exp_eta_w_event * T_1_term
    grad = w_event * status - diag_part
    grad_cp = grad.copy()
    grad[event_order] = grad_cp

    # now the diagonal of the Hessian

    diag_hess = exp_eta_w_event**2 * T_2_term - diag_part
    diag_hess_cp = diag_hess.copy()
    diag_hess[event_order] = diag_hess_cp

    diag_part_cp = diag_part.copy()
    diag_part[event_order] = diag_part_cp
    
    deviance = 2 * (loglik_sat - loglik)

    return (loglik_sat,
            deviance,
            -2 * grad,
            -2 * diag_hess,
            risk_sums,
            diag_part,
            w_avg,
            exp_w)

def _sum_over_events(arg,
                     event_order,
                     start_order,
                     first,
                     last,
                     start_map,
                     scaling,
                     status,
                     efron):
    '''
    compute sum_i (d_i Z_i ((1_{t_k>=t_i} - 1_{s_k>=t_i}) - sigma_i (1_{i <= last(k)} - 1_{i <= first(k)-1})
    '''
        
    have_start_times = start_map is not None

    C_arg = np.hstack([0, np.cumsum(arg * status)])
    value = C_arg[last+1]
    if have_start_times:
        value -= C_arg[start_map]

    if efron:
        C_arg_scale = np.hstack([0, np.cumsum(arg * scaling)])
        value -= C_arg_scale[last+1] - C_arg_scale[first]
    return value

def _sum_over_risk_set(arg,
                       event_order,
                       start_order,
                       first,
                       last,
                       event_map,
                       scaling,
                       efron):

    '''
    arg is in native order
    returns a sum in event order
    '''

    have_start_times = event_map is not None

    if have_start_times:
        (event_cumsum,
         start_cumsum) = _reversed_cumsums(arg,
                                           event_order=event_order,
                                           start_order=start_order)
    else:
        (event_cumsum,
         start_cumsum) = _reversed_cumsums(arg,
                                           event_order,
                                           start_order=None)
        
    if have_start_times:
        _sum = event_cumsum[first] - start_cumsum[event_map]
    else:
        _sum = event_cumsum[first]
        
    # compute the Efron correction, adjusting risk_sum if necessary
    
    if efron:
        # for K events,
        # this results in risk sums event_cumsum[first] to
        # event_cumsum[first] -
        # (K-1)/K [event_cumsum[last+1] - event_cumsum[first]
        # or event_cumsum[last+1] + 1/K [event_cumsum[first] - event_cumsum[last+1]]
        # to event[cumsum_first]
        delta = (event_cumsum[first] - 
                 event_cumsum[last+1])
        _sum -= delta * scaling

    # returned in event order!
    
    return _sum

def _hessian_matvec(arg,           # arg is in native order
                    eta,           # eta is in native order 
                    sample_weight, # sample_weight is in native order
                    risk_sums,
                    diag_part,
                    w_avg,
                    exp_w,
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
                    first_start,
                    have_start_times=True,
                    efron=False):                    

    if have_start_times:
        # now in event_order
        risk_sums_arg = _sum_over_risk_set(exp_w * arg, # in native order
                                           event_order,
                                           start_order,
                                           first,
                                           last,
                                           event_map,
                                           scaling,
                                           efron)
    else:
        risk_sums_arg = _sum_over_risk_set(exp_w * arg, # in native order
                                           event_order,
                                           start_order,
                                           first,
                                           last,
                                           None,
                                           scaling,
                                           efron)

    E_arg = risk_sums_arg / risk_sums
    cumsum_arg = w_avg * E_arg / risk_sums # will be multiplied
                                           # by status in _sum_over_events
    
    if have_start_times:
        value = _sum_over_events(cumsum_arg,
                                 event_order,
                                 start_order,
                                 first,
                                 last,
                                 start_map,
                                 scaling,
                                 status,
                                 efron)
    else:
        value = _sum_over_events(cumsum_arg,
                                 event_order,
                                 start_order,
                                 first,
                                 last,
                                 None,
                                 scaling,
                                 status,
                                 efron)
        
    hess_matvec = np.zeros_like(value)
    hess_matvec[event_order] = value

    hess_matvec *= exp_w 
    hess_matvec -= diag_part * arg
    return hess_matvec


