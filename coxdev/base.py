# these should be done in C ideally

import numpy as np

def _forward_cumsum(sequence, output):
    '''
    compute cumsum with a padding of 0 at the beginning
    '''
    output[:] = np.cumsum(np.hstack([0, sequence]))

def _reverse_cumsums(sequence,
                     event_buffer,
                     start_buffer,
                     event_order,
                     start_order,
                     do_event=False,
                     do_start=False):
    """
    Compute reversed cumsums of a sequence
    in start and / or event order with a 0 padded at the end.
    """
    
    # pad by 1 at the end length=n+1 for when last=n-1    

    if do_event:
        seq_event = np.hstack([sequence[event_order], 0])
        event_buffer[:] = np.cumsum(seq_event[::-1])[::-1]

    if do_start:
        seq_start = np.hstack([sequence[start_order], 0])
        start_buffer[:] = np.cumsum(seq_start[::-1])[::-1]  # length=n+1

def _to_native_from_event(arg,
                          event_order,
                          reorder_buffer):
    """
    reorder an event-ordered vector into native order,
    uses forward_scratch_buffer to make a temporary copy
    """
    reorder_buffer[:] = arg
    arg[event_order] = reorder_buffer

def _to_event_from_native(arg,
                          event_order,
                          reorder_buffer):
    """
    reorder an event-ordered vector into native order,
    uses forward_scratch_buffer to make a temporary copy
    """
    reorder_buffer[:] = arg[event_order]

def _forward_prework(status,
                     w_avg,
                     scaling,
                     risk_sums,
                     i,
                     j,
                     moment_buffer,
                     use_w_avg=True,
                     arg=None):
    """
    we need some sort of cumsums of scaling**i / risk_sums**j weighted by w_avg (within status==1)

    this function fills in appropriate buffer
    """
    if use_w_avg:
        moment_buffer[:] = status * w_avg * (scaling**i) / (risk_sums**j)
    else:
        moment_buffer[:] = status * (scaling**i) / (risk_sums**j)
    if arg is not None:
        moment_buffer[:] *= arg

def _compute_sat_loglik(_first,
                        _last,
                        _weight, # in natural order!!!
                        _event_order,
                        _status,
                        W_status):
    
    _forward_cumsum(_weight[_event_order] * _status, W_status)
    sums = W_status[_last+1] - W_status[_first]
    loglik_sat = 0
    prev_first = -1
    for f, s in zip(_first, sums):
        if s > 0 and f != prev_first:
            loglik_sat -= s * np.log(s)
        prev_first = f

    return loglik_sat

# Core function: compute log-likelihood

def _cox_dev(eta,           # eta is in native order  -- assumes centered (or otherwise normalized for numeric stability)
             sample_weight, # sample_weight is in native order
             exp_w,
             event_order,   
             start_order,
             status,        # everything below in event order
             first,
             last,
             scaling,
             event_map,
             start_map,
             loglik_sat,
             T_1_term,
             T_2_term,
             grad_buffer,
             diag_hessian_buffer,
             diag_part_buffer,
             w_avg_buffer,
             event_reorder_buffers,
             risk_sum_buffers,
             forward_cumsum_buffers,
             forward_scratch_buffer,
             reverse_cumsum_buffers,
             have_start_times=True,
             efron=False):

    n = eta.shape[0]
    
    _to_event_from_native(eta, event_order, event_reorder_buffers[0])
    eta_event = event_reorder_buffers[0]

    _to_event_from_native(sample_weight, event_order, event_reorder_buffers[1])
    w_event = event_reorder_buffers[1]

    _to_event_from_native(exp_w, event_order, event_reorder_buffers[2])
    exp_eta_w_event = event_reorder_buffers[2]
   
    if have_start_times:
        _sum_over_risk_set(exp_w, # native order
                           event_order,
                           start_order,
                           first,
                           last,
                           event_map,
                           scaling,
                           efron,
                           risk_sum_buffers[0],
                           reverse_cumsum_buffers[:2])
    else:
        _sum_over_risk_set(exp_w, # native order
                           event_order,
                           start_order,
                           first,
                           last,
                           None,
                           scaling,
                           efron,
                           risk_sum_buffers[0],
                           reverse_cumsum_buffers[:2])

    event_cumsum = reverse_cumsum_buffers[0]
    start_cumsum = reverse_cumsum_buffers[1]
    risk_sums = risk_sum_buffers[0]
    
    # some ordered terms to complete likelihood
    # calculation

    # w_cumsum is only used here, can write over forward_cumsum_buffers
    # after computing w_avg
    _forward_cumsum(w_event, forward_cumsum_buffers[0])
    w_cumsum = forward_cumsum_buffers[0]
    w_avg_buffer[:] = ((w_cumsum[last + 1] - w_cumsum[first]) /
                       (last + 1 - first))
    w_avg = w_avg_buffer # shorthand

    loglik = ((w_event * eta_event * status).sum() -
              np.sum(np.log(np.array(risk_sums)) * w_avg * status))

    # forward cumsums for gradient and Hessian
    
    # length of cumsums is n+1
    # 0 is prepended for first(k)-1, start(k)-1 lookups
    # a 1 is added to all indices

    _forward_prework(status, w_avg, scaling, risk_sums, 0, 1, forward_scratch_buffer)
    A_01 = forward_scratch_buffer
    _forward_cumsum(A_01, forward_cumsum_buffers[0]) # length=n+1 
    C_01 = forward_cumsum_buffers[0]

    _forward_prework(status, w_avg, scaling, risk_sums, 0, 2, forward_scratch_buffer)
    A_02 = forward_scratch_buffer
    _forward_cumsum(A_02, forward_cumsum_buffers[1]) # length=n+1
    C_02 = forward_cumsum_buffers[1]
    
    if not efron:
        if have_start_times:

            # +1 for start_map? depends on how  
            # a tie between a start time and an event time
            # if that means the start individual is excluded
            # we should add +1, otherwise there should be
            # no +1 in the [start_map+1] above

            T_1_term[:] = C_01[last+1] - C_01[start_map]
            T_2_term[:] = C_02[last+1] - C_02[start_map]

        else:
            T_1_term[:] = C_01[last+1]
            T_2_term[:] = C_02[last+1]
    else:
        # compute the other necessary cumsums
        
        _forward_prework(status, w_avg, scaling, risk_sums, 1, 1, forward_scratch_buffer)
        A_11 = forward_scratch_buffer
        _forward_cumsum(A_11, forward_cumsum_buffers[2]) # length=n+1
        C_11 = forward_cumsum_buffers[2]

        _forward_prework(status, w_avg, scaling, risk_sums, 2, 1, forward_scratch_buffer)
        A_21 = forward_scratch_buffer
        _forward_cumsum(A_21, forward_cumsum_buffers[3]) # length=n+1
        C_21 = forward_cumsum_buffers[3]

        _forward_prework(status, w_avg, scaling, risk_sums, 2, 2, forward_scratch_buffer)
        A_22 = forward_scratch_buffer
        _forward_cumsum(A_22, forward_cumsum_buffers[4]) # length=n+1
        C_22 = forward_cumsum_buffers[4]

        T_1_term[:] = (C_01[last+1] - 
                       (C_11[last+1] - C_11[first]))
        T_2_term[:] = ((C_22[last+1] - C_22[first]) 
                       - 2 * (C_21[last+1] - C_21[first]) + 
                       C_02[last+1])

        if have_start_times:
            T_1_term -= C_01[start_map]
            T_2_term -= C_02[first]
    
    # could do multiply by exp_w after reorder...
    # save a reorder of w * exp(eta)

    diag_part_buffer[:] = exp_eta_w_event * T_1_term
    grad_buffer[:] = w_event * status - diag_part_buffer
    grad_buffer *= -2
    # now the diagonal of the Hessian

    diag_hessian_buffer[:] = exp_eta_w_event**2 * T_2_term - diag_part_buffer
    diag_hessian_buffer *= -2
    
    _to_native_from_event(grad_buffer, event_order, forward_scratch_buffer)
    _to_native_from_event(diag_hessian_buffer, event_order, forward_scratch_buffer)
    _to_native_from_event(diag_part_buffer, event_order, forward_scratch_buffer)
    
    deviance = 2 * (loglik_sat - loglik)

    return deviance

def _sum_over_events(event_order,
                     start_order,
                     first,
                     last,
                     start_map,
                     scaling,
                     status,
                     efron,
                     forward_cumsum_buffers,
                     forward_scratch_buffer,
                     value_buffer):
    '''
    compute sum_i (d_i Z_i ((1_{t_k>=t_i} - 1_{s_k>=t_i}) - sigma_i (1_{i <= last(k)} - 1_{i <= first(k)-1})
    '''
        
    have_start_times = start_map is not None

    n = status.shape[0]

    _forward_cumsum(forward_scratch_buffer, forward_cumsum_buffers[0]) # length=n+1
    C_arg = forward_cumsum_buffers[0]
    
    value_buffer[:] = C_arg[last+1]
    if have_start_times:
        value_buffer -= C_arg[start_map]

    if efron:
        forward_scratch_buffer[:] *= scaling
        _forward_cumsum(forward_scratch_buffer, forward_cumsum_buffers[1]) # length=n+1
        C_arg_scale = forward_cumsum_buffers[1]
        value_buffer -= C_arg_scale[last+1] - C_arg_scale[first]

def _sum_over_risk_set(arg,
                       event_order,
                       start_order,
                       first,
                       last,
                       event_map,
                       scaling,
                       efron,
                       risk_sum_buffer,
                       reverse_cumsum_buffers):

    '''
    arg is in native order
    returns a sum in event order
    '''

    have_start_times = event_map is not None

    event_cumsum = reverse_cumsum_buffers[0]
    start_cumsum = reverse_cumsum_buffers[1]
    
    _reverse_cumsums(arg,
                     event_cumsum,
                     start_cumsum,
                     event_order,
                     start_order,
                     do_event=True,
                     do_start=have_start_times)

    if have_start_times:
        risk_sum_buffer[:] = event_cumsum[first] - start_cumsum[event_map]
    else:
        risk_sum_buffer[:] = event_cumsum[first]
        
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
        risk_sum_buffer[:] -= delta * scaling

def _hessian_matvec(arg,           # arg is in native order
                    eta,           # eta is in native order 
                    sample_weight, # sample_weight is in native order
                    risk_sums,
                    diag_part,
                    w_avg,
                    exp_w,
                    event_cumsum,
                    start_cumsum,
                    event_order,   
                    start_order,
                    status,        # everything below in event order
                    first,
                    last,
                    scaling,
                    event_map,
                    start_map,
                    risk_sum_buffers,
                    forward_cumsum_buffers,
                    forward_scratch_buffer,
                    reverse_cumsum_buffers,
                    hess_matvec_buffer,
                    have_start_times=True,
                    efron=False):                    

    #breakpoint()
    if have_start_times:
        # now in event_order
        _sum_over_risk_set(exp_w * arg, # in native order
                           event_order,
                           start_order,
                           first,
                           last,
                           event_map,
                           scaling,
                           efron,
                           risk_sum_buffers[1], 
                           reverse_cumsum_buffers[2:4])
    else:
        _sum_over_risk_set(exp_w * arg, # in native order
                           event_order,
                           start_order,
                           first,
                           last,
                           None,
                           scaling,
                           efron,
                           risk_sum_buffers[1], 
                           reverse_cumsum_buffers[2:4])

    risk_sums_arg = risk_sum_buffers[1]

    # E_arg = risk_sums_arg / risk_sums -- expecations under the probabilistic interpretation
    # forward_scratch_buffer[:] = status * w_avg * E_arg / risk_sums

    # one less step to compute from above representation
    forward_scratch_buffer[:] = status * w_avg * risk_sums_arg / risk_sums**2
    print(f'forward_scratch_buffer {forward_scratch_buffer}')
    
    if have_start_times:
        _sum_over_events(event_order,
                         start_order,
                         first,
                         last,
                         start_map,
                         scaling,
                         status,
                         efron,
                         forward_cumsum_buffers,
                         forward_scratch_buffer,
                         hess_matvec_buffer)
    else:
        _sum_over_events(event_order,
                         start_order,
                         first,
                         last,
                         None,
                         scaling,
                         status,
                         efron,
                         forward_cumsum_buffers,
                         forward_scratch_buffer,
                         hess_matvec_buffer)
    print(f'hess_matvec_buffer {hess_matvec_buffer}')        
    
    _to_native_from_event(hess_matvec_buffer, event_order, forward_scratch_buffer)

    hess_matvec_buffer *= exp_w 
    hess_matvec_buffer -= diag_part * arg
    print(f'hess_matvec_buffer {hess_matvec_buffer}')
    

