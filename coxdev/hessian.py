def _hessian(eta,           # eta is in native order 
             sample_weight, # sample_weight is in native order
             right_vector,
             risk_sums,
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
             efron=False,
             asarray=True):

    # be sure they're arrays so that no weird pandas indexing is used

    if efron:
        raise NotImplementedError

    if asarray:
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
    
    # order right_vector

    # 0 is appended for similar reason to calculation of `risk_sums`
    right_vector = np.hstack([right_vector[event_order], 0])
    # reversed cumsum of right vector
    right_vector_rc = np.cumsum(right_vector[::-1])[::-1]

    if risk_sums is None:

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
        
    # XXXX check indices
    # XXXXXX needs multiply by w_id_i
    after_1st_cumsum = (right_vector_rc[first] - right_vector_rc[event_map]) / risk_sums**2
    if efron:
        after_1st_cumsum -= scaling * (right_vector_rc[first] - right_vector_rc[last])


    # 0 prepended for similar reason to computing diagonal of hessian
    cumsum_2nd_0 = np.cumsum(np.hstack([0, after_1st_cumsum]))
    cumsum_2nd_1 = np.cumsum(np.hstack([0, after_1st_cumsum * scaling]))

    matvec = cumsum_2nd_0[last+1] - cumsum_2nd_0[start_map]
    if efron:
        matvec -= cumsum_2nd_1[last_1] - cumsum_2nd_1[first]

    return matvec
