#ifdef DEBUG
#include <iostream>
#endif

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

// Map python buffers list element into an Eigen double vector
// SRC_LIST = python list, OFFSET = index offset (e.g. 0 or 1),
// DEST will be the ref downstream, TMP should be **unique** throwaway name with each invocation
#define MAP_PYLIST_BUF(SRC_LIST, OFFSET, DEST, TMP)				\
  py::array_t<double> TMP = SRC_LIST[OFFSET].cast<py::array_t<double>>();    \
  Eigen::Map<Eigen::VectorXd> DEST(TMP.mutable_data(), TMP.size());

namespace py = pybind11;

//
// Since we want this to be usable both in R and python, I will use int for indexing rather than
// Eigen::Index. Later I will use a #define to emit appropriate code
// Also using doubles for status, which is really only 0, 1
//

// Compute cumsum with a padding of 0 at the beginning
// @param sequence input sequence [ro]
// @param output output sequence  [w]
void forward_cumsum(const Eigen::Ref<Eigen::VectorXd> sequence,
		    Eigen::Ref<Eigen::VectorXd> output)
{
  if (sequence.size() + 1 != output.size()) {
    throw std::runtime_error("forward_cumsum: output size must be one longer than input's.");
  }
      
  double sum = 0.0;
  output(0) = sum;
  for (int i = 1; i < output.size(); ++i) {
    sum = sum + sequence(i - 1);
    output(i) = sum;
  }
}

// Compute reversed cumsums of a sequence
// in start and / or event order with a 0 padded at the end.
// pad by 1 at the end length=n+1 for when last=n-1    
// @param sequence input sequence [ro]
// @param event_buffer [w]
// @param start_buffer [w]
// @param event_order [ro]
// @param start_order [ro]
// @param do_event a flag
// @param do_start a flag
void reverse_cumsums(const Eigen::Ref<Eigen::VectorXd> sequence,
                     Eigen::Ref<Eigen::VectorXd> event_buffer,
                     Eigen::Ref<Eigen::VectorXd> start_buffer,
                     const Eigen::Ref<Eigen::VectorXi> event_order,
                     const Eigen::Ref<Eigen::VectorXi> start_order,
		     bool do_event = false,
		     bool do_start = false)
{
  double sum = 0.0;
  
  int n = sequence.size(); // should be size_t
  if (do_event) {
    if (sequence.size() + 1 != event_buffer.size()) {
      throw std::runtime_error("reverse_cumsums: event_buffer size must be one more than input's.");
    }
    event_buffer(n) = sum;
    for (int i = n - 1; i >= 0;  --i) {
      sum = sum + sequence(event_order(i));
      event_buffer(i) = sum;
    }
  }

  if (do_start) {
    if (sequence.size() + 1 != start_buffer.size()) {
      throw std::runtime_error("reverse_cumsums: event_buffer size must be one more than input's.");
    }
    sum = 0.0;
    start_buffer(n) = sum;
    for (int i = n - 1; i >= 0;  --i) {
      sum = sum + sequence(start_order(i));
      start_buffer(i) = sum;
    }
  }
}


// reorder an event-ordered vector into native order,
// uses forward_scratch_buffer to make a temporary copy
// @param arg
// @param event_order 
// @param reorder_buffer 
void to_native_from_event(Eigen::Ref<Eigen::VectorXd> arg,
			  const Eigen::Ref<Eigen::VectorXi> event_order,
			  Eigen::Ref<Eigen::VectorXd> reorder_buffer)
{
  reorder_buffer = arg;
  for (int i = 0; i < event_order.size(); ++i) {
    arg(event_order(i)) = reorder_buffer(i);
  }
}

// reorder an event-ordered vector into native order,
// uses forward_scratch_buffer to make a temporary copy

void to_event_from_native(const Eigen::Ref<Eigen::VectorXd> arg,
                          const Eigen::Ref<Eigen::VectorXi> event_order,
                          Eigen::Ref<Eigen::VectorXd> reorder_buffer)
{
  for (int i = 0; i < event_order.size(); ++i) {
    reorder_buffer(i) = arg(event_order(i));
  }
}

// We need some sort of cumsums of scaling**i / risk_sums**j weighted by w_avg (within status==1)
// this function fills in appropriate buffer
// The arg = None is checked by a vector having size 0!
void forward_prework(const Eigen::Ref<Eigen::VectorXi> status,
                     const Eigen::Ref<Eigen::VectorXd> w_avg,
                     const Eigen::Ref<Eigen::VectorXd> scaling,
                     const Eigen::Ref<Eigen::VectorXd> risk_sums,
                     int i,
                     int j,
                     Eigen::Ref<Eigen::VectorXd> moment_buffer,
		     const Eigen::Ref<Eigen::VectorXd> arg,		     
                     bool use_w_avg = true)
{
  // No checks on size compatibility yet.
  if (use_w_avg) {
    moment_buffer = status.cast<double>().array() * w_avg.array() * scaling.array().pow(i) / risk_sums.array().pow(j);
  } else {
    moment_buffer = status.cast<double>().array() * scaling.array().pow(i) / risk_sums.array().pow(j);    
  }
  if (arg.size() > 0) {
    moment_buffer = moment_buffer.array() * arg.array();
  }
}

double compute_sat_loglik(const Eigen::Ref<Eigen::VectorXi> first,
			  const Eigen::Ref<Eigen::VectorXi> last,
			  const Eigen::Ref<Eigen::VectorXd> weight, // in natural order!!!
			  const Eigen::Ref<Eigen::VectorXi> event_order,
			  const Eigen::Ref<Eigen::VectorXi> status,
			  Eigen::Ref<Eigen::VectorXd> W_status)
{
  
  Eigen::VectorXd weight_event_order_times_status(event_order.size());
  for (int i = 0; i < event_order.size(); ++i) {
    weight_event_order_times_status(i) = weight(event_order(i)) * status(i);
  }
  forward_cumsum(weight_event_order_times_status, W_status);

  Eigen::VectorXd sums(last.size());
  for (int i = 0; i < last.size(); ++i) {
    sums(i) = W_status(last(i) + 1) - W_status(first(i));
  }
  double loglik_sat = 0.0;
  int prev_first = -1;

  for (int i = 0; i < first.size(); ++i) {
    int f = first(i); double s = sums(i);
    if (s > 0 && f != prev_first) {
      loglik_sat -= s * log(s);
    }
    prev_first = f;
  }
  return(loglik_sat);
}


// compute sum_i (d_i Z_i ((1_{t_k>=t_i} - 1_{s_k>=t_i}) - sigma_i (1_{i <= last(k)} - 1_{i <= first(k)-1})
// Note how MatrixXd storage mode can affect efficiency in Python versus R for example.
void sum_over_events(const Eigen::Ref<Eigen::VectorXi> event_order,
                     const Eigen::Ref<Eigen::VectorXi> start_order,
                     const Eigen::Ref<Eigen::VectorXi> first,
                     const Eigen::Ref<Eigen::VectorXi> last,
                     const Eigen::Ref<Eigen::VectorXi> start_map,
                     const Eigen::Ref<Eigen::VectorXd> scaling,
                     const Eigen::Ref<Eigen::VectorXi> status,
                     bool efron,
                     py::list &forward_cumsum_buffers, // List of numpy arrays (1-d)
		     Eigen::Ref<Eigen::VectorXd> forward_scratch_buffer,
                     Eigen::Ref<Eigen::VectorXd> value_buffer)
{

  bool have_start_times = start_map.size() >  0;

  // Map first element of list into Eigen vector.
  // C_arg = forward_cumsum_buffers[0]!
  
  MAP_PYLIST_BUF(forward_cumsum_buffers, 0, C_arg, tmp1)	


  forward_cumsum(forward_scratch_buffer, C_arg); //length=n+1

  if (have_start_times) {
    for (int i = 0; i < last.size(); ++i) {
      value_buffer(i) = C_arg(last(i) + 1) - C_arg(start_map(i));
    }
  } else {
    for (int i = 0; i < last.size(); ++i) {
      value_buffer(i) = C_arg(last(i) + 1);
    }
  }
  if (efron) {
    forward_scratch_buffer = forward_scratch_buffer.array() * scaling.array();
    // Map second element of list into Eigen vector.
    // C_arg_scale = forward_cumsum_buffers[1]!
    MAP_PYLIST_BUF(forward_cumsum_buffers, 1, C_arg_scale, tmp2)	
    
    forward_cumsum(forward_scratch_buffer, C_arg_scale); // length=n+1
    for (int i = 0; i < last.size(); ++i) {
      value_buffer(i) -= (C_arg_scale(last(i) + 1) - C_arg_scale(first(i)));
    }
  }
}

// arg is in native order
// returns a sum in event order
void sum_over_risk_set(const Eigen::Ref<Eigen::VectorXd> arg,
                       const Eigen::Ref<Eigen::VectorXi> event_order,
                       const Eigen::Ref<Eigen::VectorXi> start_order,
                       const Eigen::Ref<Eigen::VectorXi> first,
                       const Eigen::Ref<Eigen::VectorXi> last,
                       const Eigen::Ref<Eigen::VectorXi> event_map,
                       const Eigen::Ref<Eigen::VectorXd> scaling,
                       bool efron,
                       py::list &risk_sum_buffers,
		       int risk_sum_buffers_offset,
                       py::list &reverse_cumsum_buffers, // List of 1-d numpy arrays
		       int reverse_cumsum_buffers_offset) // starting index into buffer
{

  bool have_start_times = event_map.size() > 0;
  
  // Map first element of list into Eigen vector.
  MAP_PYLIST_BUF(reverse_cumsum_buffers, reverse_cumsum_buffers_offset, event_cumsum, tmp1)	

  // Map second element of list into Eigen vector.
  MAP_PYLIST_BUF(reverse_cumsum_buffers, reverse_cumsum_buffers_offset + 1, start_cumsum, tmp2)	

  reverse_cumsums(arg,
		  event_cumsum,
		  start_cumsum,
		  event_order,
		  start_order,
		  true, // do_event 
		  have_start_times); // do_start
    
  // Map first element of list into Eigen vector.
  MAP_PYLIST_BUF(risk_sum_buffers, risk_sum_buffers_offset, risk_sum_buffer, tmp3)
    
  if (have_start_times) {
    for (int i = 0; i < first.size(); ++i) {
      risk_sum_buffer(i) = event_cumsum(first(i)) - start_cumsum(event_map(i));
    }
  } else {
    for (int i = 0; i < first.size(); ++i) {
      risk_sum_buffer(i) = event_cumsum(first(i));
    }
  }
        
  // compute the Efron correction, adjusting risk_sum if necessary
    
  if (efron) {
    // for K events,
    // this results in risk sums event_cumsum[first] to
    // event_cumsum[first] -
    // (K-1)/K [event_cumsum[last+1] - event_cumsum[first]
    // or event_cumsum[last+1] + 1/K [event_cumsum[first] - event_cumsum[last+1]]
    // to event[cumsum_first]
    for (int i = 0; i < first.size(); ++i) {
      risk_sum_buffer(i) = risk_sum_buffer(i) - ( event_cumsum(first(i)) - event_cumsum(last(i) + 1) ) * scaling(i);
    }
  }
}

double cox_dev(const Eigen::Ref<Eigen::VectorXd> eta, //eta is in native order  -- assumes centered (or otherwise normalized for numeric stability)
	       const Eigen::Ref<Eigen::VectorXd> sample_weight, //sample_weight is in native order
	       const Eigen::Ref<Eigen::VectorXd> exp_w,
	       const Eigen::Ref<Eigen::VectorXi> event_order,   
	       const Eigen::Ref<Eigen::VectorXi> start_order,
	       const Eigen::Ref<Eigen::VectorXi> status,        //everything below in event order
	       const Eigen::Ref<Eigen::VectorXi> first,
	       const Eigen::Ref<Eigen::VectorXi> last,
	       const Eigen::Ref<Eigen::VectorXd> scaling,
	       const Eigen::Ref<Eigen::VectorXi> event_map,
	       const Eigen::Ref<Eigen::VectorXi> start_map,
	       double loglik_sat,
	       Eigen::Ref<Eigen::VectorXd> T_1_term,
	       Eigen::Ref<Eigen::VectorXd> T_2_term,
	       Eigen::Ref<Eigen::VectorXd> grad_buffer,
	       Eigen::Ref<Eigen::VectorXd> diag_hessian_buffer,
	       Eigen::Ref<Eigen::VectorXd> diag_part_buffer,
	       Eigen::Ref<Eigen::VectorXd> w_avg_buffer,
	       py::list &event_reorder_buffers,
	       py::list &risk_sum_buffers,
	       py::list &forward_cumsum_buffers,
	       Eigen::Ref<Eigen::VectorXd> forward_scratch_buffer,
	       py::list &reverse_cumsum_buffers,
	       bool have_start_times = true,
	       bool efron = false)
{
  // int n = eta.size();
    
  // eta_event: map first element of list into Eigen vector.
  MAP_PYLIST_BUF(event_reorder_buffers, 0, eta_event, tmp1)	
  to_event_from_native(eta, event_order, eta_event);

  // w_event: map second element of list into Eigen vector.
  MAP_PYLIST_BUF(event_reorder_buffers, 1, w_event, tmp2)	  
  to_event_from_native(sample_weight, event_order, w_event);

  // exp_eta_w_event: map third element of list into Eigen vector.
  MAP_PYLIST_BUF(event_reorder_buffers, 2, exp_eta_w_event, tmp3)	
  to_event_from_native(exp_w, event_order, exp_eta_w_event);

  // risk_sum_buffer[0]: map first element of list into Eigen vector.
  // We will name it risk_sums as that is what it is called in the ensuing code
  MAP_PYLIST_BUF(risk_sum_buffers, 0, risk_sums, tmp4)
    
  if (have_start_times) {
    sum_over_risk_set(exp_w, // native order
		      event_order,
		      start_order,
		      first,
		      last,
		      event_map,
		      scaling,
		      efron,
		      risk_sum_buffers, 
		      0, // 0 offset into risk_sum_buffer
		      reverse_cumsum_buffers, // We send the whole list even if only the first two will be used!
		      0); // we use zero offset
  } else {
    Eigen::VectorXi dummy;
    sum_over_risk_set(exp_w, // native order
		      event_order,
		      start_order,
		      first,
		      last,
		      dummy,
		      scaling,
		      efron,
		      risk_sum_buffers,
		      0, // 0 offset into risk_sum_buffer
		      reverse_cumsum_buffers, // We send the whole list even if only the first two will be used!
		      0); // we use zero offset
  }

  // event_cumsum: map first element of list into Eigen vector.
  MAP_PYLIST_BUF(reverse_cumsum_buffers, 0, event_cumsum, tmp5)
  // start_cumsum: map second element of list into Eigen vector.
  MAP_PYLIST_BUF(reverse_cumsum_buffers, 1, start_cumsum, tmp6)    
    

  // forward_cumsum_buffers[0]: map first element of list into Eigen vector.
  MAP_PYLIST_BUF(forward_cumsum_buffers, 0, forward_cumsum_buffers0, tmp7)
  // forward_cumsum_buffers[1]: map second element of list into Eigen vector.
  MAP_PYLIST_BUF(forward_cumsum_buffers, 1, forward_cumsum_buffers1, tmp8)
  // forward_cumsum_buffers[0]: map third element of list into Eigen vector.
  MAP_PYLIST_BUF(forward_cumsum_buffers, 2, forward_cumsum_buffers2, tmp9)
  // forward_cumsum_buffers[0]: map fourth element of list into Eigen vector.
  MAP_PYLIST_BUF(forward_cumsum_buffers, 3, forward_cumsum_buffers3, tmp10)
  // forward_cumsum_buffers[0]: map fifth element of list into Eigen vector.
  MAP_PYLIST_BUF(forward_cumsum_buffers, 4, forward_cumsum_buffers4, tmp11)

  // some ordered terms to complete likelihood
  // calculation

  // w_cumsum is only used here, can write over forward_cumsum_buffers
  // after computing w_avg

  // For us w_cumsum is forward_cumsum_buffers[0] which in C++ is forward_cumsum_buffers0
  for (int i = 0; i < w_avg_buffer.size(); ++i) {
    w_avg_buffer(i) = (forward_cumsum_buffers0(last(i) + 1) - forward_cumsum_buffers0(first(i))) / ((double) (last(i) + 1 - first(i)));
  }
  // w_avg = w_avg_buffer # shorthand
  double loglik = ( w_event.array() * eta_event.array() * status.cast<double>().array() ).sum() -
		   ( risk_sums.array().log() * w_avg_buffer.array() * status.cast<double>().array() ).sum();
    
  // forward cumsums for gradient and Hessian
  
  //# length of cumsums is n+1
  //# 0 is prepended for first(k)-1, start(k)-1 lookups
  //# a 1 is added to all indices

  Eigen::VectorXd dummy; // dummy argument for use where None is used
  
  forward_prework(status, w_avg_buffer, scaling, risk_sums, 0, 1, forward_scratch_buffer, dummy, true);
  Eigen::Ref<Eigen::VectorXd> A_01 = forward_scratch_buffer; // Make a reference rather than a copy
  forward_cumsum(A_01, forward_cumsum_buffers0); // length=n+1 
  Eigen::Ref<Eigen::VectorXd> C_01 = forward_cumsum_buffers0; // Make a reference rather than a copy
  
  forward_prework(status, w_avg_buffer, scaling, risk_sums, 0, 2, forward_scratch_buffer, dummy, true);
  Eigen::Ref<Eigen::VectorXd> A_02 = forward_scratch_buffer; // Make a reference rather than a copy
  forward_cumsum(A_02, forward_cumsum_buffers1); // # length=n+1
  Eigen::Ref<Eigen::VectorXd> C_02 = forward_cumsum_buffers1; // Make a reference rather than a copy
    
  if (!efron) {
    if (have_start_times) {

            // # +1 for start_map? depends on how  
            // # a tie between a start time and an event time
            // # if that means the start individual is excluded
            // # we should add +1, otherwise there should be
            // # no +1 in the [start_map+1] above
      for (int i = 0; i < last.size(); ++i) {
	T_1_term(i) = C_01(last(i) + 1) - C_01(start_map(i));
	T_2_term(i) = C_02(last(i) + 1) - C_02(start_map(i));
      }
    } else {
      for (int i = 0; i < last.size(); ++i) {
	T_1_term(i) = C_01(last(i) + 1);
	T_2_term(i) = C_02(last(i) + 1);
      }
    }
  } else {
    // # compute the other necessary cumsums
        
    forward_prework(status, w_avg_buffer, scaling, risk_sums, 1, 1, forward_scratch_buffer, dummy, true);
    Eigen::Ref<Eigen::VectorXd> A_11 = forward_scratch_buffer; // Make a reference rather than a copy
    forward_cumsum(A_11, forward_cumsum_buffers2); // # length=n+1
    Eigen::Ref<Eigen::VectorXd> C_11 = forward_cumsum_buffers2; // Make a reference rather than a copy

    forward_prework(status, w_avg_buffer, scaling, risk_sums, 2, 1, forward_scratch_buffer, dummy, true);
    Eigen::Ref<Eigen::VectorXd> A_21 = forward_scratch_buffer; // Make a reference rather than a copy
    forward_cumsum(A_21, forward_cumsum_buffers3); // # length=n+1
    Eigen::Ref<Eigen::VectorXd> C_21 = forward_cumsum_buffers3; // Make a reference rather than a copy

    forward_prework(status, w_avg_buffer, scaling, risk_sums, 2, 2, forward_scratch_buffer, dummy, true);
    Eigen::Ref<Eigen::VectorXd> A_22 = forward_scratch_buffer; // Make a reference rather than a copy
    forward_cumsum(A_22, forward_cumsum_buffers4); // # length=n+1
    Eigen::Ref<Eigen::VectorXd> C_22 = forward_cumsum_buffers4; // Make a reference rather than a copy

    for (int i = 0; i < last.size(); ++i) {
      T_1_term(i) = (C_01(last(i) + 1) - 
		     (C_11(last(i) + 1) - C_11(first(i))));
      T_2_term(i) = ((C_22(last(i) + 1) - C_22(first(i))) 
		      - 2 * (C_21(last(i) + 1) - C_21(first(i))) + 
		      C_02(last(i) + 1));
    }
    if (have_start_times) {
      for (int i = 0; i < start_map.size(); ++i) {
	T_1_term(i) -= C_01(start_map(i));
      }
      for (int i = 0; i < first.size(); ++i) {      
	T_2_term(i) -= C_02(first(i));
      }
    }
  }
  // # could do multiply by exp_w after reorder...
  // # save a reorder of w * exp(eta)
  
  diag_part_buffer = exp_eta_w_event.array() * T_1_term.array();
  grad_buffer = w_event.array() * status.cast<double>().array() - diag_part_buffer.array();
  grad_buffer.array() *= -2.0;
  
  // # now the diagonal of the Hessian
  
  diag_hessian_buffer = exp_eta_w_event.array().pow(2) * T_2_term.array() - diag_part_buffer.array();
  diag_hessian_buffer.array() *= -2.0;
  
  to_native_from_event(grad_buffer, event_order, forward_scratch_buffer);
  to_native_from_event(diag_hessian_buffer, event_order, forward_scratch_buffer);
  to_native_from_event(diag_part_buffer, event_order, forward_scratch_buffer);
  
  double deviance = 2.0 * (loglik_sat - loglik);
  return(deviance);
}

void hessian_matvec(const Eigen::Ref<Eigen::VectorXd> arg, // # arg is in native order
                    const Eigen::Ref<Eigen::VectorXd> eta, // # eta is in native order 
                    const Eigen::Ref<Eigen::VectorXd> sample_weight, //# sample_weight is in native order
                    const Eigen::Ref<Eigen::VectorXd> risk_sums,
                    const Eigen::Ref<Eigen::VectorXd> diag_part,
                    const Eigen::Ref<Eigen::VectorXd> w_avg,
                    const Eigen::Ref<Eigen::VectorXd> exp_w,
                    const Eigen::Ref<Eigen::VectorXd> event_cumsum,
                    const Eigen::Ref<Eigen::VectorXd> start_cumsum,
                    const Eigen::Ref<Eigen::VectorXi> event_order,   
                    const Eigen::Ref<Eigen::VectorXi> start_order,
                    const Eigen::Ref<Eigen::VectorXi> status, // # everything below in event order
                    const Eigen::Ref<Eigen::VectorXi> first,
                    const Eigen::Ref<Eigen::VectorXi> last,
                    const Eigen::Ref<Eigen::VectorXd> scaling,
                    const Eigen::Ref<Eigen::VectorXi> event_map,
                    const Eigen::Ref<Eigen::VectorXi> start_map,
		    py::list &risk_sum_buffers,
                    py::list &forward_cumsum_buffers,
                    Eigen::Ref<Eigen::VectorXd> forward_scratch_buffer,
                    py::list &reverse_cumsum_buffers,
                    Eigen::Ref<Eigen::VectorXd> hess_matvec_buffer,
                    bool have_start_times = true,
                    bool efron = false) {

  
  Eigen::VectorXd exp_w_times_arg = exp_w.array() * arg.array();
  
  if (have_start_times) {
    // # now in event_order
    sum_over_risk_set(exp_w_times_arg, // # in native order
		      event_order,
		      start_order,
		      first,
		      last,
		      event_map,
		      scaling,
		      efron,
		      risk_sum_buffers,
		      1, // offset 1 into risk_sum_buffers
		      reverse_cumsum_buffers,
		      2); // offset from index 2 of reverse_cumsum_buffers 
  } else {
    Eigen::VectorXi dummy;
    sum_over_risk_set(exp_w_times_arg, // # in native order
		      event_order,
		      start_order,
		      first,
		      last,
		      dummy,
		      scaling,
		      efron,
		      risk_sum_buffers,
		      1, // offset 1 into risk_sum_buffers
		      reverse_cumsum_buffers,
		      2);// offset from index 2 of reverse_cumsum_buffers 
  }
  // risk_sums_arg: map second element of list into Eigen vector.
  MAP_PYLIST_BUF(risk_sum_buffers, 1, risk_sums_arg, tmp1)
    
  // # E_arg = risk_sums_arg / risk_sums -- expecations under the probabilistic interpretation
  // # forward_scratch_buffer[:] = status * w_avg * E_arg / risk_sums

  // # one less step to compute from above representation
  forward_scratch_buffer = ( status.cast<double>().array() * w_avg.array() * risk_sums_arg.array() ) / risk_sums.array().pow(2);

#ifdef DEBUG
  std::cout << "forward_scratch_buffer" << std::endl;
  std::cout << forward_scratch_buffer << std::endl;
#endif
  
  if (have_start_times) {
    sum_over_events(event_order,
		    start_order,
		    first,
		    last,
		    start_map,
		    scaling,
		    status,
		    efron,
		    forward_cumsum_buffers,
		    forward_scratch_buffer,
		    hess_matvec_buffer);
  } else {
    Eigen::VectorXi dummy;
    sum_over_events(event_order,
		    start_order,
		    first,
		    last,
		    dummy,
		    scaling,
		    status,
		    efron,
		    forward_cumsum_buffers,
		    forward_scratch_buffer,
		    hess_matvec_buffer);
  }

#ifdef DEBUG
  std::cout << "hess_matvec_buffer" << std::endl;
  std::cout << hess_matvec_buffer << std::endl;
#endif
  
  to_native_from_event(hess_matvec_buffer, event_order, forward_scratch_buffer);

  // Eigen::VectorXd buffer = hess_matvec_buffer.array() * exp_w.array();
  hess_matvec_buffer = hess_matvec_buffer.array() * exp_w.array() - (diag_part.array() * arg.array());
#ifdef DEBUG
  std::cout << "hess_matvec_buffer" << std::endl;
  std::cout << hess_matvec_buffer << std::endl;
#endif
}
  

/* Start of C implementation of preprocess */

#include <vector>
#include <tuple>
#include <algorithm> // For std::sort and other algorithms

/**
 * Equivalent of numpy.lexsort for our case where a is stacked_is_start, b is stacked_status_c,
 * and c is stacked event time.
 */
std::vector<int> lexsort(const Eigen::VectorXi & a, 
                         const Eigen::VectorXi & b, 
                         const Eigen::VectorXd & c) {
  std::vector<int> idx(a.size());
  std::iota(idx.begin(), idx.end(), 0); // Fill idx with 0, 1, ..., a.size() - 1
  
  auto comparator = [&](int i, int j) {
    if (c[i] != c[j]) return c[i] < c[j];
    if (b[i] != b[j]) return b[i] < b[j];
    return a[i] < a[j];
  };
  
  std::sort(idx.begin(), idx.end(), comparator);
  
  return idx;
}

/**
 * Compute various functions of the start / event / status to be used to help in computing cumsums
 * This is best done in C++ also to avoid dealing with 1-based indexing in R  and 0-based indexing 
 * elsewhere.
 */
std::tuple<py::dict, Eigen::VectorXi, Eigen::VectorXi> preprocess(const Eigen::Ref<Eigen::VectorXd> start,
								  const Eigen::Ref<Eigen::VectorXd> event,
								  const Eigen::Ref<Eigen::VectorXi> status)
{
  int nevent = status.size();
  Eigen::VectorXi ones = Eigen::VectorXi::Ones(nevent);
  Eigen::VectorXi zeros = Eigen::VectorXi::Zero(nevent);

  // second column of stacked_array is 1-status...
  Eigen::VectorXd stacked_time(nevent + nevent);
  stacked_time.segment(0, nevent) = start;
  stacked_time.segment(nevent, nevent) = event;

  Eigen::VectorXi stacked_status_c(nevent + nevent);
  stacked_status_c.segment(0, nevent) = ones;
  stacked_status_c.segment(nevent, nevent) = ones - status; // complement of status

  Eigen::VectorXi stacked_is_start(nevent + nevent);
  stacked_is_start.segment(0, nevent) = ones;
  stacked_is_start.segment(nevent, nevent) = zeros;

  Eigen::VectorXi stacked_index(nevent + nevent);
  stacked_index.segment(0, nevent) = Eigen::VectorXi::LinSpaced(nevent, 0, nevent - 1);
  stacked_index.segment(nevent, nevent) =  Eigen::VectorXi::LinSpaced(nevent, 0, nevent - 1);

  std::vector<int> sort_order = lexsort(stacked_is_start, stacked_status_c, stacked_time);
  Eigen::VectorXi argsort = Eigen::Map<const Eigen::VectorXi>(sort_order.data(), sort_order.size());

  // Since they are all the same size, we can put them in one loop for efficiency!
  Eigen::VectorXd sorted_time(stacked_time.size()), sorted_status(stacked_status_c.size()),
    sorted_is_start(stacked_is_start.size()), sorted_index(stacked_index.size());
  for (int i = 0; i < sorted_time.size(); ++i) {
    int j = argsort(i);
    sorted_time(i) = stacked_time(j);
    sorted_status(i) = 1 - stacked_status_c(j);
    sorted_is_start(i) = stacked_is_start(j);
    sorted_index(i) = stacked_index(j);    
  }

  // do the joint sort

  int event_count = 0, start_count = 0;
  std::vector<int> event_order_vec, start_order_vec, start_map_vec, event_map_vec, first_vec;
  int which_event = -1, first_event = -1, num_successive_event = 1;
  double last_row_time;
  bool last_row_time_set = false;

  for (int i = 0; i < sorted_time.size(); ++i) {
    double _time = sorted_time(i); 
    int _status = sorted_status(i);
    int _is_start = sorted_is_start(i);
    int _index = sorted_index(i);
    if (_is_start == 1) { //a start time
      start_order_vec.push_back(_index);
      start_map_vec.push_back(event_count);
      start_count++;
    } else { // an event / stop time
      if (_status == 1) {
	// if it's an event and the time is same as last row 
	// it is the same event
	// else it's the next "which_event"
	// CHANGED THE ORIGINAL COMPARISON time != last_row_time below to
	// _time > last_row_time since time is sorted! 
	if (last_row_time_set  && _time > last_row_time) {// # index of next `status==1` 
	  first_event += num_successive_event;
	  num_successive_event = 1;
	  which_event++;
	} else {
	  num_successive_event++;
	}
	first_vec.push_back(first_event);
      } else {
	first_event += num_successive_event;
	num_successive_event = 1;
	first_vec.push_back(first_event); // # this event time was not an failure time
      }
      event_map_vec.push_back(start_count);
      event_order_vec.push_back(_index);
      event_count++;
    }
    last_row_time = _time;
    last_row_time_set = true;
  }

  // Except for start_order and event_order which are returned, we can probably not make copies
  // for others here.
  Eigen::VectorXi _first = Eigen::Map<Eigen::VectorXi>(first_vec.data(), first_vec.size());
  Eigen::VectorXi start_order = Eigen::Map<Eigen::VectorXi>(start_order_vec.data(), start_order_vec.size());
  Eigen::VectorXi event_order = Eigen::Map<Eigen::VectorXi>(event_order_vec.data(), event_order_vec.size());
  Eigen::VectorXi start_map = Eigen::Map<Eigen::VectorXi>(start_map_vec.data(), start_map_vec.size());
  Eigen::VectorXi _event_map = Eigen::Map<Eigen::VectorXi>(event_map_vec.data(), event_map_vec.size());

  // Eigen::VectorXi first(first_vec.size());
  // for (size_t i = 0; i < first.size(); ++i) {
  //   first[i] = first_vec[i];
  // }
  // Eigen::VectorXi start_order(start_order_vec.size());
  // for (size_t i = 0; i < start_order.size(); ++i) {
  //   start_order[i] = start_order_vec[i];
  // }
  // Eigen::VectorXi event_order(event_order_vec.size());
  // for (size_t i = 0; i < event_order.size(); ++i) {
  //   event_order[i] = event_order_vec[i];
  // }
  // Eigen::VectorXi start_map(start_map_vec.size());
  // for (size_t i = 0; i < start_map.size(); ++i) {
  //   start_map[i] = start_map_vec[i];
  // }
  // Eigen::VectorXi event_map(event_map_vec.size());
  // for (size_t i = 0; i < event_map.size(); ++i) {
  //   event_map[i] = event_map_vec[i];
  // }

  // reset start_map to original order
  Eigen::VectorXi start_map_cp = start_map;
  for (int i = 0; i < start_map.size(); ++i) {
    start_map(start_order(i)) = start_map_cp(i);
  }

  // set to event order
  Eigen::VectorXi _status(status.size());
  for (int i = 0; i < status.size(); ++i) {
    _status(i) = status(event_order(i));
  }
  
  // Eigen::VectorXi _first = first;
  
  Eigen::VectorXi _start_map(start_map.size());
  for (int i = 0; i < start_map.size(); ++i) {
    _start_map(i) = start_map(event_order(i));
  }

  // Eigen::VectorXi _event_map = event_map;

  Eigen::VectorXd _event(event.size());
  for (int i = 0; i < event.size(); ++i) {
    _event(i) = event(event_order(i));
  }

  Eigen::VectorXd _start(event.size());
  for (int i = 0; i < event.size(); ++i) {
    _start(i) = event(start_order(i));
  }

  std::vector<int> last_vec;
  int last_event = nevent - 1, first_size = _first.size();
  for (int i = 0; i < first_size; ++i) {
    int f = _first(first_size - i - 1);
    last_vec.push_back(last_event);
    // immediately following a last event, `first` will agree with np.arange
    if (f - (nevent - 1 - i) == 0) {
      last_event = f - 1;
    }
  }
  Eigen::VectorXi last = Eigen::Map<Eigen::VectorXi>(last_vec.data(), last_vec.size());  

  int last_size = last.size();
  Eigen::VectorXi _last(last_size);
  // Now reverse last into _last
  for (int i = 0; i < _last.size(); ++i) {
    _last(i) = last_vec[last_size - i - 1];
  }

  Eigen::VectorXd _scaling(nevent);
  for (int i = 0; i < nevent; ++i) {
    double fi = (double) _first(i);
    _scaling(i) = ((double) i - fi) / ((double) _last(i) + 1.0 - fi);
  }
  
  py::dict preproc;
  preproc["start"] = _start;
  preproc["event"] = _event;
  preproc["first"] = _first;
  preproc["last"] = _last;
  preproc["scaling"] = _scaling;
  preproc["start_map"] = _start_map;
  preproc["event_map"] = _event_map;
  preproc["status"] = _status;
  
  return std::make_tuple(preproc, event_order, start_order);
}


PYBIND11_MODULE(coxc, m) {
  m.doc() = "Cumsum implementations";
  m.def("forward_cumsum", &forward_cumsum, "Cumsum a vector");
  m.def("reverse_cumsums", &reverse_cumsums, "Reversed cumsum a vector");
  m.def("to_native_from_event", &to_native_from_event, "To Native from event");
  m.def("to_event_from_native", &to_event_from_native, "To Event from native");
  m.def("forward_prework", &forward_prework, "Cumsums of scaled and weighted quantities");
  m.def("compute_sat_loglik", &compute_sat_loglik, "Compute saturated log likelihood");
  m.def("cox_dev", &cox_dev, "Compute Cox deviance");
  m.def("hessian_matvec", &hessian_matvec, "Hessian Matrix Vector");
  m.def("c_preprocess", &preprocess, "C Preprocessing");
  
}
