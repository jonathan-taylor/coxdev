/**
 * Stratified Cox Proportional Hazards Model Implementation
 *
 * This file provides C++ implementations for stratified Cox models,
 * supporting both Python (pybind11) and R (Rcpp) interfaces.
 *
 * The stratified model fits independent baseline hazards for each stratum
 * while sharing regression coefficients across strata.
 *
 * IMPORTANT: The algorithm in cox_dev_single_stratum MUST be kept identical
 * to cox_dev in coxdev.cpp. Do NOT simplify or "improve" the algorithm.
 *
 * Architecture:
 * - Core implementation functions (*_impl) are interface-neutral
 * - Thin wrappers handle Python/R specific concerns (memory management, types)
 */

#ifdef PY_INTERFACE
#include "coxdev_strata.h"
#endif
#ifdef R_INTERFACE
#include "../inst/include/coxdev_strata.h"
#endif

#include <algorithm>
#include <cmath>
#include <numeric>

// Forward declarations for functions from coxdev.cpp that we'll reuse
void forward_cumsum(const EIGEN_REF<Eigen::VectorXd> sequence,
                    EIGEN_REF<Eigen::VectorXd> output);

void reverse_cumsums(const EIGEN_REF<Eigen::VectorXd> sequence,
                     EIGEN_REF<Eigen::VectorXd> event_buffer,
                     EIGEN_REF<Eigen::VectorXd> start_buffer,
                     const EIGEN_REF<Eigen::VectorXi> event_order,
                     const EIGEN_REF<Eigen::VectorXi> start_order,
                     bool do_event,
                     bool do_start);

void to_native_from_event(EIGEN_REF<Eigen::VectorXd> arg,
                          const EIGEN_REF<Eigen::VectorXi> event_order,
                          EIGEN_REF<Eigen::VectorXd> reorder_buffer);

void to_event_from_native(const EIGEN_REF<Eigen::VectorXd> arg,
                          const EIGEN_REF<Eigen::VectorXi> event_order,
                          EIGEN_REF<Eigen::VectorXd> reorder_buffer);

void forward_prework(const EIGEN_REF<Eigen::VectorXi> status,
                     const EIGEN_REF<Eigen::VectorXd> w_avg,
                     const EIGEN_REF<Eigen::VectorXd> scaling,
                     const EIGEN_REF<Eigen::VectorXd> risk_sums,
                     int i,
                     int j,
                     EIGEN_REF<Eigen::VectorXd> moment_buffer,
                     const EIGEN_REF<Eigen::VectorXd> arg,
                     bool use_w_avg);

void compute_weighted_scaling(
    const EIGEN_REF<Eigen::VectorXd> weights,
    const EIGEN_REF<Eigen::VectorXi> first,
    const EIGEN_REF<Eigen::VectorXi> last,
    EIGEN_REF<Eigen::VectorXd> scaling);

void compute_effective_cluster_sizes(
    const EIGEN_REF<Eigen::VectorXd> weights,
    const EIGEN_REF<Eigen::VectorXi> first,
    const EIGEN_REF<Eigen::VectorXi> last,
    EIGEN_REF<Eigen::VectorXd> effective_sizes);

// lexsort from coxdev.cpp
std::vector<int> lexsort(const Eigen::VectorXi & a,
                         const Eigen::VectorXi & b,
                         const Eigen::VectorXd & c);

// ============================================================================
// SINGLE STRATUM HELPER FUNCTIONS
// ============================================================================

/**
 * Internal function to preprocess a single stratum.
 * Replicates the logic from c_preprocess in coxdev.cpp but stores
 * results directly in CoxPreprocessed struct.
 */
static void preprocess_single_stratum(
    const Eigen::VectorXd& start_local,
    const Eigen::VectorXd& event_local,
    const Eigen::VectorXi& status_local,
    CoxPreprocessed<double, int>& preproc,
    bool efron)
{
    int nevent = status_local.size();
    preproc.resize(nevent);
    preproc.have_start_times = true;  // Will be determined by actual data
    preproc.efron = efron;

    Eigen::VectorXi ones = Eigen::VectorXi::Ones(nevent);
    Eigen::VectorXi zeros = Eigen::VectorXi::Zero(nevent);

    // Stack arrays for sorting
    Eigen::VectorXd stacked_time(nevent + nevent);
    stacked_time.segment(0, nevent) = start_local;
    stacked_time.segment(nevent, nevent) = event_local;

    Eigen::VectorXi stacked_status_c(nevent + nevent);
    stacked_status_c.segment(0, nevent) = ones;
    stacked_status_c.segment(nevent, nevent) = ones - status_local;

    Eigen::VectorXi stacked_is_start(nevent + nevent);
    stacked_is_start.segment(0, nevent) = ones;
    stacked_is_start.segment(nevent, nevent) = zeros;

    Eigen::VectorXi stacked_index(nevent + nevent);
    stacked_index.segment(0, nevent) = Eigen::VectorXi::LinSpaced(nevent, 0, nevent - 1);
    stacked_index.segment(nevent, nevent) = Eigen::VectorXi::LinSpaced(nevent, 0, nevent - 1);

    // Sort: primary by time, secondary by (1-status), tertiary by is_start
    std::vector<int> sort_order = lexsort(stacked_is_start, stacked_status_c, stacked_time);
    Eigen::VectorXi argsort = Eigen::Map<const Eigen::VectorXi>(sort_order.data(), sort_order.size());

    // Create sorted arrays
    Eigen::VectorXd sorted_time(stacked_time.size());
    Eigen::VectorXi sorted_status(stacked_status_c.size());
    Eigen::VectorXi sorted_is_start(stacked_is_start.size());
    Eigen::VectorXi sorted_index(stacked_index.size());
    for (int i = 0; i < sorted_time.size(); ++i) {
        int j = argsort(i);
        sorted_time(i) = stacked_time(j);
        sorted_status(i) = 1 - stacked_status_c(j);  // convert back to status
        sorted_is_start(i) = stacked_is_start(j);
        sorted_index(i) = stacked_index(j);
    }

    // Process the joint sort to build orders and maps
    int event_count = 0, start_count = 0;
    std::vector<int> event_order_vec, start_order_vec, start_map_vec, event_map_vec, first_vec;
    int first_event = -1, num_successive_event = 1;
    double last_row_time = 0.0;
    bool last_row_time_set = false;

    for (int i = 0; i < sorted_time.size(); ++i) {
        double _time = sorted_time(i);
        int _status = sorted_status(i);
        int _is_start = sorted_is_start(i);
        int _index = sorted_index(i);

        if (_is_start == 1) {  // a start time
            start_order_vec.push_back(_index);
            start_map_vec.push_back(event_count);
            start_count++;
        } else {  // an event / stop time
            if (_status == 1) {
                if (last_row_time_set && _time > last_row_time) {
                    first_event += num_successive_event;
                    num_successive_event = 1;
                } else {
                    num_successive_event++;
                }
                first_vec.push_back(first_event);
            } else {
                first_event += num_successive_event;
                num_successive_event = 1;
                first_vec.push_back(first_event);
            }
            event_map_vec.push_back(start_count);
            event_order_vec.push_back(_index);
            event_count++;
        }
        last_row_time = _time;
        last_row_time_set = true;
    }

    // Convert vectors to Eigen
    Eigen::VectorXi _first = Eigen::Map<Eigen::VectorXi>(first_vec.data(), first_vec.size());
    Eigen::VectorXi start_order_tmp = Eigen::Map<Eigen::VectorXi>(start_order_vec.data(), start_order_vec.size());
    Eigen::VectorXi event_order_tmp = Eigen::Map<Eigen::VectorXi>(event_order_vec.data(), event_order_vec.size());
    Eigen::VectorXi start_map_tmp = Eigen::Map<Eigen::VectorXi>(start_map_vec.data(), start_map_vec.size());
    Eigen::VectorXi _event_map = Eigen::Map<Eigen::VectorXi>(event_map_vec.data(), event_map_vec.size());

    // Copy to preproc struct
    preproc.event_order = event_order_tmp;
    preproc.start_order = start_order_tmp;
    preproc.event_map = _event_map;

    // Reset start_map to original order
    Eigen::VectorXi start_map_reord(nevent);
    for (int i = 0; i < nevent; ++i) {
        start_map_reord(start_order_tmp(i)) = start_map_tmp(i);
    }

    // Set status in event order
    for (int i = 0; i < nevent; ++i) {
        preproc.status(i) = status_local(preproc.event_order(i));
    }

    // Set first
    preproc.first = _first;

    // Set start_map in event order
    for (int i = 0; i < nevent; ++i) {
        preproc.start_map(i) = start_map_reord(preproc.event_order(i));
    }

    // Set event and start times
    for (int i = 0; i < nevent; ++i) {
        preproc.event(i) = event_local(preproc.event_order(i));
        preproc.start(i) = start_local(preproc.start_order(i));
    }

    // Compute last from first (reverse scan)
    std::vector<int> last_vec;
    int last_event = nevent - 1;
    int first_size = _first.size();
    for (int i = 0; i < first_size; ++i) {
        int f = _first(first_size - i - 1);
        last_vec.push_back(last_event);
        if (f - (nevent - 1 - i) == 0) {
            last_event = f - 1;
        }
    }
    // Reverse into preproc.last
    for (int i = 0; i < first_size; ++i) {
        preproc.last(i) = last_vec[first_size - i - 1];
    }

    // Compute scaling
    for (int i = 0; i < nevent; ++i) {
        double fi = static_cast<double>(preproc.first(i));
        preproc.scaling(i) = (static_cast<double>(i) - fi) /
                             (static_cast<double>(preproc.last(i)) + 1.0 - fi);
    }

    // Store original scaling for restoration after zero-weight adjustments
    preproc.original_scaling = preproc.scaling;
}

/**
 * Internal: compute saturated log-likelihood for a single stratum.
 * Uses the EXACT same logic as compute_sat_loglik in coxdev.cpp.
 */
static double compute_sat_loglik_stratum(
    const Eigen::VectorXi& first,
    const Eigen::VectorXi& last,
    const Eigen::VectorXd& weight,  // native order
    const Eigen::VectorXi& event_order,
    const Eigen::VectorXi& status,  // event order
    Eigen::VectorXd& W_status_buffer)  // workspace, size n+1
{
    int n = first.size();

    // Compute weight_event_order_times_status
    Eigen::VectorXd weight_event_order_times_status(n);
    for (int i = 0; i < n; ++i) {
        weight_event_order_times_status(i) = weight(event_order(i)) * status(i);
    }

    // Forward cumsum
    W_status_buffer(0) = 0.0;
    for (int i = 0; i < n; ++i) {
        W_status_buffer(i + 1) = W_status_buffer(i) + weight_event_order_times_status(i);
    }

    // Compute sums for each element
    Eigen::VectorXd sums(n);
    for (int i = 0; i < n; ++i) {
        sums(i) = W_status_buffer(last(i) + 1) - W_status_buffer(first(i));
    }

    double loglik_sat = 0.0;
    int prev_first = -1;
    for (int i = 0; i < n; ++i) {
        int f = first(i);
        double s = sums(i);
        if (s > 0 && f != prev_first) {
            loglik_sat -= s * std::log(s);
        }
        prev_first = f;
    }

    return loglik_sat;
}

/**
 * Internal: sum over risk set for a single stratum.
 * IDENTICAL to sum_over_risk_set in coxdev.cpp but uses workspace buffers.
 */
static void sum_over_risk_set_stratum(
    const Eigen::VectorXd& arg,  // native order
    const Eigen::VectorXi& event_order,
    const Eigen::VectorXi& start_order,
    const Eigen::VectorXi& first,
    const Eigen::VectorXi& last,
    const Eigen::VectorXi& event_map,
    const Eigen::VectorXd& scaling,
    bool efron,
    bool have_start_times,
    Eigen::VectorXd& risk_sum_buffer,  // output, length n
    Eigen::VectorXd& event_cumsum,     // buffer, length n+1
    Eigen::VectorXd& start_cumsum)     // buffer, length n+1
{
    int n = arg.size();

    // Reverse cumsum in event order
    double sum = 0.0;
    event_cumsum(n) = sum;
    for (int i = n - 1; i >= 0; --i) {
        sum = sum + arg(event_order(i));
        event_cumsum(i) = sum;
    }

    // Reverse cumsum in start order (for left truncation)
    if (have_start_times) {
        sum = 0.0;
        start_cumsum(n) = sum;
        for (int i = n - 1; i >= 0; --i) {
            sum = sum + arg(start_order(i));
            start_cumsum(i) = sum;
        }
    }

    // Compute risk sums
    if (have_start_times) {
        for (int i = 0; i < n; ++i) {
            risk_sum_buffer(i) = event_cumsum(first(i)) - start_cumsum(event_map(i));
        }
    } else {
        for (int i = 0; i < n; ++i) {
            risk_sum_buffer(i) = event_cumsum(first(i));
        }
    }

    // Efron correction
    if (efron) {
        for (int i = 0; i < n; ++i) {
            risk_sum_buffer(i) = risk_sum_buffer(i) -
                (event_cumsum(first(i)) - event_cumsum(last(i) + 1)) * scaling(i);
        }
    }
}

/**
 * Internal: sum over events for a single stratum.
 * IDENTICAL to sum_over_events in coxdev.cpp but uses workspace buffers.
 */
static void sum_over_events_stratum(
    const Eigen::VectorXi& event_order,
    const Eigen::VectorXi& start_order,
    const Eigen::VectorXi& first,
    const Eigen::VectorXi& last,
    const Eigen::VectorXi& start_map,
    const Eigen::VectorXd& scaling,
    const Eigen::VectorXi& status,
    bool efron,
    bool have_start_times,
    Eigen::VectorXd& forward_scratch_buffer,  // input in event order
    Eigen::VectorXd& C_arg,       // buffer, length n+1
    Eigen::VectorXd& C_arg_scale, // buffer, length n+1 (only used if efron)
    Eigen::VectorXd& value_buffer)  // output in event order
{
    int n = forward_scratch_buffer.size();

    // Forward cumsum
    C_arg(0) = 0.0;
    for (int i = 0; i < n; ++i) {
        C_arg(i + 1) = C_arg(i) + forward_scratch_buffer(i);
    }

    // Compute value_buffer
    if (have_start_times) {
        for (int i = 0; i < n; ++i) {
            value_buffer(i) = C_arg(last(i) + 1) - C_arg(start_map(i));
        }
    } else {
        for (int i = 0; i < n; ++i) {
            value_buffer(i) = C_arg(last(i) + 1);
        }
    }

    // Efron correction
    if (efron) {
        // Scale the scratch buffer
        Eigen::VectorXd scaled_scratch = forward_scratch_buffer.array() * scaling.array();

        // Forward cumsum of scaled
        C_arg_scale(0) = 0.0;
        for (int i = 0; i < n; ++i) {
            C_arg_scale(i + 1) = C_arg_scale(i) + scaled_scratch(i);
        }

        for (int i = 0; i < n; ++i) {
            value_buffer(i) -= (C_arg_scale(last(i) + 1) - C_arg_scale(first(i)));
        }
    }
}

/**
 * Internal: Cox deviance for a single stratum.
 *
 * CRITICAL: This function MUST use the EXACT same algorithm as cox_dev in coxdev.cpp.
 * Do NOT simplify or "improve" the algorithm. Every step must match.
 */
static double cox_dev_single_stratum(
    const Eigen::VectorXd& eta_local,      // native order, centered
    const Eigen::VectorXd& weight_local,   // native order
    CoxPreprocessed<double, int>& preproc,
    CoxWorkspace<double>& ws,
    double loglik_sat,
    bool have_start_times,
    bool efron_stratum)
{
    int n = preproc.n;

    // =========================================================================
    // Step 1: Compute exp(eta) * weight (in native order)
    // =========================================================================
    for (int i = 0; i < n; ++i) {
        ws.exp_w_buffer(i) = weight_local(i) * std::exp(eta_local(i));
    }

    // =========================================================================
    // Step 2: Reorder to event order (eta_event, w_event, exp_eta_w_event)
    // Using event_reorder_buffers[0], [1], [2]
    // =========================================================================
    Eigen::VectorXd& eta_event = ws.event_reorder_buffers[0];
    Eigen::VectorXd& w_event = ws.event_reorder_buffers[1];
    Eigen::VectorXd& exp_eta_w_event = ws.event_reorder_buffers[2];

    for (int i = 0; i < n; ++i) {
        eta_event(i) = eta_local(preproc.event_order(i));
        w_event(i) = weight_local(preproc.event_order(i));
        exp_eta_w_event(i) = ws.exp_w_buffer(preproc.event_order(i));
    }

    // =========================================================================
    // Step 3: Compute risk sums
    // Using risk_sum_buffers[0] for output, reverse_cumsum_buffers[0],[1] as workspace
    // =========================================================================
    Eigen::VectorXd& risk_sums = ws.risk_sum_buffers[0];
    Eigen::VectorXd& event_cumsum = ws.reverse_cumsum_buffers[0];
    Eigen::VectorXd& start_cumsum = ws.reverse_cumsum_buffers[1];

    sum_over_risk_set_stratum(
        ws.exp_w_buffer,  // native order
        preproc.event_order,
        preproc.start_order,
        preproc.first,
        preproc.last,
        preproc.event_map,
        preproc.scaling,
        efron_stratum,
        have_start_times,
        risk_sums,
        event_cumsum,
        start_cumsum);

    // =========================================================================
    // Step 4: Compute w_avg (average weights in tie groups)
    // Using forward_cumsum_buffers[0] for cumsum
    // =========================================================================
    Eigen::VectorXd& w_cumsum = ws.forward_cumsum_buffers[0];

    // Forward cumsum of w_event
    w_cumsum(0) = 0.0;
    for (int i = 0; i < n; ++i) {
        w_cumsum(i + 1) = w_cumsum(i) + w_event(i);
    }

    // Compute w_avg
    if (ws.use_zero_weight_handling) {
        // Use effective cluster sizes (for zero-weight handling)
        for (int i = 0; i < n; ++i) {
            if (ws.zero_weight_mask(i) == 0.0) {
                ws.w_avg_buffer(i) = 0.0;
                continue;
            }
            double eff_size = ws.effective_cluster_sizes(i);
            if (eff_size > 0) {
                ws.w_avg_buffer(i) = (w_cumsum(preproc.last(i) + 1) - w_cumsum(preproc.first(i))) / eff_size;
            } else {
                ws.w_avg_buffer(i) = 0.0;
            }
        }
    } else {
        // Standard case
        for (int i = 0; i < n; ++i) {
            ws.w_avg_buffer(i) = (w_cumsum(preproc.last(i) + 1) - w_cumsum(preproc.first(i))) /
                                 static_cast<double>(preproc.last(i) + 1 - preproc.first(i));
        }
    }

    // =========================================================================
    // Step 5: Compute log-likelihood
    // =========================================================================
    Eigen::VectorXd safe_log_risk_sums = risk_sums.array().max(1e-100).log();
    double loglik = (w_event.array() * eta_event.array() * preproc.status.cast<double>().array()).sum() -
                    (safe_log_risk_sums.array() * ws.w_avg_buffer.array() * preproc.status.cast<double>().array()).sum();

    // =========================================================================
    // Step 6: Forward prework and cumsums for gradient and Hessian
    // =========================================================================
    Eigen::VectorXd dummy;  // empty vector for forward_prework when arg is None
    Eigen::Map<Eigen::VectorXd> dummy_map(dummy.data(), dummy.size());

    // Aliases for forward_cumsum_buffers
    Eigen::VectorXd& C_01 = ws.forward_cumsum_buffers[0];
    Eigen::VectorXd& C_02 = ws.forward_cumsum_buffers[1];
    Eigen::VectorXd& C_11 = ws.forward_cumsum_buffers[2];
    Eigen::VectorXd& C_21 = ws.forward_cumsum_buffers[3];
    Eigen::VectorXd& C_22 = ws.forward_cumsum_buffers[4];

    // A_01 = status * w_avg * scaling^0 / risk_sums^1
    Eigen::Map<Eigen::VectorXi> status_map(preproc.status.data(), preproc.status.size());
    Eigen::Map<Eigen::VectorXd> w_avg_map(ws.w_avg_buffer.data(), ws.w_avg_buffer.size());
    Eigen::Map<Eigen::VectorXd> scaling_map(preproc.scaling.data(), preproc.scaling.size());
    Eigen::Map<Eigen::VectorXd> risk_sums_map(risk_sums.data(), risk_sums.size());
    Eigen::Map<Eigen::VectorXd> scratch_map(ws.forward_scratch_buffer.data(), ws.forward_scratch_buffer.size());
    Eigen::Map<Eigen::VectorXd> C_01_map(C_01.data(), C_01.size());
    Eigen::Map<Eigen::VectorXd> C_02_map(C_02.data(), C_02.size());

    forward_prework(status_map, w_avg_map, scaling_map, risk_sums_map, 0, 1, scratch_map, dummy_map, true);
    forward_cumsum(scratch_map, C_01_map);

    // A_02 = status * w_avg * scaling^0 / risk_sums^2
    forward_prework(status_map, w_avg_map, scaling_map, risk_sums_map, 0, 2, scratch_map, dummy_map, true);
    forward_cumsum(scratch_map, C_02_map);

    if (!efron_stratum) {
        // Non-Efron case
        if (have_start_times) {
            for (int i = 0; i < n; ++i) {
                ws.T_1_term(i) = C_01(preproc.last(i) + 1) - C_01(preproc.start_map(i));
                ws.T_2_term(i) = C_02(preproc.last(i) + 1) - C_02(preproc.start_map(i));
            }
        } else {
            for (int i = 0; i < n; ++i) {
                ws.T_1_term(i) = C_01(preproc.last(i) + 1);
                ws.T_2_term(i) = C_02(preproc.last(i) + 1);
            }
        }
    } else {
        // Efron case: need C_11, C_21, C_22 as well
        Eigen::Map<Eigen::VectorXd> C_11_map(C_11.data(), C_11.size());
        Eigen::Map<Eigen::VectorXd> C_21_map(C_21.data(), C_21.size());
        Eigen::Map<Eigen::VectorXd> C_22_map(C_22.data(), C_22.size());

        forward_prework(status_map, w_avg_map, scaling_map, risk_sums_map, 1, 1, scratch_map, dummy_map, true);
        forward_cumsum(scratch_map, C_11_map);

        forward_prework(status_map, w_avg_map, scaling_map, risk_sums_map, 2, 1, scratch_map, dummy_map, true);
        forward_cumsum(scratch_map, C_21_map);

        forward_prework(status_map, w_avg_map, scaling_map, risk_sums_map, 2, 2, scratch_map, dummy_map, true);
        forward_cumsum(scratch_map, C_22_map);

        for (int i = 0; i < n; ++i) {
            ws.T_1_term(i) = (C_01(preproc.last(i) + 1) -
                             (C_11(preproc.last(i) + 1) - C_11(preproc.first(i))));
            ws.T_2_term(i) = ((C_22(preproc.last(i) + 1) - C_22(preproc.first(i)))
                             - 2 * (C_21(preproc.last(i) + 1) - C_21(preproc.first(i))) +
                             C_02(preproc.last(i) + 1));
        }

        if (have_start_times) {
            for (int i = 0; i < n; ++i) {
                ws.T_1_term(i) -= C_01(preproc.start_map(i));
            }
            for (int i = 0; i < n; ++i) {
                ws.T_2_term(i) -= C_02(preproc.first(i));
            }
        }
    }

    // =========================================================================
    // Step 7: Compute gradient and diagonal Hessian (in event order)
    // =========================================================================
    ws.diag_part_buffer = exp_eta_w_event.array() * ws.T_1_term.array();
    ws.grad_buffer = w_event.array() * preproc.status.cast<double>().array() - ws.diag_part_buffer.array();
    ws.grad_buffer.array() *= -2.0;

    ws.diag_hessian_buffer = exp_eta_w_event.array().pow(2) * ws.T_2_term.array() - ws.diag_part_buffer.array();
    ws.diag_hessian_buffer.array() *= -2.0;

    // =========================================================================
    // Step 8: Convert to native order
    // =========================================================================
    // Reorder grad_buffer from event order to native order
    Eigen::VectorXd temp = ws.grad_buffer;
    for (int i = 0; i < n; ++i) {
        ws.grad_buffer(preproc.event_order(i)) = temp(i);
    }

    // Reorder diag_hessian_buffer from event order to native order
    temp = ws.diag_hessian_buffer;
    for (int i = 0; i < n; ++i) {
        ws.diag_hessian_buffer(preproc.event_order(i)) = temp(i);
    }

    // Reorder diag_part_buffer from event order to native order (needed for hessian_matvec)
    temp = ws.diag_part_buffer;
    for (int i = 0; i < n; ++i) {
        ws.diag_part_buffer(preproc.event_order(i)) = temp(i);
    }

    // =========================================================================
    // Step 9: Return deviance
    // =========================================================================
    double deviance = 2.0 * (loglik_sat - loglik);
    return deviance;
}

/**
 * Internal: Hessian matvec for a single stratum.
 *
 * CRITICAL: This function MUST use the EXACT same algorithm as hessian_matvec in coxdev.cpp.
 * Do NOT simplify or "improve" the algorithm.
 */
static void hessian_matvec_single_stratum(
    const Eigen::VectorXd& arg_local,  // native order
    CoxPreprocessed<double, int>& preproc,
    CoxWorkspace<double>& ws,
    bool have_start_times,
    bool efron_stratum,
    Eigen::VectorXd& result_local)  // output, native order
{
    int n = preproc.n;

    // exp_w * arg (native order)
    Eigen::VectorXd exp_w_times_arg = ws.exp_w_buffer.array() * arg_local.array();

    // Compute risk sums of exp_w * arg
    Eigen::VectorXd& risk_sums_arg = ws.risk_sum_buffers[1];
    Eigen::VectorXd& event_cumsum_arg = ws.reverse_cumsum_buffers[2];
    Eigen::VectorXd& start_cumsum_arg = ws.reverse_cumsum_buffers[3];

    sum_over_risk_set_stratum(
        exp_w_times_arg,
        preproc.event_order,
        preproc.start_order,
        preproc.first,
        preproc.last,
        preproc.event_map,
        preproc.scaling,
        efron_stratum,
        have_start_times,
        risk_sums_arg,
        event_cumsum_arg,
        start_cumsum_arg);

    // risk_sums is in risk_sum_buffers[0] (computed by cox_dev)
    Eigen::VectorXd& risk_sums = ws.risk_sum_buffers[0];

    // forward_scratch_buffer = status * w_avg * risk_sums_arg / risk_sums^2
    // Use safe division: when risk_sums is 0, set result to 0
    Eigen::ArrayXd risk_sums_sq = risk_sums.array().pow(2);
    Eigen::ArrayXd numerator = preproc.status.cast<double>().array()
                             * ws.w_avg_buffer.array()
                             * risk_sums_arg.array();
    ws.forward_scratch_buffer = (risk_sums_sq > 0.0).select(numerator / risk_sums_sq.max(1e-100), 0.0);

    // Sum over events
    Eigen::VectorXd& C_arg = ws.forward_cumsum_buffers[0];
    Eigen::VectorXd& C_arg_scale = ws.forward_cumsum_buffers[1];

    sum_over_events_stratum(
        preproc.event_order,
        preproc.start_order,
        preproc.first,
        preproc.last,
        preproc.start_map,
        preproc.scaling,
        preproc.status,
        efron_stratum,
        have_start_times,
        ws.forward_scratch_buffer,
        C_arg,
        C_arg_scale,
        ws.hess_matvec_buffer);

    // Reorder to native order
    Eigen::VectorXd temp = ws.hess_matvec_buffer;
    for (int i = 0; i < n; ++i) {
        ws.hess_matvec_buffer(preproc.event_order(i)) = temp(i);
    }

    // Final result
    result_local = ws.hess_matvec_buffer.array() * ws.exp_w_buffer.array()
                 - ws.diag_part_buffer.array() * arg_local.array();
}

// ============================================================================
// CORE IMPLEMENTATION FUNCTIONS (Interface-neutral)
// ============================================================================

/**
 * Core preprocessing implementation.
 * Populates a StratifiedCoxData structure with preprocessed data for all strata.
 */
static void preprocess_stratified_impl(
    const Eigen::VectorXd& start,
    const Eigen::VectorXd& event,
    const Eigen::VectorXi& status,
    const Eigen::VectorXi& strata,
    bool efron,
    StratifiedCoxData<double, int>& strat_data)
{
    int n_total = status.size();
    strat_data.n_total = n_total;

    // Find unique strata
    std::set<int> unique_strata_set;
    for (int i = 0; i < n_total; ++i) {
        unique_strata_set.insert(strata(i));
    }

    strat_data.n_strata = static_cast<int>(unique_strata_set.size());
    strat_data.resize(strat_data.n_strata);

    // Copy unique strata to vector
    int s_idx = 0;
    for (int s : unique_strata_set) {
        strat_data.strata_labels[s_idx++] = s;
    }

    // Build stratum_indices: for each stratum, collect global indices
    for (int s = 0; s < strat_data.n_strata; ++s) {
        int label = strat_data.strata_labels[s];
        for (int i = 0; i < n_total; ++i) {
            if (strata(i) == label) {
                strat_data.stratum_indices[s].push_back(i);
            }
        }
    }

    // Check for start times (not all -inf)
    bool have_start_times = false;
    for (int i = 0; i < n_total; ++i) {
        if (start(i) > -1e30) {
            have_start_times = true;
            break;
        }
    }

    // Preprocess each stratum
    for (int s = 0; s < strat_data.n_strata; ++s) {
        const std::vector<int>& idx = strat_data.stratum_indices[s];
        int n_s = static_cast<int>(idx.size());

        // Extract local data
        Eigen::VectorXd start_local(n_s);
        Eigen::VectorXd event_local(n_s);
        Eigen::VectorXi status_local(n_s);

        for (int i = 0; i < n_s; ++i) {
            start_local(i) = start(idx[i]);
            event_local(i) = event(idx[i]);
            status_local(i) = status(idx[i]);
        }

        // Preprocess this stratum
        preprocess_single_stratum(start_local, event_local, status_local,
                                  strat_data.preproc[s], efron);
        strat_data.preproc[s].have_start_times = have_start_times;

        // Allocate workspace
        strat_data.workspace[s].resize(n_s);

        // Initialize effective_cluster_sizes to 0 (will be computed at call time if needed)
        strat_data.workspace[s].effective_cluster_sizes.setZero();

        // Determine if Efron applies (only if there are ties)
        double scaling_norm = strat_data.preproc[s].scaling.norm();
        strat_data.efron_stratum[s] = efron && (scaling_norm > 0);

        // Initialize saturated log-likelihood to 0 (computed at call time with weights)
        strat_data.loglik_sat[s] = 0.0;
    }
}

/**
 * Core Cox deviance computation for stratified data.
 * Computes deviance, gradient, and diagonal Hessian for all strata.
 */
static double cox_dev_stratified_impl(
    const Eigen::VectorXd& eta,
    const Eigen::VectorXd& sample_weight,
    StratifiedCoxData<double, int>& strat_data,
    Eigen::VectorXd& grad_output,
    Eigen::VectorXd& diag_hess_output)
{
    double total_deviance = 0.0;

    // Zero output arrays
    grad_output.setZero();
    diag_hess_output.setZero();

    for (int s = 0; s < strat_data.n_strata; ++s) {
        const std::vector<int>& idx = strat_data.stratum_indices[s];
        int n_s = static_cast<int>(idx.size());
        CoxPreprocessed<double, int>& preproc = strat_data.preproc[s];
        CoxWorkspace<double>& ws = strat_data.workspace[s];

        // Extract local eta and weights (using pre-allocated workspace buffers)
        Eigen::VectorXd& eta_local = ws.eta_local_buffer;
        Eigen::VectorXd& weight_local = ws.weight_local_buffer;
        for (int i = 0; i < n_s; ++i) {
            eta_local(i) = eta(idx[i]);
            weight_local(i) = sample_weight(idx[i]);
        }

        // Center eta using weighted mean
        double weight_sum = weight_local.sum();
        double eta_mean;
        if (weight_sum > 0) {
            eta_mean = (eta_local.array() * weight_local.array()).sum() / weight_sum;
        } else {
            eta_mean = eta_local.mean();
        }
        eta_local.array() -= eta_mean;

        // Handle zero weights for Efron
        bool has_zero_weights = (weight_local.array() == 0).any();
        if (strat_data.efron_stratum[s] && has_zero_weights) {
            // Compute weight in event order (reuse event_reorder_buffers[1] as temporary)
            Eigen::VectorXd& w_event = ws.event_reorder_buffers[1];
            for (int i = 0; i < n_s; ++i) {
                w_event(i) = weight_local(preproc.event_order(i));
            }

            // Create maps for the function calls
            Eigen::Map<Eigen::VectorXd> w_event_map(w_event.data(), n_s);
            Eigen::Map<Eigen::VectorXi> first_map(preproc.first.data(), preproc.first.size());
            Eigen::Map<Eigen::VectorXi> last_map(preproc.last.data(), preproc.last.size());
            Eigen::Map<Eigen::VectorXd> scaling_map(preproc.scaling.data(), preproc.scaling.size());
            Eigen::Map<Eigen::VectorXd> eff_sizes_map(ws.effective_cluster_sizes.data(), ws.effective_cluster_sizes.size());

            compute_weighted_scaling(w_event_map, first_map, last_map, scaling_map);
            compute_effective_cluster_sizes(w_event_map, first_map, last_map, eff_sizes_map);

            // Set zero_weight_mask
            for (int i = 0; i < n_s; ++i) {
                ws.zero_weight_mask(i) = (w_event(i) > 0.0) ? 1.0 : 0.0;
            }
            ws.use_zero_weight_handling = true;  // enable zero-weight handling
        } else if (strat_data.efron_stratum[s]) {
            // Restore original scaling
            preproc.scaling = preproc.original_scaling;
            // Reset effective_cluster_sizes to indicate standard mode
            ws.effective_cluster_sizes.setZero();
            ws.use_zero_weight_handling = false;  // disable zero-weight handling
        } else {
            // Breslow or no ties - no zero-weight handling needed
            ws.use_zero_weight_handling = false;
        }

        // Compute saturated log-likelihood for this stratum
        strat_data.loglik_sat[s] = compute_sat_loglik_stratum(
            preproc.first,
            preproc.last,
            weight_local,
            preproc.event_order,
            preproc.status,
            ws.forward_cumsum_buffers[0]);

        // Compute deviance for this stratum
        double dev = cox_dev_single_stratum(
            eta_local,
            weight_local,
            preproc,
            ws,
            strat_data.loglik_sat[s],
            preproc.have_start_times,
            strat_data.efron_stratum[s]);

        total_deviance += dev;

        // Scatter results back to global arrays
        for (int i = 0; i < n_s; ++i) {
            grad_output(idx[i]) = ws.grad_buffer(i);
            diag_hess_output(idx[i]) = ws.diag_hessian_buffer(i);
        }
    }

    return total_deviance;
}

/**
 * Core Hessian matrix-vector product for stratified data.
 */
static void hessian_matvec_stratified_impl(
    const Eigen::VectorXd& arg,
    StratifiedCoxData<double, int>& strat_data,
    Eigen::VectorXd& result)
{
    // Zero output
    result.setZero();

    for (int s = 0; s < strat_data.n_strata; ++s) {
        const std::vector<int>& idx = strat_data.stratum_indices[s];
        int n_s = static_cast<int>(idx.size());
        CoxPreprocessed<double, int>& preproc = strat_data.preproc[s];
        CoxWorkspace<double>& ws = strat_data.workspace[s];

        // Extract local arg
        Eigen::VectorXd arg_local(n_s);
        for (int i = 0; i < n_s; ++i) {
            arg_local(i) = arg(idx[i]);
        }

        // Compute matvec for this stratum
        Eigen::VectorXd result_local(n_s);
        hessian_matvec_single_stratum(
            arg_local,
            preproc,
            ws,
            preproc.have_start_times,
            strat_data.efron_stratum[s],
            result_local);

        // Scatter back to global
        for (int i = 0; i < n_s; ++i) {
            result(idx[i]) = result_local(i);
        }
    }
}

// ============================================================================
// PYTHON INTERFACE
// ============================================================================

#ifdef PY_INTERFACE

#include <pybind11/stl.h>

namespace py = pybind11;

/**
 * Python preprocess function - returns by value.
 */
STRAT_DATA_TYPE preprocess_stratified(
    const EIGEN_REF<Eigen::VectorXd> start,
    const EIGEN_REF<Eigen::VectorXd> event,
    const EIGEN_REF<Eigen::VectorXi> status,
    const EIGEN_REF<Eigen::VectorXi> strata,
    bool efron)
{
    STRAT_DATA_TYPE strat_data;
    preprocess_stratified_impl(start, event, status, strata, efron, strat_data);
    return strat_data;
}

/**
 * Python wrapper class for stratified Cox deviance.
 */
class StratifiedCoxDevianceCpp {
public:
    STRAT_DATA_TYPE strat_data;
    bool have_start_times;

    StratifiedCoxDevianceCpp(
        py::array_t<double> event,
        py::array_t<int> status,
        py::array_t<int> strata,
        py::array_t<double> start,
        bool efron)
    {
        auto event_buf = event.request();
        auto status_buf = status.request();
        auto strata_buf = strata.request();
        auto start_buf = start.request();

        int n = static_cast<int>(event_buf.size);

        Eigen::Map<Eigen::VectorXd> event_vec(static_cast<double*>(event_buf.ptr), n);
        Eigen::Map<Eigen::VectorXi> status_vec(static_cast<int*>(status_buf.ptr), n);
        Eigen::Map<Eigen::VectorXi> strata_vec(static_cast<int*>(strata_buf.ptr), n);
        Eigen::Map<Eigen::VectorXd> start_vec(static_cast<double*>(start_buf.ptr), n);

        // Check if we have actual start times (not all -inf)
        have_start_times = false;
        for (int i = 0; i < n; ++i) {
            if (start_vec(i) > -1e30) {
                have_start_times = true;
                break;
            }
        }

        // Use core implementation
        preprocess_stratified_impl(start_vec, event_vec, status_vec, strata_vec, efron, strat_data);

        // Set have_start_times for all strata
        for (int s = 0; s < strat_data.n_strata; ++s) {
            strat_data.preproc[s].have_start_times = have_start_times;
        }
    }

    py::tuple call(py::array_t<double> linear_predictor, py::array_t<double> sample_weight) {
        auto eta_buf = linear_predictor.request();
        auto weight_buf = sample_weight.request();
        int n = static_cast<int>(eta_buf.size);

        Eigen::Map<Eigen::VectorXd> eta_vec(static_cast<double*>(eta_buf.ptr), n);
        Eigen::Map<Eigen::VectorXd> weight_vec(static_cast<double*>(weight_buf.ptr), n);

        Eigen::VectorXd grad(n);
        Eigen::VectorXd diag_hess(n);

        // Use core implementation
        double deviance = cox_dev_stratified_impl(eta_vec, weight_vec, strat_data, grad, diag_hess);

        // Compute total loglik_sat
        double total_loglik_sat = 0.0;
        for (int s = 0; s < strat_data.n_strata; ++s) {
            total_loglik_sat += strat_data.loglik_sat[s];
        }

        // Return as numpy arrays
        py::array_t<double> grad_out(n);
        py::array_t<double> diag_hess_out(n);

        auto grad_ptr = grad_out.mutable_data();
        auto diag_hess_ptr = diag_hess_out.mutable_data();

        for (int i = 0; i < n; ++i) {
            grad_ptr[i] = grad(i);
            diag_hess_ptr[i] = diag_hess(i);
        }

        return py::make_tuple(deviance, total_loglik_sat, grad_out, diag_hess_out);
    }

    py::array_t<double> hessian_matvec(
        py::array_t<double> arg,
        py::array_t<double> linear_predictor,
        py::array_t<double> sample_weight)
    {
        auto arg_buf = arg.request();
        int n = static_cast<int>(arg_buf.size);

        Eigen::Map<Eigen::VectorXd> arg_vec(static_cast<double*>(arg_buf.ptr), n);

        Eigen::VectorXd result(n);

        // Use core implementation
        hessian_matvec_stratified_impl(arg_vec, strat_data, result);

        // Return as numpy array
        py::array_t<double> result_out(n);
        auto result_ptr = result_out.mutable_data();
        for (int i = 0; i < n; ++i) {
            result_ptr[i] = result(i);
        }

        return result_out;
    }

    int n_strata() const { return strat_data.n_strata; }
    int n_total() const { return strat_data.n_total; }
};

// Register Python bindings in the existing coxc module
void bind_stratified(py::module_& m) {
    py::class_<StratifiedCoxDevianceCpp>(m, "StratifiedCoxDevianceCpp")
        .def(py::init<py::array_t<double>, py::array_t<int>, py::array_t<int>, py::array_t<double>, bool>(),
             py::arg("event"), py::arg("status"), py::arg("strata"), py::arg("start"), py::arg("efron") = true)
        .def("__call__", &StratifiedCoxDevianceCpp::call,
             py::arg("linear_predictor"), py::arg("sample_weight"))
        .def("hessian_matvec", &StratifiedCoxDevianceCpp::hessian_matvec,
             py::arg("arg"), py::arg("linear_predictor"), py::arg("sample_weight"))
        .def_property_readonly("n_strata", &StratifiedCoxDevianceCpp::n_strata)
        .def_property_readonly("n_total", &StratifiedCoxDevianceCpp::n_total);
}

#endif // PY_INTERFACE

// ============================================================================
// R INTERFACE
// ============================================================================

#ifdef R_INTERFACE

/**
 * R interface for stratified Cox model using Rcpp and XPtr.
 *
 * The StratifiedCoxData object is managed via an external pointer (XPtr)
 * to avoid copying large data structures between R and C++.
 */

// Custom destructor for cleanup
inline void stratified_cox_data_finalizer(StratifiedCoxData<double, int>* ptr) {
    delete ptr;
}

// [[Rcpp::export(.preprocess_stratified)]]
SEXP preprocess_stratified_r(
    const Eigen::Map<Eigen::VectorXd> start,
    const Eigen::Map<Eigen::VectorXd> event,
    const Eigen::Map<Eigen::VectorXi> status,
    const Eigen::Map<Eigen::VectorXi> strata,
    bool efron)
{
    // Create the stratified data on the heap
    StratifiedCoxData<double, int>* strat_data_ptr = new StratifiedCoxData<double, int>();

    // Use core implementation
    // Need to convert Eigen::Map to Eigen::VectorXd for the impl function
    Eigen::VectorXd start_copy = start;
    Eigen::VectorXd event_copy = event;
    Eigen::VectorXi status_copy = status;
    Eigen::VectorXi strata_copy = strata;

    preprocess_stratified_impl(start_copy, event_copy, status_copy, strata_copy, efron, *strat_data_ptr);

    // Wrap in XPtr and return
    Rcpp::XPtr<StratifiedCoxData<double, int>> xptr(strat_data_ptr, true);
    return xptr;
}

// [[Rcpp::export(.cox_dev_stratified)]]
Rcpp::List cox_dev_stratified_r(
    SEXP strat_data_xptr,
    const Eigen::Map<Eigen::VectorXd> eta,
    const Eigen::Map<Eigen::VectorXd> sample_weight)
{
    // Extract the XPtr
    Rcpp::XPtr<StratifiedCoxData<double, int>> xptr(strat_data_xptr);
    StratifiedCoxData<double, int>& strat_data = *xptr;

    int n = eta.size();
    Eigen::VectorXd grad(n);
    Eigen::VectorXd diag_hess(n);

    // Convert Maps to VectorXd for impl function
    Eigen::VectorXd eta_copy = eta;
    Eigen::VectorXd weight_copy = sample_weight;

    // Use core implementation
    double total_deviance = cox_dev_stratified_impl(eta_copy, weight_copy, strat_data, grad, diag_hess);

    // Compute total loglik_sat
    double total_loglik_sat = 0.0;
    for (int s = 0; s < strat_data.n_strata; ++s) {
        total_loglik_sat += strat_data.loglik_sat[s];
    }

    return Rcpp::List::create(
        Rcpp::_["deviance"] = total_deviance,
        Rcpp::_["loglik_sat"] = total_loglik_sat,
        Rcpp::_["gradient"] = Rcpp::wrap(grad),
        Rcpp::_["diag_hessian"] = Rcpp::wrap(diag_hess)
    );
}

// [[Rcpp::export(.hessian_matvec_stratified)]]
Eigen::VectorXd hessian_matvec_stratified_r(
    SEXP strat_data_xptr,
    const Eigen::Map<Eigen::VectorXd> arg,
    const Eigen::Map<Eigen::VectorXd> eta,
    const Eigen::Map<Eigen::VectorXd> sample_weight)
{
    // Extract the XPtr
    Rcpp::XPtr<StratifiedCoxData<double, int>> xptr(strat_data_xptr);
    StratifiedCoxData<double, int>& strat_data = *xptr;

    int n = arg.size();
    Eigen::VectorXd result(n);

    // Convert Map to VectorXd
    Eigen::VectorXd arg_copy = arg;

    // Use core implementation
    hessian_matvec_stratified_impl(arg_copy, strat_data, result);

    return result;
}

// [[Rcpp::export(.get_n_strata)]]
int get_n_strata_r(SEXP strat_data_xptr) {
    Rcpp::XPtr<StratifiedCoxData<double, int>> xptr(strat_data_xptr);
    return xptr->n_strata;
}

// [[Rcpp::export(.get_n_total)]]
int get_n_total_r(SEXP strat_data_xptr) {
    Rcpp::XPtr<StratifiedCoxData<double, int>> xptr(strat_data_xptr);
    return xptr->n_total;
}

#endif // R_INTERFACE
