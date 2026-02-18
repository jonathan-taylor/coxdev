/**
 * Cox Proportional Hazards Model - Standalone Header-Only Library
 *
 * This file provides unified C++ implementations for Cox models.
 * It is the canonical source shared by:
 *   - coxdev R package (via Rcpp bindings)
 *   - coxdev Python package (via pybind11 bindings)
 *   - glmnet R package (via cox_adapter.hpp)
 *
 * Usage:
 *   - For coxdev R/Python: include directly, uses namespace coxdev::
 *   - For glmnet: define GLMNET_INTERFACE before include, uses namespace glmnetpp::coxdev::
 *
 * IMPORTANT: The algorithms are complex and correct. Do NOT simplify without thorough testing.
 */

#pragma once
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <vector>
#include <set>
#include <cmath>
#include <numeric>

#ifdef GLMNET_INTERFACE
#include <glmnetpp_bits/util/exceptions.hpp>
#else
#include <stdexcept>
#endif

// =============================================================================
// Namespace setup: glmnetpp::coxdev for glmnet, coxdev for standalone
// =============================================================================

#ifdef GLMNET_INTERFACE
namespace glmnetpp {
#endif

namespace coxdev {

// =============================================================================
// Data Structures
// =============================================================================

/**
 * Survival data structure for Cox regression.
 * Passed through the layers from driver to point solver.
 */
template <class ValueType, class IndexType>
struct CoxSurvivalData {
    using value_t = ValueType;
    using index_t = IndexType;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    using ivec_t = Eigen::Matrix<index_t, Eigen::Dynamic, 1>;

    // Survival response data
    vec_t start;   // start times (for left-truncation)
    vec_t stop;    // stop/event times
    ivec_t status; // event indicator (1=event, 0=censored)
    ivec_t strata; // strata labels (default: all same stratum)

    // Tie-breaking method
    bool efron;  // true for Efron, false for Breslow

    template <class StartType, class StopType, class StatusType>
    CoxSurvivalData(const StartType& start_, const StopType& stop_,
                    const StatusType& status_, bool efron_)
        : start(start_)
        , stop(stop_)
        , status(status_)
        , strata()  // empty strata signals single-stratum to preprocess_stratified
        , efron(efron_)
    {}

    template <class StartType, class StopType, class StatusType, class StrataType>
    CoxSurvivalData(const StartType& start_, const StopType& stop_,
                    const StatusType& status_, const StrataType& strata_, bool efron_)
        : start(start_)
        , stop(stop_)
        , status(status_)
        , strata(strata_)
        , efron(efron_)
    {}
};

/**
 * Preprocessed survival data for Cox regression.
 * All vectors are in event order unless otherwise noted.
 */
template <class ValueType, class IndexType>
struct CoxPreprocessed {
    using value_t = ValueType;
    using index_t = IndexType;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    using ivec_t = Eigen::Matrix<index_t, Eigen::Dynamic, 1>;

    ivec_t event_order;  // permutation: native -> event order
    ivec_t start_order;  // permutation: native -> start order
    ivec_t status;       // event status in event order
    ivec_t first;        // first index of tie group
    ivec_t last;         // last index of tie group
    vec_t scaling;       // Efron scaling factors (may be adjusted for zero weights)
    vec_t original_scaling;  // original scaling (before zero-weight correction)
    ivec_t event_map;    // maps event order to start count
    ivec_t start_map;    // maps event order to event count (at start time)
    vec_t event;         // event times in event order
    vec_t start;         // start times in start order
    bool have_start_times; // whether we have (start, stop) data
    bool efron;          // whether to use Efron tie-breaking
    int n;               // number of observations in this stratum

    void resize(int n_) {
        n = n_;
        event_order.resize(n);
        start_order.resize(n);
        status.resize(n);
        first.resize(n);
        last.resize(n);
        scaling.resize(n);
        original_scaling.resize(n);
        event.resize(n);
        start.resize(n);
        event_map.resize(n);
        start_map.resize(n);
    }
};

/**
 * Workspace buffers for Cox deviance/gradient/Hessian computation.
 * Pre-allocated to avoid repeated allocation during IRLS iterations.
 */
template <class ValueType>
struct CoxWorkspace {
    using value_t = ValueType;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;

    // Main output buffers (in native order after reordering)
    vec_t grad_buffer;
    vec_t diag_hessian_buffer;
    vec_t diag_part_buffer;

    // Intermediate buffers
    vec_t exp_w_buffer;
    vec_t T_1_term;
    vec_t T_2_term;
    vec_t w_avg_buffer;
    vec_t forward_scratch_buffer;

    // Event-order reorder buffers
    vec_t eta_event;
    vec_t w_event;
    vec_t exp_eta_w_event;

    // Risk sum buffers
    vec_t risk_sums;
    vec_t risk_sums_arg;

    // Forward cumsum buffers (length n+1)
    vec_t forward_cumsum_buffer0;
    vec_t forward_cumsum_buffer1;
    vec_t forward_cumsum_buffer2;
    vec_t forward_cumsum_buffer3;
    vec_t forward_cumsum_buffer4;

    // Reverse cumsum buffers (length n+1)
    vec_t event_cumsum;
    vec_t start_cumsum;
    vec_t event_cumsum2;
    vec_t start_cumsum2;

    // For hessian matvec
    vec_t hess_matvec_buffer;

    // W_status buffer for saturated loglik
    vec_t W_status_buffer;

    // Zero-weight handling buffers
    vec_t effective_cluster_sizes;  // count of non-zero weight obs per tie group
    vec_t zero_weight_mask;         // 1 for non-zero weight, 0 for zero weight

    void resize(int n) {
        grad_buffer.resize(n);
        diag_hessian_buffer.resize(n);
        diag_part_buffer.resize(n);
        exp_w_buffer.resize(n);
        T_1_term.resize(n);
        T_2_term.resize(n);
        w_avg_buffer.resize(n);
        forward_scratch_buffer.resize(n);

        eta_event.resize(n);
        w_event.resize(n);
        exp_eta_w_event.resize(n);

        risk_sums.resize(n);
        risk_sums_arg.resize(n);

        forward_cumsum_buffer0.resize(n + 1);
        forward_cumsum_buffer1.resize(n + 1);
        forward_cumsum_buffer2.resize(n + 1);
        forward_cumsum_buffer3.resize(n + 1);
        forward_cumsum_buffer4.resize(n + 1);

        event_cumsum.resize(n + 1);
        start_cumsum.resize(n + 1);
        event_cumsum2.resize(n + 1);
        start_cumsum2.resize(n + 1);

        hess_matvec_buffer.resize(n);
        W_status_buffer.resize(n + 1);

        // Zero-weight handling buffers
        effective_cluster_sizes.resize(n);
        zero_weight_mask.resize(n);
    }
};

/**
 * Container for stratified Cox model data.
 * Holds preprocessed data and workspace for all strata,
 * plus global-to-local index mappings.
 */
template <class ValueType, class IndexType>
struct StratifiedCoxData {
    using value_t = ValueType;
    using index_t = IndexType;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    using ivec_t = Eigen::Matrix<index_t, Eigen::Dynamic, 1>;

    int n_strata;                                   // number of strata
    int n_total;                                    // total number of observations
    std::vector<int> strata_labels;                 // unique strata values
    std::vector<std::vector<int>> stratum_indices;  // global indices per stratum
    std::vector<CoxPreprocessed<value_t, index_t>> preproc;  // per-stratum preprocessing
    std::vector<CoxWorkspace<value_t>> workspace;            // per-stratum workspace
    std::vector<value_t> loglik_sat;                // per-stratum saturated log-likelihood
    std::vector<bool> efron_stratum;                // per-stratum efron flag (may differ if no ties)

    // Additional buffers for local data extraction (per stratum)
    std::vector<vec_t> eta_local_buffers;
    std::vector<vec_t> weight_local_buffers;

    void resize(int n_strata_) {
        n_strata = n_strata_;
        strata_labels.resize(n_strata);
        stratum_indices.resize(n_strata);
        preproc.resize(n_strata);
        workspace.resize(n_strata);
        loglik_sat.resize(n_strata);
        efron_stratum.resize(n_strata);
        eta_local_buffers.resize(n_strata);
        weight_local_buffers.resize(n_strata);
    }

    // Resize a specific stratum
    void resize_stratum(int s, int n_s) {
        preproc[s].resize(n_s);
        workspace[s].resize(n_s);
        eta_local_buffers[s].resize(n_s);
        weight_local_buffers[s].resize(n_s);
    }
};

// =============================================================================
// Cumsum operations
// =============================================================================

/**
 * Compute cumsum with a padding of 0 at the beginning.
 * Output has length = input.size() + 1.
 */
template <class SeqType, class OutType>
inline void forward_cumsum(const SeqType& sequence, OutType& output) {
    int n = sequence.size();
    double sum = 0.0;
    output(0) = sum;
    for (int i = 1; i <= n; ++i) {
        sum += sequence(i - 1);
        output(i) = sum;
    }
}

/**
 * Compute reversed cumsums of a sequence in start and/or event order
 * with a 0 padded at the end (length = n + 1).
 */
template <class SeqType, class EventBufType, class StartBufType,
          class EventOrderType, class StartOrderType>
inline void reverse_cumsums(const SeqType& sequence,
                           EventBufType& event_buffer,
                           StartBufType& start_buffer,
                           const EventOrderType& event_order,
                           const StartOrderType& start_order,
                           bool do_event,
                           bool do_start) {
    int n = sequence.size();
    double sum = 0.0;

    if (do_event) {
        event_buffer(n) = sum;
        for (int i = n - 1; i >= 0; --i) {
            sum += sequence(event_order(i));
            event_buffer(i) = sum;
        }
    }

    if (do_start) {
        sum = 0.0;
        start_buffer(n) = sum;
        for (int i = n - 1; i >= 0; --i) {
            sum += sequence(start_order(i));
            start_buffer(i) = sum;
        }
    }
}

// =============================================================================
// Reordering operations
// =============================================================================

/**
 * Reorder an event-ordered vector into native order.
 */
template <class ArgType, class EventOrderType, class BufferType>
inline void to_native_from_event(ArgType& arg,
                                 const EventOrderType& event_order,
                                 BufferType& reorder_buffer) {
    int n = event_order.size();
    reorder_buffer.head(n) = arg.head(n);
    for (int i = 0; i < n; ++i) {
        arg(event_order(i)) = reorder_buffer(i);
    }
}

/**
 * Reorder a native-ordered vector into event order.
 */
template <class ArgType, class EventOrderType, class BufferType>
inline void to_event_from_native(const ArgType& arg,
                                 const EventOrderType& event_order,
                                 BufferType& reorder_buffer) {
    int n = event_order.size();
    for (int i = 0; i < n; ++i) {
        reorder_buffer(i) = arg(event_order(i));
    }
}

// =============================================================================
// Forward prework for gradient/Hessian computation
// =============================================================================

/**
 * Compute scaled/weighted quantities for cumsums.
 * moment_buffer = status * (w_avg if use_w_avg else 1) * scaling^i / risk_sums^j * (arg if provided else 1)
 * Uses safe division to avoid inf when risk_sums is 0 (can happen with zero-weight observations).
 * Uses 1e-100 as minimum to avoid underflow when raised to powers (1e-300^2 = 0 in double precision).
 */
template <class StatusType, class WAvgType, class ScalingType,
          class RiskSumsType, class MomentBufType, class ArgType>
inline void forward_prework(const StatusType& status,
                           const WAvgType& w_avg,
                           const ScalingType& scaling,
                           const RiskSumsType& risk_sums,
                           int i, int j,
                           MomentBufType& moment_buffer,
                           const ArgType* arg,
                           bool use_w_avg) {
    int n = status.size();
    for (int k = 0; k < n; ++k) {
        double val = static_cast<double>(status(k));
        if (use_w_avg) {
            val *= w_avg(k);
        }
        // Safe division: use max(risk_sums, 1e-100) to avoid inf
        double safe_risk_sum = std::max(risk_sums(k), 1e-100);
        val *= std::pow(scaling(k), i) / std::pow(safe_risk_sum, j);
        if (arg != nullptr) {
            val *= (*arg)(k);
        }
        moment_buffer(k) = val;
    }
}

/**
 * Overload without arg pointer (simpler interface).
 */
template <class StatusType, class WAvgType, class ScalingType,
          class RiskSumsType, class MomentBufType>
inline void forward_prework(const StatusType& status,
                           const WAvgType& w_avg,
                           const ScalingType& scaling,
                           const RiskSumsType& risk_sums,
                           int i, int j,
                           MomentBufType& moment_buffer,
                           bool use_w_avg = true) {
    int n = status.size();
    for (int k = 0; k < n; ++k) {
        double val = static_cast<double>(status(k));
        if (use_w_avg) {
            val *= w_avg(k);
        }
        // Safe division: use max(risk_sums, 1e-100) to avoid inf
        double safe_risk_sum = std::max(risk_sums(k), 1e-100);
        val *= std::pow(scaling(k), i) / std::pow(safe_risk_sum, j);
        moment_buffer(k) = val;
    }
}

// =============================================================================
// Saturated log-likelihood
// =============================================================================

/**
 * Compute saturated log-likelihood for Cox model.
 *
 * For Breslow: LL_sat = -sum_C W_C * log(W_C)
 * For Efron:   LL_sat = -sum_C W_C * [log(W_C) + (1/K_C) * (lgamma(K_C+1) - K_C*log(K_C))]
 *
 * where W_C is sum of weights in tie group C, and K_C is count of events with positive weight.
 */
template <class ValueType, class IndexType, class WeightType>
inline ValueType compute_sat_loglik(
        const Eigen::Matrix<IndexType, Eigen::Dynamic, 1>& first,
        const Eigen::Matrix<IndexType, Eigen::Dynamic, 1>& last,
        const WeightType& weight,
        const Eigen::Matrix<IndexType, Eigen::Dynamic, 1>& event_order,
        const Eigen::Matrix<IndexType, Eigen::Dynamic, 1>& status,
        Eigen::Matrix<ValueType, Eigen::Dynamic, 1>& W_status,
        bool efron = false) {

    int n = event_order.size();
    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> weight_event_order_times_status(n);
    for (int i = 0; i < n; ++i) {
        weight_event_order_times_status(i) = weight(event_order(i)) * status(i);
    }
    forward_cumsum(weight_event_order_times_status, W_status);

    ValueType loglik_sat = 0.0;
    int prev_first = -1;

    for (int i = 0; i < n; ++i) {
        int f = first(i);
        int l = last(i);
        if (f != prev_first) {
            // W_C = sum of weights for events in this tie group
            ValueType W_C = W_status(l + 1) - W_status(f);
            if (W_C > 0) {
                // Breslow term: -W_C * log(W_C)
                loglik_sat -= W_C * std::log(W_C);

                if (efron) {
                    // Efron penalty: -(W_C / K_C) * (lgamma(K_C+1) - K_C*log(K_C))
                    // K_C = effective cluster size (count of events with positive weight)
                    int K_C = 0;
                    for (int j = f; j <= l; ++j) {
                        if (weight_event_order_times_status(j) > 0) {
                            K_C++;
                        }
                    }
                    if (K_C > 1) {
                        ValueType penalty = (W_C / K_C) * (std::lgamma(K_C + 1) - K_C * std::log(K_C));
                        loglik_sat -= penalty;
                    }
                }
            }
        }
        prev_first = f;
    }
    return loglik_sat;
}

// =============================================================================
// Zero-weight handling utilities
// =============================================================================

/**
 * Computes effective cluster sizes (count of non-zero weight observations per tie group).
 *
 * @param weights weights in event order
 * @param first first indices for tie groups (event order)
 * @param last last indices for tie groups (event order)
 * @param effective_sizes output buffer for effective cluster sizes
 */
template <class WeightsType, class FirstType, class LastType, class EffSizesType>
inline void compute_effective_cluster_sizes(
        const WeightsType& weights,
        const FirstType& first,
        const LastType& last,
        EffSizesType& effective_sizes) {
    int nevent = weights.size();

    for (int i = 0; i < nevent; ++i) {
        int fi = first(i);
        int li = last(i);

        // Calculate the effective cluster size (count of non-zero weights)
        int effective_cluster_size = 0;
        for (int j = fi; j <= li; ++j) {
            if (weights(j) > 0.0) {
                effective_cluster_size++;
            }
        }
        effective_sizes(i) = static_cast<double>(effective_cluster_size);
    }
}

/**
 * Updates the scaling vector to account for zero-weighted observations.
 *
 * Logic:
 * 1. For each cluster of tied values (defined by first and last),
 *    calculate the "effective size" (count of observations where weights > 0).
 * 2. For each element i, calculate the "effective rank" (count of non-zero
 *    weighted observations in the same cluster appearing before index i).
 * 3. scaling(i) = effective_rank / effective_size.
 *
 * @param weights weights in event order
 * @param first first indices for tie groups (event order)
 * @param last last indices for tie groups (event order)
 * @param scaling output buffer for corrected scaling values
 */
template <class WeightsType, class FirstType, class LastType, class ScalingType>
inline void compute_weighted_scaling(
        const WeightsType& weights,
        const FirstType& first,
        const LastType& last,
        ScalingType& scaling) {
    int nevent = weights.size();

    for (int i = 0; i < nevent; ++i) {
        // If the weight is 0, the scaling value is effectively ignored in survival calculations,
        // but we'll set it to 0.0 for consistency.
        if (weights(i) <= 0.0) {
            scaling(i) = 0.0;
            continue;
        }

        int fi = first(i);
        int li = last(i);

        // 1. Calculate the effective cluster size (denominator)
        // This is the count of all non-zero weights in the range [fi, li]
        int effective_cluster_size = 0;
        for (int j = fi; j <= li; ++j) {
            if (weights(j) > 0.0) {
                effective_cluster_size++;
            }
        }

        // 2. Calculate the effective rank (numerator)
        // This is the count of non-zero weights in the range [fi, i-1]
        int effective_rank = 0;
        for (int j = fi; j < i; ++j) {
            if (weights(j) > 0.0) {
                effective_rank++;
            }
        }

        // 3. Perform division with safety check for clusters with only zero weights
        if (effective_cluster_size > 0) {
            scaling(i) = static_cast<double>(effective_rank) / static_cast<double>(effective_cluster_size);
        } else {
            scaling(i) = 0.0;
        }
    }
}

/**
 * Compute a mask for zero-weight observations (1=non-zero, 0=zero weight).
 *
 * @param weights weights in event order
 * @param mask output buffer for zero-weight mask
 */
template <class WeightsType, class MaskType>
inline void compute_zero_weight_mask(
        const WeightsType& weights,
        MaskType& mask) {
    int n = weights.size();
    for (int i = 0; i < n; ++i) {
        mask(i) = (weights(i) > 0.0) ? 1.0 : 0.0;
    }
}

// =============================================================================
// Sum over risk set
// =============================================================================

/**
 * Sum over risk set for Cox model.
 * arg is in native order, returns a sum in event order in risk_sum_buffer.
 */
template <class ValueType, class ArgType, class EventOrderType, class StartOrderType,
          class FirstType, class LastType, class EventMapType, class ScalingType,
          class RiskSumBufType, class EventCumsumType, class StartCumsumType>
inline void sum_over_risk_set(
        const ArgType& arg,
        const EventOrderType& event_order,
        const StartOrderType& start_order,
        const FirstType& first,
        const LastType& last,
        const EventMapType& event_map,
        const ScalingType& scaling,
        bool efron,
        bool have_start_times,
        RiskSumBufType& risk_sum_buffer,
        EventCumsumType& event_cumsum,
        StartCumsumType& start_cumsum) {

    int n = first.size();

    reverse_cumsums(arg, event_cumsum, start_cumsum,
                   event_order, start_order,
                   true, have_start_times);

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
            risk_sum_buffer(i) -= (event_cumsum(first(i)) - event_cumsum(last(i) + 1)) * scaling(i);
        }
    }
}

// =============================================================================
// Sum over events
// =============================================================================

/**
 * Sum over events for Cox model.
 */
template <class ValueType, class EventOrderType, class StartOrderType,
          class FirstType, class LastType, class StartMapType, class ScalingType,
          class StatusType, class ScratchBufType, class CumsumBuf0Type,
          class CumsumBuf1Type, class ValueBufType>
inline void sum_over_events(
        const EventOrderType& event_order,
        const StartOrderType& start_order,
        const FirstType& first,
        const LastType& last,
        const StartMapType& start_map,
        const ScalingType& scaling,
        const StatusType& status,
        bool efron,
        bool have_start_times,
        ScratchBufType& forward_scratch_buffer,
        CumsumBuf0Type& C_arg,
        CumsumBuf1Type& C_arg_scale,
        ValueBufType& value_buffer) {

    int n = last.size();

    forward_cumsum(forward_scratch_buffer, C_arg);

    if (have_start_times) {
        for (int i = 0; i < n; ++i) {
            value_buffer(i) = C_arg(last(i) + 1) - C_arg(start_map(i));
        }
    } else {
        for (int i = 0; i < n; ++i) {
            value_buffer(i) = C_arg(last(i) + 1);
        }
    }

    if (efron) {
        // Scale scratch buffer
        for (int i = 0; i < n; ++i) {
            forward_scratch_buffer(i) = forward_scratch_buffer(i) * scaling(i);
        }
        forward_cumsum(forward_scratch_buffer, C_arg_scale);
        for (int i = 0; i < n; ++i) {
            value_buffer(i) -= (C_arg_scale(last(i) + 1) - C_arg_scale(first(i)));
        }
    }
}

// =============================================================================
// Main cox_dev function
// =============================================================================

/**
 * Compute Cox deviance, gradient, and diagonal Hessian.
 * Full implementation with Efron tie-breaking and left-truncation support.
 *
 * @param eta Linear predictor in native order (assumed centered for stability)
 * @param sample_weight Sample weights in native order
 * @param preproc Preprocessed survival data
 * @param ws Workspace buffers
 * @param loglik_sat Saturated log-likelihood
 * @param efron Whether to use Efron tie-breaking
 * @return Deviance = 2 * (loglik_sat - loglik)
 */
template <class ValueType, class IndexType, class EtaType, class WeightType>
inline ValueType cox_dev(
        const EtaType& eta,
        const WeightType& sample_weight,
        const CoxPreprocessed<ValueType, IndexType>& preproc,
        CoxWorkspace<ValueType>& ws,
        ValueType loglik_sat,
        bool efron) {

    using vec_t = Eigen::Matrix<ValueType, Eigen::Dynamic, 1>;

    int n = eta.size();
    const auto& event_order = preproc.event_order;
    const auto& start_order = preproc.start_order;
    const auto& status = preproc.status;
    const auto& first = preproc.first;
    const auto& last = preproc.last;
    const auto& scaling = preproc.scaling;
    const auto& event_map = preproc.event_map;
    const auto& start_map = preproc.start_map;
    bool have_start_times = preproc.have_start_times;

    // Compute exp(eta) * weight
    ws.exp_w_buffer = (eta.array().exp() * sample_weight.array()).matrix();

    // Reorder to event order
    to_event_from_native(eta, event_order, ws.eta_event);
    to_event_from_native(sample_weight, event_order, ws.w_event);
    to_event_from_native(ws.exp_w_buffer, event_order, ws.exp_eta_w_event);

    // Compute risk sums
    sum_over_risk_set<ValueType>(ws.exp_w_buffer,
                                  event_order, start_order,
                                  first, last, event_map, scaling,
                                  efron, have_start_times,
                                  ws.risk_sums,
                                  ws.event_cumsum, ws.start_cumsum);

    // Compute w_avg for each tie group
    // First do a forward cumsum of w_event
    for (int i = 0; i < n; ++i) {
        ws.forward_scratch_buffer(i) = ws.w_event(i) * status(i);
    }
    forward_cumsum(ws.forward_scratch_buffer, ws.forward_cumsum_buffer0);

    // Check if there are any zero weights - FAST PATH optimization
    // The expensive O(n^2) compute_effective_cluster_sizes is only needed when zero weights exist
    bool has_zero_weights = (ws.w_event.array() == 0).any();

    if (has_zero_weights) {
        // SLOW PATH: Zero weights present - need expensive effective cluster size computation
        compute_effective_cluster_sizes(ws.w_event, first, last, ws.effective_cluster_sizes);
        compute_zero_weight_mask(ws.w_event, ws.zero_weight_mask);

        // Compute w_avg using effective cluster sizes (handles zero weights correctly)
        for (int i = 0; i < n; ++i) {
            // Zero out w_avg for zero-weight observations
            if (ws.zero_weight_mask(i) == 0.0) {
                ws.w_avg_buffer(i) = 0.0;
                continue;
            }
            ValueType eff_size = ws.effective_cluster_sizes(i);
            if (eff_size > 0) {
                ws.w_avg_buffer(i) = (ws.forward_cumsum_buffer0(last(i) + 1) - ws.forward_cumsum_buffer0(first(i))) / eff_size;
            } else {
                ws.w_avg_buffer(i) = 0.0;
            }
        }
    } else {
        // FAST PATH: No zero weights - use simple cluster size (last - first + 1)
        for (int i = 0; i < n; ++i) {
            int cluster_size = last(i) - first(i) + 1;
            if (cluster_size > 0) {
                ws.w_avg_buffer(i) = (ws.forward_cumsum_buffer0(last(i) + 1) - ws.forward_cumsum_buffer0(first(i))) / cluster_size;
            } else {
                ws.w_avg_buffer(i) = 0.0;
            }
        }
    }

    // Compute log-likelihood using safe log to avoid log(0)
    ValueType loglik = 0.0;
    for (int i = 0; i < n; ++i) {
        if (status(i) == 1) {
            // Safe log: use max(risk_sums, 1e-100) to avoid log(0)
            ValueType safe_risk_sum = std::max(ws.risk_sums(i), static_cast<ValueType>(1e-100));
            loglik += ws.w_event(i) * ws.eta_event(i) - ws.w_avg_buffer(i) * std::log(safe_risk_sum);
        }
    }

    // Forward cumsums for gradient and Hessian
    // A_01 = status * w_avg * scaling^0 / risk_sums^1
    forward_prework(status, ws.w_avg_buffer, scaling, ws.risk_sums, 0, 1,
                   ws.forward_scratch_buffer, true);
    forward_cumsum(ws.forward_scratch_buffer, ws.forward_cumsum_buffer0);
    auto& C_01 = ws.forward_cumsum_buffer0;

    // A_02 = status * w_avg * scaling^0 / risk_sums^2
    forward_prework(status, ws.w_avg_buffer, scaling, ws.risk_sums, 0, 2,
                   ws.forward_scratch_buffer, true);
    forward_cumsum(ws.forward_scratch_buffer, ws.forward_cumsum_buffer1);
    auto& C_02 = ws.forward_cumsum_buffer1;

    if (!efron) {
        if (have_start_times) {
            for (int i = 0; i < n; ++i) {
                ws.T_1_term(i) = C_01(last(i) + 1) - C_01(start_map(i));
                ws.T_2_term(i) = C_02(last(i) + 1) - C_02(start_map(i));
            }
        } else {
            for (int i = 0; i < n; ++i) {
                ws.T_1_term(i) = C_01(last(i) + 1);
                ws.T_2_term(i) = C_02(last(i) + 1);
            }
        }
    } else {
        // Efron: compute additional cumsums
        // C_11 = status * w_avg * scaling^1 / risk_sums^1
        forward_prework(status, ws.w_avg_buffer, scaling, ws.risk_sums, 1, 1,
                       ws.forward_scratch_buffer, true);
        forward_cumsum(ws.forward_scratch_buffer, ws.forward_cumsum_buffer2);
        auto& C_11 = ws.forward_cumsum_buffer2;

        // C_12 = status * w_avg * scaling^1 / risk_sums^2
        forward_prework(status, ws.w_avg_buffer, scaling, ws.risk_sums, 1, 2,
                       ws.forward_scratch_buffer, true);
        forward_cumsum(ws.forward_scratch_buffer, ws.forward_cumsum_buffer3);
        auto& C_12 = ws.forward_cumsum_buffer3;

        // C_22 = status * w_avg * scaling^2 / risk_sums^2
        forward_prework(status, ws.w_avg_buffer, scaling, ws.risk_sums, 2, 2,
                       ws.forward_scratch_buffer, true);
        forward_cumsum(ws.forward_scratch_buffer, ws.forward_cumsum_buffer4);
        auto& C_22 = ws.forward_cumsum_buffer4;

        for (int i = 0; i < n; ++i) {
            ws.T_1_term(i) = C_01(last(i) + 1) - (C_11(last(i) + 1) - C_11(first(i)));
            ws.T_2_term(i) = (C_22(last(i) + 1) - C_22(first(i)))
                            - 2.0 * (C_12(last(i) + 1) - C_12(first(i)))
                            + C_02(last(i) + 1);
        }
        if (have_start_times) {
            for (int i = 0; i < n; ++i) {
                ws.T_1_term(i) -= C_01(start_map(i));
            }
            for (int i = 0; i < n; ++i) {
                ws.T_2_term(i) -= C_02(first(i));
            }
        }
    }

    // Compute gradient and diagonal Hessian in event order
    ws.diag_part_buffer = ws.exp_eta_w_event.array() * ws.T_1_term.array();
    ws.grad_buffer = ws.w_event.array() * status.template cast<ValueType>().array() - ws.diag_part_buffer.array();
    ws.grad_buffer.array() *= -2.0;

    ws.diag_hessian_buffer = ws.exp_eta_w_event.array().pow(2) * ws.T_2_term.array()
                            - ws.diag_part_buffer.array();
    ws.diag_hessian_buffer.array() *= -2.0;

    // Reorder to native order
    to_native_from_event(ws.grad_buffer, event_order, ws.forward_scratch_buffer);
    to_native_from_event(ws.diag_hessian_buffer, event_order, ws.forward_scratch_buffer);
    to_native_from_event(ws.diag_part_buffer, event_order, ws.forward_scratch_buffer);

    return 2.0 * (loglik_sat - loglik);
}

// =============================================================================
// Hessian matrix-vector product
// =============================================================================

/**
 * Compute Hessian matrix-vector product for Cox model.
 */
template <class ValueType, class IndexType, class ArgType, class EtaType, class WeightType>
inline void hessian_matvec(
        const ArgType& arg,
        const EtaType& eta,
        const WeightType& sample_weight,
        const CoxPreprocessed<ValueType, IndexType>& preproc,
        CoxWorkspace<ValueType>& ws,
        bool efron,
        Eigen::Matrix<ValueType, Eigen::Dynamic, 1>& result) {

    int n = arg.size();
    const auto& event_order = preproc.event_order;
    const auto& start_order = preproc.start_order;
    const auto& status = preproc.status;
    const auto& first = preproc.first;
    const auto& last = preproc.last;
    const auto& scaling = preproc.scaling;
    const auto& event_map = preproc.event_map;
    const auto& start_map = preproc.start_map;
    bool have_start_times = preproc.have_start_times;

    // exp_w * arg
    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> exp_w_times_arg =
        ws.exp_w_buffer.array() * arg.array();

    // Sum over risk set for exp_w * arg
    sum_over_risk_set<ValueType>(exp_w_times_arg,
                                  event_order, start_order,
                                  first, last, event_map, scaling,
                                  efron, have_start_times,
                                  ws.risk_sums_arg,
                                  ws.event_cumsum2, ws.start_cumsum2);

    // forward_scratch = status * w_avg * risk_sums_arg / risk_sums^2
    // Use safe division: when risk_sums is 0, set result to 0 (no contribution from empty risk sets)
    for (int i = 0; i < n; ++i) {
        ValueType risk_sums_sq = ws.risk_sums(i) * ws.risk_sums(i);
        if (risk_sums_sq > 0.0) {
            // Use max to avoid division by very small numbers
            ValueType safe_risk_sums_sq = std::max(risk_sums_sq, static_cast<ValueType>(1e-100));
            ws.forward_scratch_buffer(i) = static_cast<ValueType>(status(i)) * ws.w_avg_buffer(i) *
                                            ws.risk_sums_arg(i) / safe_risk_sums_sq;
        } else {
            ws.forward_scratch_buffer(i) = 0.0;
        }
    }

    // Sum over events
    sum_over_events<ValueType>(event_order, start_order,
                                first, last, start_map, scaling, status,
                                efron, have_start_times,
                                ws.forward_scratch_buffer,
                                ws.forward_cumsum_buffer0,
                                ws.forward_cumsum_buffer1,
                                ws.hess_matvec_buffer);

    // Reorder to native order
    to_native_from_event(ws.hess_matvec_buffer, event_order, ws.forward_scratch_buffer);

    // result = hess_matvec_buffer * exp_w - diag_part * arg
    result = ws.hess_matvec_buffer.array() * ws.exp_w_buffer.array()
           - ws.diag_part_buffer.array() * arg.array();
}

// =============================================================================
// Preprocessing
// =============================================================================

/**
 * Lexsort for preprocessing: sort by (time, status_c, is_start).
 */
inline std::vector<int> lexsort(const Eigen::VectorXi& a,
                                const Eigen::VectorXi& b,
                                const Eigen::VectorXd& c) {
    std::vector<int> idx(a.size());
    std::iota(idx.begin(), idx.end(), 0);

    auto comparator = [&](int i, int j) {
        if (c[i] != c[j]) return c[i] < c[j];
        if (b[i] != b[j]) return b[i] < b[j];
        return a[i] < a[j];
    };

    std::sort(idx.begin(), idx.end(), comparator);
    return idx;
}

/**
 * Preprocess survival data for Cox regression.
 * Full implementation with left-truncation support.
 */
template <class ValueType, class IndexType, class StartType, class StopType, class StatusType>
inline CoxPreprocessed<ValueType, IndexType> preprocess(
        const StartType& start,
        const StopType& event,
        const StatusType& status_in) {

    using vec_t = Eigen::Matrix<ValueType, Eigen::Dynamic, 1>;
    using ivec_t = Eigen::Matrix<IndexType, Eigen::Dynamic, 1>;

    int nevent = status_in.size();
    CoxPreprocessed<ValueType, IndexType> result;

    // Check if we have real start times (not all zero)
    result.have_start_times = false;
    for (int i = 0; i < nevent; ++i) {
        if (start(i) != 0) {
            result.have_start_times = true;
            break;
        }
    }

    ivec_t ones = ivec_t::Ones(nevent);
    ivec_t zeros = ivec_t::Zero(nevent);

    // Stack arrays for lexsort
    vec_t stacked_time(2 * nevent);
    stacked_time.head(nevent) = start;
    stacked_time.tail(nevent) = event;

    ivec_t stacked_status_c(2 * nevent);
    stacked_status_c.head(nevent) = ones;
    stacked_status_c.tail(nevent) = ones - status_in.template cast<IndexType>();

    ivec_t stacked_is_start(2 * nevent);
    stacked_is_start.head(nevent) = ones;
    stacked_is_start.tail(nevent) = zeros;

    ivec_t stacked_index(2 * nevent);
    for (int i = 0; i < nevent; ++i) {
        stacked_index(i) = i;
        stacked_index(nevent + i) = i;
    }

    // Lexsort
    std::vector<int> sort_order = lexsort(stacked_is_start, stacked_status_c, stacked_time);

    // Process sorted data
    int event_count = 0, start_count = 0;
    std::vector<int> event_order_vec, start_order_vec, start_map_vec, event_map_vec, first_vec;
    int first_event = -1, num_successive_event = 1;
    double last_row_time = 0.0;
    bool last_row_time_set = false;

    for (int idx : sort_order) {
        double _time = stacked_time(idx);
        int _status = 1 - stacked_status_c(idx);
        int _is_start = stacked_is_start(idx);
        int _index = stacked_index(idx);

        if (_is_start == 1) {
            start_order_vec.push_back(_index);
            start_map_vec.push_back(event_count);
            start_count++;
        } else {
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
    result.event_order.resize(nevent);
    result.start_order.resize(nevent);
    result.first.resize(nevent);
    ivec_t start_map_orig(nevent);
    result.event_map.resize(nevent);

    for (int i = 0; i < nevent; ++i) {
        result.event_order(i) = event_order_vec[i];
        result.start_order(i) = start_order_vec[i];
        result.first(i) = first_vec[i];
        start_map_orig(i) = start_map_vec[i];
        result.event_map(i) = event_map_vec[i];
    }

    // Reset start_map to original order then to event order
    ivec_t start_map_native(nevent);
    for (int i = 0; i < nevent; ++i) {
        start_map_native(result.start_order(i)) = start_map_orig(i);
    }
    result.start_map.resize(nevent);
    for (int i = 0; i < nevent; ++i) {
        result.start_map(i) = start_map_native(result.event_order(i));
    }

    // Status in event order
    result.status.resize(nevent);
    for (int i = 0; i < nevent; ++i) {
        result.status(i) = status_in(result.event_order(i));
    }

    // Event times in event order
    result.event.resize(nevent);
    for (int i = 0; i < nevent; ++i) {
        result.event(i) = event(result.event_order(i));
    }

    // Start times in start order
    result.start.resize(nevent);
    for (int i = 0; i < nevent; ++i) {
        result.start(i) = start(result.start_order(i));
    }

    // Compute last indices
    std::vector<int> last_vec;
    int last_event_idx = nevent - 1;
    for (int i = 0; i < nevent; ++i) {
        int rev_i = nevent - 1 - i;
        int f = result.first(rev_i);
        last_vec.push_back(last_event_idx);
        if (f - rev_i == 0) {
            last_event_idx = f - 1;
        }
    }
    result.last.resize(nevent);
    for (int i = 0; i < nevent; ++i) {
        result.last(i) = last_vec[nevent - 1 - i];
    }

    // Compute scaling factors
    result.scaling.resize(nevent);
    for (int i = 0; i < nevent; ++i) {
        ValueType fi = static_cast<ValueType>(result.first(i));
        result.scaling(i) = (static_cast<ValueType>(i) - fi) /
                           (static_cast<ValueType>(result.last(i)) + 1.0 - fi);
    }

    return result;
}

// =============================================================================
// Lambda max computation
// =============================================================================

/**
 * Compute lambda_max for Cox regression with elastic net penalty.
 */
template <class XType, class ValueType, class IndexType, class EtaType, class WeightType,
          class XMType, class XSType, class VPType, class ExcludeType>
inline ValueType compute_lambda_max(
        const XType& X,
        const EtaType& eta,
        const WeightType& sample_weight,
        const XMType& xm,
        const XSType& xs,
        const VPType& vp,
        const ExcludeType& exclude,
        ValueType alpha,
        const CoxPreprocessed<ValueType, IndexType>& preproc,
        CoxWorkspace<ValueType>& workspace,
        ValueType loglik_sat,
        bool efron) {

    int nvars = X.cols();
    int nobs = X.rows();

    // Compute gradient at eta
    cox_dev<ValueType, IndexType>(eta, sample_weight, preproc, workspace, loglik_sat, efron);

    // Scale gradient from deviance (-2 * loglik) to loglik scale
    // cox_dev returns grad_buffer = -2 * d(loglik)/d(eta)
    // We want d(loglik)/d(eta) to match coxgrad() in R
    workspace.grad_buffer /= (-2.0);

    // Create exclude set
    std::set<int> exclude_set;
    for (int i = 0; i < exclude.size(); ++i) {
        exclude_set.insert(exclude(i));
    }

    ValueType grad_sum = workspace.grad_buffer.sum();
    ValueType max_g = 0.0;

    for (int j = 0; j < nvars; ++j) {
        if (exclude_set.count(j) > 0 || vp(j) <= 0) continue;

        // Inner product: grad' * X[,j]
        ValueType inner_prod = workspace.grad_buffer.dot(X.col(j));

        // Adjust for centering and scaling
        ValueType g_j = std::abs((inner_prod - grad_sum * xm(j)) / xs(j));

        // Adjust by penalty factor
        g_j /= vp(j);

        if (g_j > max_g) max_g = g_j;
    }

    return max_g / std::max(alpha, static_cast<ValueType>(1e-3));
}

/**
 * Compute lambda_max for sparse X matrices.
 */
template <class ValueType, class IndexType, class EtaType, class WeightType,
          class XMType, class XSType, class VPType, class ExcludeType>
inline ValueType compute_lambda_max_sparse(
        const Eigen::SparseMatrix<ValueType>& X,
        const EtaType& eta,
        const WeightType& sample_weight,
        const XMType& xm,
        const XSType& xs,
        const VPType& vp,
        const ExcludeType& exclude,
        ValueType alpha,
        const CoxPreprocessed<ValueType, IndexType>& preproc,
        CoxWorkspace<ValueType>& workspace,
        ValueType loglik_sat,
        bool efron) {

    int nvars = X.cols();
    int nobs = X.rows();

    // Compute gradient at eta
    cox_dev<ValueType, IndexType>(eta, sample_weight, preproc, workspace, loglik_sat, efron);

    // Scale gradient from deviance (-2 * loglik) to loglik scale
    // cox_dev returns grad_buffer = -2 * d(loglik)/d(eta)
    // We want d(loglik)/d(eta) to match coxgrad() in R
    workspace.grad_buffer /= (-2.0);

    // Create exclude set
    std::set<int> exclude_set;
    for (int i = 0; i < exclude.size(); ++i) {
        exclude_set.insert(exclude(i));
    }

    ValueType grad_sum = workspace.grad_buffer.sum();
    ValueType max_g = 0.0;

    for (int j = 0; j < nvars; ++j) {
        if (exclude_set.count(j) > 0 || vp(j) <= 0) continue;

        // Inner product for sparse column
        ValueType inner_prod = 0.0;
        for (typename Eigen::SparseMatrix<ValueType>::InnerIterator it(X, j); it; ++it) {
            inner_prod += workspace.grad_buffer(it.row()) * it.value();
        }

        // Adjust for centering and scaling
        ValueType g_j = std::abs((inner_prod - grad_sum * xm(j)) / xs(j));

        // Adjust by penalty factor
        g_j /= vp(j);

        if (g_j > max_g) max_g = g_j;
    }

    return max_g / std::max(alpha, static_cast<ValueType>(1e-3));
}

// =============================================================================
// Stratified Cox preprocessing and computation
// =============================================================================

/**
 * Preprocess a single stratum of survival data.
 * Similar to preprocess() but stores result in an existing CoxPreprocessed struct.
 */
template <class ValueType, class IndexType, class StartType, class StopType, class StatusType>
inline void preprocess_single_stratum(
        const StartType& start,
        const StopType& event,
        const StatusType& status_in,
        CoxPreprocessed<ValueType, IndexType>& result,
        bool efron) {

    using vec_t = Eigen::Matrix<ValueType, Eigen::Dynamic, 1>;
    using ivec_t = Eigen::Matrix<IndexType, Eigen::Dynamic, 1>;

    int nevent = status_in.size();
    result.resize(nevent);
    result.efron = efron;

    // Check if we have real start times (not all zero)
    result.have_start_times = false;
    for (int i = 0; i < nevent; ++i) {
        if (start(i) != 0) {
            result.have_start_times = true;
            break;
        }
    }

    ivec_t ones = ivec_t::Ones(nevent);
    ivec_t zeros = ivec_t::Zero(nevent);

    // Stack arrays for lexsort
    vec_t stacked_time(2 * nevent);
    stacked_time.head(nevent) = start;
    stacked_time.tail(nevent) = event;

    ivec_t stacked_status_c(2 * nevent);
    stacked_status_c.head(nevent) = ones;
    stacked_status_c.tail(nevent) = ones - status_in.template cast<IndexType>();

    ivec_t stacked_is_start(2 * nevent);
    stacked_is_start.head(nevent) = ones;
    stacked_is_start.tail(nevent) = zeros;

    ivec_t stacked_index(2 * nevent);
    for (int i = 0; i < nevent; ++i) {
        stacked_index(i) = i;
        stacked_index(nevent + i) = i;
    }

    // Lexsort
    std::vector<int> sort_order = lexsort(stacked_is_start, stacked_status_c, stacked_time);

    // Process sorted data
    int event_count = 0, start_count = 0;
    std::vector<int> event_order_vec, start_order_vec, start_map_vec, event_map_vec, first_vec;
    int first_event = -1, num_successive_event = 1;
    double last_row_time = 0.0;
    bool last_row_time_set = false;

    for (int idx : sort_order) {
        double _time = stacked_time(idx);
        int _status = 1 - stacked_status_c(idx);
        int _is_start = stacked_is_start(idx);
        int _index = stacked_index(idx);

        if (_is_start == 1) {
            start_order_vec.push_back(_index);
            start_map_vec.push_back(event_count);
            start_count++;
        } else {
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
    for (int i = 0; i < nevent; ++i) {
        result.event_order(i) = event_order_vec[i];
        result.start_order(i) = start_order_vec[i];
        result.first(i) = first_vec[i];
        result.event_map(i) = event_map_vec[i];
    }

    // Reset start_map to original order then to event order
    ivec_t start_map_orig(nevent);
    for (int i = 0; i < nevent; ++i) {
        start_map_orig(i) = start_map_vec[i];
    }

    ivec_t start_map_native(nevent);
    for (int i = 0; i < nevent; ++i) {
        start_map_native(result.start_order(i)) = start_map_orig(i);
    }
    for (int i = 0; i < nevent; ++i) {
        result.start_map(i) = start_map_native(result.event_order(i));
    }

    // Status in event order
    for (int i = 0; i < nevent; ++i) {
        result.status(i) = status_in(result.event_order(i));
    }

    // Event times in event order
    for (int i = 0; i < nevent; ++i) {
        result.event(i) = event(result.event_order(i));
    }

    // Start times in start order
    for (int i = 0; i < nevent; ++i) {
        result.start(i) = start(result.start_order(i));
    }

    // Compute last indices
    std::vector<int> last_vec;
    int last_event_idx = nevent - 1;
    for (int i = 0; i < nevent; ++i) {
        int rev_i = nevent - 1 - i;
        int f = result.first(rev_i);
        last_vec.push_back(last_event_idx);
        if (f - rev_i == 0) {
            last_event_idx = f - 1;
        }
    }
    for (int i = 0; i < nevent; ++i) {
        result.last(i) = last_vec[nevent - 1 - i];
    }

    // Compute scaling factors
    for (int i = 0; i < nevent; ++i) {
        ValueType fi = static_cast<ValueType>(result.first(i));
        result.scaling(i) = (static_cast<ValueType>(i) - fi) /
                           (static_cast<ValueType>(result.last(i)) + 1.0 - fi);
    }

    // Store original scaling for restoration after zero-weight adjustments
    result.original_scaling = result.scaling;
}

/**
 * Preprocess stratified survival data.
 * Creates a StratifiedCoxData object with all per-stratum preprocessing done.
 * Empty strata vector signals single-stratum (unstratified) case.
 */
template <class ValueType, class IndexType>
inline void preprocess_stratified(
        const CoxSurvivalData<ValueType, IndexType>& surv,
        StratifiedCoxData<ValueType, IndexType>& strat_data) {

    using vec_t = Eigen::Matrix<ValueType, Eigen::Dynamic, 1>;
    using ivec_t = Eigen::Matrix<IndexType, Eigen::Dynamic, 1>;

    int n_total = surv.status.size();
    strat_data.n_total = n_total;

    // Handle empty strata as single-stratum (optimization: avoid O(2n) loops)
    if (surv.strata.size() == 0) {
        strat_data.n_strata = 1;
        strat_data.resize(1);
        strat_data.strata_labels[0] = 1;
        strat_data.stratum_indices[0].resize(n_total);
        std::iota(strat_data.stratum_indices[0].begin(),
                  strat_data.stratum_indices[0].end(), 0);
    } else {
        // Find unique strata
        std::set<int> unique_strata_set;
        for (int i = 0; i < n_total; ++i) {
            unique_strata_set.insert(surv.strata(i));
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
                if (surv.strata(i) == label) {
                    strat_data.stratum_indices[s].push_back(i);
                }
            }
        }
    }

    // Check for start times (not all zero)
    bool have_start_times = false;
    for (int i = 0; i < n_total; ++i) {
        if (surv.start(i) != 0) {
            have_start_times = true;
            break;
        }
    }

    // Preprocess each stratum
    for (int s = 0; s < strat_data.n_strata; ++s) {
        const std::vector<int>& idx = strat_data.stratum_indices[s];
        int n_s = static_cast<int>(idx.size());

        // Extract local data
        vec_t start_local(n_s);
        vec_t event_local(n_s);
        ivec_t status_local(n_s);

        for (int i = 0; i < n_s; ++i) {
            start_local(i) = surv.start(idx[i]);
            event_local(i) = surv.stop(idx[i]);
            status_local(i) = surv.status(idx[i]);
        }

        // Resize stratum data
        strat_data.resize_stratum(s, n_s);

        // Preprocess this stratum
        preprocess_single_stratum<ValueType, IndexType>(
            start_local, event_local, status_local,
            strat_data.preproc[s], surv.efron);
        strat_data.preproc[s].have_start_times = have_start_times;

        // Determine if Efron applies (only if there are ties)
        ValueType scaling_norm = strat_data.preproc[s].scaling.norm();
        strat_data.efron_stratum[s] = surv.efron && (scaling_norm > 0);

        // Initialize saturated log-likelihood to 0 (computed at call time with weights)
        strat_data.loglik_sat[s] = 0.0;
    }
}

/**
 * Compute Cox deviance for a single stratum.
 * Uses workspace buffers to avoid allocation.
 */
template <class ValueType, class IndexType, class EtaType, class WeightType>
inline ValueType cox_dev_single_stratum(
        const EtaType& eta_local,
        const WeightType& weight_local,
        CoxPreprocessed<ValueType, IndexType>& preproc,
        CoxWorkspace<ValueType>& ws,
        ValueType loglik_sat,
        bool efron_stratum) {

    using vec_t = Eigen::Matrix<ValueType, Eigen::Dynamic, 1>;

    int n = preproc.n;
    const auto& event_order = preproc.event_order;
    const auto& start_order = preproc.start_order;
    const auto& status = preproc.status;
    const auto& first = preproc.first;
    const auto& last = preproc.last;
    // Note: scaling is accessed via preproc.scaling directly (may be modified for zero weights)
    const auto& event_map = preproc.event_map;
    const auto& start_map = preproc.start_map;
    bool have_start_times = preproc.have_start_times;

    // Compute exp(eta) * weight
    ws.exp_w_buffer = (eta_local.array().exp() * weight_local.array()).matrix();

    // Reorder to event order
    to_event_from_native(eta_local, event_order, ws.eta_event);
    to_event_from_native(weight_local, event_order, ws.w_event);
    to_event_from_native(ws.exp_w_buffer, event_order, ws.exp_eta_w_event);

    // Check if there are any zero weights BEFORE computing risk sums
    // This is critical for Efron correction: zero-weight observations must not
    // affect the scaling factor used in the Efron correction
    bool has_zero_weights = (ws.w_event.array() == 0).any();

    // Reset scaling to original before potentially adjusting for zero weights
    // This ensures correct behavior across multiple calls with different weights
    preproc.scaling = preproc.original_scaling;

    if (has_zero_weights) {
        compute_effective_cluster_sizes(ws.w_event, first, last, ws.effective_cluster_sizes);
        compute_zero_weight_mask(ws.w_event, ws.zero_weight_mask);

        // For Efron correction, adjust the scaling to account for zero-weight observations
        // This ensures zero-weight obs don't affect the Efron tie correction
        if (efron_stratum) {
            compute_weighted_scaling(ws.w_event, first, last, preproc.scaling);
        }
    }

    // Compute risk sums (uses preproc.scaling which may have been adjusted for zero weights)
    sum_over_risk_set<ValueType>(ws.exp_w_buffer,
                                  event_order, start_order,
                                  first, last, event_map, preproc.scaling,
                                  efron_stratum, have_start_times,
                                  ws.risk_sums,
                                  ws.event_cumsum, ws.start_cumsum);

    // Compute w_avg for each tie group
    for (int i = 0; i < n; ++i) {
        ws.forward_scratch_buffer(i) = ws.w_event(i) * status(i);
    }
    forward_cumsum(ws.forward_scratch_buffer, ws.forward_cumsum_buffer0);

    if (has_zero_weights) {
        for (int i = 0; i < n; ++i) {
            if (ws.zero_weight_mask(i) == 0.0) {
                ws.w_avg_buffer(i) = 0.0;
                continue;
            }
            ValueType eff_size = ws.effective_cluster_sizes(i);
            if (eff_size > 0) {
                ws.w_avg_buffer(i) = (ws.forward_cumsum_buffer0(last(i) + 1) - ws.forward_cumsum_buffer0(first(i))) / eff_size;
            } else {
                ws.w_avg_buffer(i) = 0.0;
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            int cluster_size = last(i) - first(i) + 1;
            if (cluster_size > 0) {
                ws.w_avg_buffer(i) = (ws.forward_cumsum_buffer0(last(i) + 1) - ws.forward_cumsum_buffer0(first(i))) / cluster_size;
            } else {
                ws.w_avg_buffer(i) = 0.0;
            }
        }
    }

    // Compute log-likelihood using safe log to avoid log(0)
    ValueType loglik = 0.0;
    for (int i = 0; i < n; ++i) {
        if (status(i) == 1) {
            ValueType safe_risk_sum = std::max(ws.risk_sums(i), static_cast<ValueType>(1e-100));
            loglik += ws.w_event(i) * ws.eta_event(i) - ws.w_avg_buffer(i) * std::log(safe_risk_sum);
        }
    }

    // Forward cumsums for gradient and Hessian
    forward_prework(status, ws.w_avg_buffer, preproc.scaling, ws.risk_sums, 0, 1,
                   ws.forward_scratch_buffer, true);
    forward_cumsum(ws.forward_scratch_buffer, ws.forward_cumsum_buffer0);
    auto& C_01 = ws.forward_cumsum_buffer0;

    forward_prework(status, ws.w_avg_buffer, preproc.scaling, ws.risk_sums, 0, 2,
                   ws.forward_scratch_buffer, true);
    forward_cumsum(ws.forward_scratch_buffer, ws.forward_cumsum_buffer1);
    auto& C_02 = ws.forward_cumsum_buffer1;

    if (!efron_stratum) {
        if (have_start_times) {
            for (int i = 0; i < n; ++i) {
                ws.T_1_term(i) = C_01(last(i) + 1) - C_01(start_map(i));
                ws.T_2_term(i) = C_02(last(i) + 1) - C_02(start_map(i));
            }
        } else {
            for (int i = 0; i < n; ++i) {
                ws.T_1_term(i) = C_01(last(i) + 1);
                ws.T_2_term(i) = C_02(last(i) + 1);
            }
        }
    } else {
        forward_prework(status, ws.w_avg_buffer, preproc.scaling, ws.risk_sums, 1, 1,
                       ws.forward_scratch_buffer, true);
        forward_cumsum(ws.forward_scratch_buffer, ws.forward_cumsum_buffer2);
        auto& C_11 = ws.forward_cumsum_buffer2;

        // C_12 = status * w_avg * scaling^1 / risk_sums^2
        forward_prework(status, ws.w_avg_buffer, preproc.scaling, ws.risk_sums, 1, 2,
                       ws.forward_scratch_buffer, true);
        forward_cumsum(ws.forward_scratch_buffer, ws.forward_cumsum_buffer3);
        auto& C_12 = ws.forward_cumsum_buffer3;

        // C_22 = status * w_avg * scaling^2 / risk_sums^2
        forward_prework(status, ws.w_avg_buffer, preproc.scaling, ws.risk_sums, 2, 2,
                       ws.forward_scratch_buffer, true);
        forward_cumsum(ws.forward_scratch_buffer, ws.forward_cumsum_buffer4);
        auto& C_22 = ws.forward_cumsum_buffer4;

        for (int i = 0; i < n; ++i) {
            ws.T_1_term(i) = C_01(last(i) + 1) - (C_11(last(i) + 1) - C_11(first(i)));
            ws.T_2_term(i) = (C_22(last(i) + 1) - C_22(first(i)))
                            - 2.0 * (C_12(last(i) + 1) - C_12(first(i)))
                            + C_02(last(i) + 1);
        }
        if (have_start_times) {
            for (int i = 0; i < n; ++i) {
                ws.T_1_term(i) -= C_01(start_map(i));
            }
            for (int i = 0; i < n; ++i) {
                ws.T_2_term(i) -= C_02(first(i));
            }
        }
    }

    // Compute gradient and diagonal Hessian in event order
    ws.diag_part_buffer = ws.exp_eta_w_event.array() * ws.T_1_term.array();
    ws.grad_buffer = ws.w_event.array() * status.template cast<ValueType>().array() - ws.diag_part_buffer.array();
    ws.grad_buffer.array() *= -2.0;

    ws.diag_hessian_buffer = ws.exp_eta_w_event.array().pow(2) * ws.T_2_term.array()
                            - ws.diag_part_buffer.array();
    ws.diag_hessian_buffer.array() *= -2.0;

    // Reorder to native order
    to_native_from_event(ws.grad_buffer, event_order, ws.forward_scratch_buffer);
    to_native_from_event(ws.diag_hessian_buffer, event_order, ws.forward_scratch_buffer);
    to_native_from_event(ws.diag_part_buffer, event_order, ws.forward_scratch_buffer);

    // Note: scaling is NOT restored here. hessian_matvec (if called) will use the
    // zero-weight adjusted scaling, which is correct for the same weights.
    // On the next cox_dev call, scaling will be reset to original and re-adjusted if needed.

    return 2.0 * (loglik_sat - loglik);
}

/**
 * Compute Cox deviance for stratified data.
 * Returns total deviance and fills grad/diag_hess in strat_data workspaces.
 */
template <class ValueType, class IndexType, class EtaType, class WeightType>
inline ValueType cox_dev_stratified(
        const EtaType& eta,
        const WeightType& sample_weight,
        StratifiedCoxData<ValueType, IndexType>& strat_data,
        Eigen::Matrix<ValueType, Eigen::Dynamic, 1>& grad_output,
        Eigen::Matrix<ValueType, Eigen::Dynamic, 1>& diag_hess_output) {

    using vec_t = Eigen::Matrix<ValueType, Eigen::Dynamic, 1>;

    ValueType total_deviance = 0.0;

    // Zero output arrays
    grad_output.setZero();
    diag_hess_output.setZero();

    for (int s = 0; s < strat_data.n_strata; ++s) {
        const std::vector<int>& idx = strat_data.stratum_indices[s];
        int n_s = static_cast<int>(idx.size());
        auto& preproc = strat_data.preproc[s];
        auto& ws = strat_data.workspace[s];
        auto& eta_local = strat_data.eta_local_buffers[s];
        auto& weight_local = strat_data.weight_local_buffers[s];

        // Extract local eta and weights
        for (int i = 0; i < n_s; ++i) {
            eta_local(i) = eta(idx[i]);
            weight_local(i) = sample_weight(idx[i]);
        }

        // Center eta using weighted mean
        ValueType weight_sum = weight_local.sum();
        ValueType eta_mean;
        if (weight_sum > 0) {
            eta_mean = (eta_local.array() * weight_local.array()).sum() / weight_sum;
        } else {
            eta_mean = eta_local.mean();
        }
        eta_local.array() -= eta_mean;

        // Compute saturated log-likelihood for this stratum
        strat_data.loglik_sat[s] = compute_sat_loglik<ValueType, IndexType>(
            preproc.first, preproc.last, weight_local,
            preproc.event_order, preproc.status,
            ws.W_status_buffer,
            strat_data.efron_stratum[s]);

        // Compute deviance for this stratum
        ValueType dev = cox_dev_single_stratum<ValueType, IndexType>(
            eta_local, weight_local,
            preproc, ws,
            strat_data.loglik_sat[s],
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

} // namespace coxdev

#ifdef GLMNET_INTERFACE
} // namespace glmnetpp
#endif
