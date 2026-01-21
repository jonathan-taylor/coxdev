#ifndef COXDEV_H
#define COXDEV_H

// =============================================================================
// Interface-specific includes and macros
// =============================================================================

#ifdef DEBUG
#include <iostream>
#endif

#define MAKE_MAP_Xd(y) Eigen::Map<Eigen::VectorXd>((y).data(), (y).size())
#define MAKE_MAP_Xi(y) Eigen::Map<Eigen::VectorXi>((y).data(), (y).size())

#ifdef PY_INTERFACE

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;
#define EIGEN_REF Eigen::Ref
#define ERROR_MSG(x) throw std::runtime_error(x)

#endif // PY_INTERFACE

#ifdef R_INTERFACE

#include <RcppEigen.h>

using namespace Rcpp;
#define EIGEN_REF Eigen::Map
#define ERROR_MSG(x) Rcpp::stop(x)

#endif // R_INTERFACE

// =============================================================================
// Common includes
// =============================================================================

#include <vector>
#include <set>
#ifndef PY_INTERFACE
#include <Eigen/Dense>
#endif

/**
 * Preprocessed data for a single stratum.
 * Contains all the ordering and mapping information computed once
 * during initialization.
 *
 * All fields are RO (read-only) after preprocessing - do not modify during computation.
 */
template <class ValueType = double, class IndexType = int>
struct CoxPreprocessed {
    // RO: All fields set once during preprocessing, then read-only
    Eigen::VectorXi event_order;      // RO: permutation to sort by event time
    Eigen::VectorXi start_order;      // RO: permutation to sort by start time
    Eigen::VectorXi status;           // RO: event indicator in event order
    Eigen::VectorXi first;            // RO: first index of tie group
    Eigen::VectorXi last;             // RO: last index of tie group
    Eigen::VectorXd scaling;          // RO: Efron scaling factors (may be adjusted for zero weights)
    Eigen::VectorXd original_scaling; // RO: original scaling (before zero-weight correction)
    Eigen::VectorXd event;            // RO: event times in event order
    Eigen::VectorXd start;            // RO: start times in start order
    Eigen::VectorXi event_map;        // RO: maps event order to start order
    Eigen::VectorXi start_map;        // RO: maps start order to event order
    bool have_start_times;            // RO: whether left truncation is present
    bool efron;                       // RO: whether to use Efron tie-breaking
    int n;                            // RO: number of observations in this stratum

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
 * Working buffers for Cox deviance computation within a single stratum.
 * Pre-allocated to avoid repeated memory allocation during optimization.
 *
 * Buffer access patterns:
 *   RW: Read-write output buffers - primary outputs scattered to caller
 *   RW-PERSIST: Intermediate values preserved across deviance/hessian_matvec calls
 *   SCRATCH: Temporary storage, contents not preserved between calls
 */
template <class ValueType = double>
struct CoxWorkspace {
    // =========================================================================
    // RW: Output buffers - primary outputs scattered to caller
    // =========================================================================
    Eigen::VectorXd grad_buffer;           // RW: gradient output (native order)
    Eigen::VectorXd diag_hessian_buffer;   // RW: diagonal Hessian output (native order)
    Eigen::VectorXd matvec_result_buffer;  // RW: hessian_matvec result (native order)

    // =========================================================================
    // RW-PERSIST: Intermediate values preserved across deviance/hessian_matvec
    // These are computed in cox_dev and reused in hessian_matvec for same eta/weight
    // =========================================================================
    Eigen::VectorXd diag_part_buffer;      // RW-PERSIST: exp_eta_w * T_1_term (native order)
    Eigen::VectorXd exp_w_buffer;          // RW-PERSIST: weight * exp(eta) (native order)
    Eigen::VectorXd w_avg_buffer;          // RW-PERSIST: weighted averages in tie groups (event order)
    Eigen::VectorXd eta_local_buffer;      // RW-PERSIST: centered eta for stratum (native order)
    Eigen::VectorXd weight_local_buffer;   // RW-PERSIST: weights for stratum (native order)

    // risk_sum_buffers[0]: RW-PERSIST - risk set sums, reused in hessian_matvec
    // risk_sum_buffers[1]: SCRATCH - arg-weighted risk sums (hessian_matvec only)
    std::vector<Eigen::VectorXd> risk_sum_buffers;  // 2 buffers, length n

    // =========================================================================
    // RW: Zero-weight handling buffers (computed per-call when zero weights exist)
    // =========================================================================
    Eigen::VectorXd effective_cluster_sizes;  // RW: non-zero weight counts per cluster
    Eigen::VectorXd zero_weight_mask;         // RW: 1.0 for positive weights, 0.0 otherwise
    bool use_zero_weight_handling;            // RW: flag for conditional code paths

    // =========================================================================
    // SCRATCH: Temporary buffers - contents not preserved between calls
    // =========================================================================
    Eigen::VectorXd T_1_term;              // SCRATCH: first moment terms (event order)
    Eigen::VectorXd T_2_term;              // SCRATCH: second moment terms (event order)
    Eigen::VectorXd forward_scratch_buffer; // SCRATCH: temporary for cumsum steps
    Eigen::VectorXd hess_matvec_buffer;    // SCRATCH: intermediate hessian_matvec result

    // SCRATCH: Cumsum buffers - reused with different semantics across calls
    std::vector<Eigen::VectorXd> forward_cumsum_buffers;  // 5 buffers, length n+1
    std::vector<Eigen::VectorXd> reverse_cumsum_buffers;  // 4 buffers, length n+1

    // SCRATCH: Event reorder buffers - coordinate transformations
    std::vector<Eigen::VectorXd> event_reorder_buffers;   // 3 buffers, length n

    void resize(int n) {
        grad_buffer.resize(n);
        diag_hessian_buffer.resize(n);
        diag_part_buffer.resize(n);
        exp_w_buffer.resize(n);
        T_1_term.resize(n);
        T_2_term.resize(n);
        w_avg_buffer.resize(n);
        forward_scratch_buffer.resize(n);
        hess_matvec_buffer.resize(n);
        effective_cluster_sizes.resize(n);
        zero_weight_mask.resize(n);
        use_zero_weight_handling = false;  // initialize flag
        eta_local_buffer.resize(n);
        weight_local_buffer.resize(n);
        matvec_result_buffer.resize(n);

        // Risk sum buffers
        risk_sum_buffers.resize(2);
        for (auto& buf : risk_sum_buffers) {
            buf.resize(n);
        }

        // Forward cumsum buffers (n+1)
        forward_cumsum_buffers.resize(5);
        for (auto& buf : forward_cumsum_buffers) {
            buf.resize(n + 1);
        }

        // Reverse cumsum buffers (n+1)
        reverse_cumsum_buffers.resize(4);
        for (auto& buf : reverse_cumsum_buffers) {
            buf.resize(n + 1);
        }

        // Event reorder buffers
        event_reorder_buffers.resize(3);
        for (auto& buf : event_reorder_buffers) {
            buf.resize(n);
        }
    }
};

/**
 * Container for stratified Cox model data.
 * Holds preprocessed data and workspace for all strata,
 * plus global-to-local index mappings.
 */
template <class ValueType = double, class IndexType = int>
struct StratifiedCoxData {
    int n_strata;                                   // number of strata
    int n_total;                                    // total number of observations
    std::vector<int> strata_labels;                 // unique strata values
    std::vector<std::vector<int>> stratum_indices;  // global indices per stratum
    std::vector<CoxPreprocessed<ValueType, IndexType>> preproc;  // per-stratum preprocessing
    std::vector<CoxWorkspace<ValueType>> workspace;              // per-stratum workspace
    std::vector<double> loglik_sat;                 // per-stratum saturated log-likelihood
    std::vector<bool> efron_stratum;                // per-stratum efron flag (may differ if no ties)

    void resize(int n_strata_) {
        n_strata = n_strata_;
        strata_labels.resize(n_strata);
        stratum_indices.resize(n_strata);
        preproc.resize(n_strata);
        workspace.resize(n_strata);
        loglik_sat.resize(n_strata);
        efron_stratum.resize(n_strata);
    }

    // Resize a specific stratum
    void resize_stratum(int s, int n_s) {
        preproc[s].resize(n_s);
        workspace[s].resize(n_s);
    }
};

// Interface-specific type definitions
#ifdef PY_INTERFACE
#define STRAT_DATA_TYPE StratifiedCoxData<double, int>
#define STRAT_DATA_REF StratifiedCoxData<double, int>&
#endif

#ifdef R_INTERFACE
#define STRAT_DATA_TYPE StratifiedCoxData<double, int>
#define STRAT_DATA_REF Rcpp::XPtr<StratifiedCoxData<double, int>>
#endif

// Function declarations

/**
 * Preprocess stratified survival data.
 * Creates a StratifiedCoxData object with all per-stratum preprocessing done.
 */
STRAT_DATA_TYPE preprocess_stratified(
    const EIGEN_REF<Eigen::VectorXd> start,
    const EIGEN_REF<Eigen::VectorXd> event,
    const EIGEN_REF<Eigen::VectorXi> status,
    const EIGEN_REF<Eigen::VectorXi> strata,
    bool efron
);

/**
 * Compute Cox deviance for stratified data.
 * Returns total deviance summed across all strata.
 * Fills grad_output and diag_hess_output with global-order results.
 */
double cox_dev_stratified(
    const EIGEN_REF<Eigen::VectorXd> eta,            // global native order
    const EIGEN_REF<Eigen::VectorXd> sample_weight,  // global native order
    STRAT_DATA_REF strat_data,
    EIGEN_REF<Eigen::VectorXd> grad_output,          // global native order
    EIGEN_REF<Eigen::VectorXd> diag_hess_output      // global native order
);

/**
 * Compute Hessian matrix-vector product for stratified data.
 * Fills result with the global-order product.
 */
void hessian_matvec_stratified(
    const EIGEN_REF<Eigen::VectorXd> arg,            // global native order
    const EIGEN_REF<Eigen::VectorXd> eta,            // global native order
    const EIGEN_REF<Eigen::VectorXd> sample_weight,  // global native order
    STRAT_DATA_REF strat_data,
    EIGEN_REF<Eigen::VectorXd> result                // global native order
);

#endif // COXDEV_H
