#ifndef COXDEV_STRATA_H
#define COXDEV_STRATA_H

#include <vector>
#include <set>
#include <Eigen/Dense>

// Include base coxdev.h for shared macros
#include "coxdev.h"

/**
 * Preprocessed data for a single stratum.
 * Contains all the ordering and mapping information computed once
 * during initialization.
 */
template <class ValueType = double, class IndexType = int>
struct CoxPreprocessed {
    Eigen::VectorXi event_order;    // permutation to sort by event time
    Eigen::VectorXi start_order;    // permutation to sort by start time
    Eigen::VectorXi status;         // event indicator in event order
    Eigen::VectorXi first;          // first index of tie group
    Eigen::VectorXi last;           // last index of tie group
    Eigen::VectorXd scaling;        // Efron scaling factors
    Eigen::VectorXd original_scaling; // original scaling (before zero-weight correction)
    Eigen::VectorXd event;          // event times in event order
    Eigen::VectorXd start;          // start times in start order
    Eigen::VectorXi event_map;      // maps event order to start order
    Eigen::VectorXi start_map;      // maps start order to event order
    bool have_start_times;          // whether left truncation is present
    bool efron;                     // whether to use Efron tie-breaking
    int n;                          // number of observations in this stratum

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
 */
template <class ValueType = double>
struct CoxWorkspace {
    // Gradient and Hessian outputs
    Eigen::VectorXd grad_buffer;
    Eigen::VectorXd diag_hessian_buffer;
    Eigen::VectorXd diag_part_buffer;

    // Intermediate computation buffers
    Eigen::VectorXd exp_w_buffer;       // weight * exp(eta)
    Eigen::VectorXd T_1_term;           // first moment terms
    Eigen::VectorXd T_2_term;           // second moment terms
    Eigen::VectorXd w_avg_buffer;       // average weights in tie groups
    Eigen::VectorXd forward_scratch_buffer;
    Eigen::VectorXd hess_matvec_buffer;

    // Risk sum buffers (length n)
    std::vector<Eigen::VectorXd> risk_sum_buffers;  // 2 buffers

    // Forward cumsum buffers (length n+1)
    std::vector<Eigen::VectorXd> forward_cumsum_buffers;  // 5 buffers

    // Reverse cumsum buffers (length n+1)
    std::vector<Eigen::VectorXd> reverse_cumsum_buffers;  // 4 buffers

    // Event reorder buffers (length n)
    std::vector<Eigen::VectorXd> event_reorder_buffers;  // 3 buffers

    // Zero-weight handling
    Eigen::VectorXd effective_cluster_sizes;
    Eigen::VectorXd zero_weight_mask;

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

#endif // COXDEV_STRATA_H
