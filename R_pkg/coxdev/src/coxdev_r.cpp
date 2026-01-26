/**
 * R bindings for coxdev using Rcpp.
 *
 * This file provides R interface to the standalone coxdev.hpp library.
 * Uses Eigen::Map for zero-copy array access.
 */

// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <RcppEigen.h>

#include "../inst/include/coxdev.hpp"

// Convenience type aliases
using value_t = double;
using index_t = int;
using vec_t = Eigen::VectorXd;
using ivec_t = Eigen::VectorXi;

// =============================================================================
// StratifiedCoxData Interface
// =============================================================================

/**
 * Wrapper struct that holds StratifiedCoxData and sample weights.
 * Weights are stored at initialization and used by all subsequent calls.
 */
struct StratifiedCoxWrapper {
    coxdev::StratifiedCoxData<value_t, index_t> strat_data;
    vec_t sample_weight;

    StratifiedCoxWrapper() = default;
};

// Custom destructor for cleanup
inline void stratified_cox_wrapper_finalizer(StratifiedCoxWrapper* ptr) {
    delete ptr;
}

// [[Rcpp::export(.preprocess_stratified)]]
SEXP preprocess_stratified_r(
    const Eigen::Map<Eigen::VectorXd> start,
    const Eigen::Map<Eigen::VectorXd> event,
    const Eigen::Map<Eigen::VectorXi> status,
    const Eigen::Map<Eigen::VectorXi> strata,
    const Eigen::Map<Eigen::VectorXd> sample_weight,
    bool efron)
{
    // Create the wrapper on the heap
    auto* wrapper_ptr = new StratifiedCoxWrapper();

    // Store sample weights
    wrapper_ptr->sample_weight = sample_weight;

    // Create CoxSurvivalData and preprocess
    if (strata.size() > 0) {
        coxdev::CoxSurvivalData<value_t, index_t> surv(start, event, status, strata, efron);
        coxdev::preprocess_stratified(surv, wrapper_ptr->strat_data);
    } else {
        coxdev::CoxSurvivalData<value_t, index_t> surv(start, event, status, efron);
        coxdev::preprocess_stratified(surv, wrapper_ptr->strat_data);
    }

    // Wrap in XPtr and return
    Rcpp::XPtr<StratifiedCoxWrapper> xptr(wrapper_ptr, true);
    return xptr;
}

// [[Rcpp::export(.cox_dev_stratified)]]
Rcpp::List cox_dev_stratified_r(
    SEXP wrapper_xptr,
    const Eigen::Map<Eigen::VectorXd> eta)
{
    // Extract the XPtr
    Rcpp::XPtr<StratifiedCoxWrapper> xptr(wrapper_xptr);
    auto& wrapper = *xptr;
    auto& strat_data = wrapper.strat_data;

    int n = eta.size();
    vec_t grad(n);
    vec_t diag_hess(n);

    // Compute deviance using stored weights
    value_t total_deviance = coxdev::cox_dev_stratified<value_t, index_t>(
        eta, wrapper.sample_weight, strat_data, grad, diag_hess);

    // Compute total loglik_sat
    value_t total_loglik_sat = 0.0;
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
    SEXP wrapper_xptr,
    const Eigen::Map<Eigen::VectorXd> arg)
{
    // Extract the XPtr
    // Note: Uses cached buffers from the last cox_dev_stratified_r call.
    // Caller must ensure cox_dev_stratified_r was called with the desired
    // linear_predictor before calling this function.
    Rcpp::XPtr<StratifiedCoxWrapper> xptr(wrapper_xptr);
    auto& strat_data = xptr->strat_data;

    int n = arg.size();
    vec_t result(n);
    result.setZero();

    // Compute hessian matvec for each stratum
    for (int s = 0; s < strat_data.n_strata; ++s) {
        const std::vector<int>& idx = strat_data.stratum_indices[s];
        int n_s = static_cast<int>(idx.size());
        auto& preproc = strat_data.preproc[s];
        auto& ws = strat_data.workspace[s];

        // Extract local arg
        vec_t arg_local(n_s);
        for (int i = 0; i < n_s; ++i) {
            arg_local(i) = arg(idx[i]);
        }

        // Compute matvec
        vec_t result_local(n_s);
        coxdev::hessian_matvec<value_t, index_t>(
            arg_local,
            strat_data.eta_local_buffers[s],
            strat_data.weight_local_buffers[s],
            preproc, ws,
            strat_data.efron_stratum[s],
            result_local);

        // Scatter back
        for (int i = 0; i < n_s; ++i) {
            result(idx[i]) = result_local(i);
        }
    }

    return result;
}

// [[Rcpp::export(.get_n_strata)]]
int get_n_strata_r(SEXP wrapper_xptr) {
    Rcpp::XPtr<StratifiedCoxWrapper> xptr(wrapper_xptr);
    return xptr->strat_data.n_strata;
}

// [[Rcpp::export(.get_n_total)]]
int get_n_total_r(SEXP wrapper_xptr) {
    Rcpp::XPtr<StratifiedCoxWrapper> xptr(wrapper_xptr);
    return xptr->strat_data.n_total;
}

// [[Rcpp::export(.get_sample_weight)]]
Eigen::VectorXd get_sample_weight_r(SEXP wrapper_xptr) {
    Rcpp::XPtr<StratifiedCoxWrapper> xptr(wrapper_xptr);
    return xptr->sample_weight;
}

// =============================================================================
// CoxIRLSState Interface
// =============================================================================

/**
 * R wrapper struct for IRLS state.
 * Holds working quantities for coordinate descent integration.
 * References the wrapper (non-owning) for access to strat_data and sample_weight.
 */
struct CoxIRLSStateR {
    StratifiedCoxWrapper* wrapper_ptr;  // Non-owning
    vec_t working_weights;
    vec_t working_response;
    vec_t residuals;
    vec_t grad_buffer;
    vec_t diag_hess_buffer;
    value_t deviance;

    CoxIRLSStateR(StratifiedCoxWrapper* ptr)
        : wrapper_ptr(ptr), deviance(0.0)
    {
        int n = ptr->strat_data.n_total;
        working_weights.resize(n);
        working_response.resize(n);
        residuals.resize(n);
        grad_buffer.resize(n);
        diag_hess_buffer.resize(n);
    }
};

inline void irls_state_finalizer(CoxIRLSStateR* ptr) {
    delete ptr;
}

// [[Rcpp::export(.create_irls_state)]]
SEXP create_irls_state_r(SEXP wrapper_xptr) {
    Rcpp::XPtr<StratifiedCoxWrapper> data_xptr(wrapper_xptr);

    auto* state_ptr = new CoxIRLSStateR(data_xptr.get());

    Rcpp::XPtr<CoxIRLSStateR> xptr(state_ptr, true);
    return xptr;
}

// [[Rcpp::export(.irls_recompute_outer)]]
double irls_recompute_outer_r(
    SEXP irls_state_xptr,
    const Eigen::Map<Eigen::VectorXd> eta)
{
    Rcpp::XPtr<CoxIRLSStateR> xptr(irls_state_xptr);
    auto& state = *xptr;
    auto& wrapper = *state.wrapper_ptr;
    auto& strat_data = wrapper.strat_data;
    int n = eta.size();

    // Compute deviance and gradients using stored weights
    state.deviance = coxdev::cox_dev_stratified<value_t, index_t>(
        eta, wrapper.sample_weight, strat_data, state.grad_buffer, state.diag_hess_buffer);

    // Extract working weights: w = -diag_hessian / 2 (make positive)
    state.working_weights = state.diag_hess_buffer / 2.0;

    // Compute working response: z = eta - grad/diag_hess
    for (int i = 0; i < n; ++i) {
        if (std::abs(state.diag_hess_buffer(i)) > 1e-10) {
            state.working_response(i) = eta(i) - state.grad_buffer(i) /
                                                  state.diag_hess_buffer(i);
        } else {
            state.working_response(i) = eta(i);
        }
    }

    // Initialize residuals: r = w * (z - eta)
    state.residuals = state.working_weights.array() * (state.working_response - eta).array();

    return state.deviance;
}

// [[Rcpp::export(.irls_working_weights)]]
Eigen::VectorXd irls_working_weights_r(SEXP irls_state_xptr) {
    Rcpp::XPtr<CoxIRLSStateR> xptr(irls_state_xptr);
    return xptr->working_weights;
}

// [[Rcpp::export(.irls_working_response)]]
Eigen::VectorXd irls_working_response_r(SEXP irls_state_xptr) {
    Rcpp::XPtr<CoxIRLSStateR> xptr(irls_state_xptr);
    return xptr->working_response;
}

// [[Rcpp::export(.irls_residuals)]]
Eigen::VectorXd irls_residuals_r(SEXP irls_state_xptr) {
    Rcpp::XPtr<CoxIRLSStateR> xptr(irls_state_xptr);
    return xptr->residuals;
}

// [[Rcpp::export(.irls_current_deviance)]]
double irls_current_deviance_r(SEXP irls_state_xptr) {
    Rcpp::XPtr<CoxIRLSStateR> xptr(irls_state_xptr);
    return xptr->deviance;
}

// [[Rcpp::export(.irls_weighted_inner_product)]]
Rcpp::NumericVector irls_weighted_inner_product_r(
    SEXP irls_state_xptr,
    const Eigen::Map<Eigen::VectorXd> x_j)
{
    Rcpp::XPtr<CoxIRLSStateR> xptr(irls_state_xptr);
    auto& state = *xptr;

    value_t grad_j = (x_j.array() * state.residuals.array()).sum();
    value_t hess_jj = (state.working_weights.array() * x_j.array().square()).sum();

    return Rcpp::NumericVector::create(
        Rcpp::_["gradient"] = grad_j,
        Rcpp::_["hessian"] = hess_jj
    );
}

// [[Rcpp::export(.irls_update_residuals)]]
void irls_update_residuals_r(
    SEXP irls_state_xptr,
    double delta,
    const Eigen::Map<Eigen::VectorXd> x_j)
{
    Rcpp::XPtr<CoxIRLSStateR> xptr(irls_state_xptr);
    xptr->residuals.array() -= delta * xptr->working_weights.array() * x_j.array();
}

// [[Rcpp::export(.irls_reset_residuals)]]
void irls_reset_residuals_r(
    SEXP irls_state_xptr,
    const Eigen::Map<Eigen::VectorXd> eta_current)
{
    Rcpp::XPtr<CoxIRLSStateR> xptr(irls_state_xptr);
    auto& state = *xptr;
    state.residuals = state.working_weights.array() * (state.working_response - eta_current).array();
}
