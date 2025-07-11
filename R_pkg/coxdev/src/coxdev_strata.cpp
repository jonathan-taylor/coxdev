
#ifdef PY_INTERFACE
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

#ifdef R_INTERFACE
#include <RcppEigen.h>
#endif

#include <vector>
#include <numeric>
#include <algorithm>
#include "coxdev.h"


// Use const references for read-only data and a mutable reference for the writeable buffer.
double compute_sat_loglik(const Eigen::VectorXi& first,
                         const Eigen::VectorXi& last,
                         const Eigen::VectorXd& weight,
                         const Eigen::VectorXi& event_order,
                         const Eigen::VectorXi& status,
                         Eigen::VectorXd& W_status); // Note: W_status is now mutable

// Use const references for all read-only data. Mutable buffers are already references.
double cox_dev(const Eigen::VectorXd& eta,
               const Eigen::VectorXd& sample_weight,
               const Eigen::VectorXd& exp_w,
               const Eigen::VectorXi& event_order,
               const Eigen::VectorXi& start_order,
               const Eigen::VectorXi& status,
               const Eigen::VectorXi& first,
               const Eigen::VectorXi& last,
               const Eigen::VectorXd& scaling,
               const Eigen::VectorXi& event_map,
               const Eigen::VectorXi& start_map,
               double loglik_sat,
               Eigen::VectorXd& T_1_term,
               Eigen::VectorXd& T_2_term,
               Eigen::VectorXd& grad_buffer,
               Eigen::VectorXd& diag_hessian_buffer,
               Eigen::VectorXd& diag_part_buffer,
               Eigen::VectorXd& w_avg_buffer,
               py::list& event_reorder_buffers,   // Using py::list& for Rcpp::List/py::list compatibility
               py::list& risk_sum_buffers,
               py::list& forward_cumsum_buffers,
               Eigen::VectorXd& forward_scratch_buffer,
               py::list& reverse_cumsum_buffers,
               bool have_start_times,
               bool efron);


// Assuming EIGEN_REF and other types are properly defined through coxdev.h

/**
 * @brief Wrapper function to call cox_dev for each stratum.
 * * @param linear_predictor The linear predictor vector for all samples.
 * @param sample_weight The sample weight vector for all samples.
 * @param stratum_indices A vector of vectors, where each inner vector contains the indices for a stratum.
 * @param _first A vector of 'first' vectors for each stratum.
 * @param _last A vector of 'last' vectors for each stratum.
 * @param _event_order A vector of 'event_order' vectors for each stratum.
 * @param _start_order A vector of 'start_order' vectors for each stratum.
 * @param _status_list A vector of 'status' vectors for each stratum.
 * @param _scaling A vector of 'scaling' vectors for each stratum.
 * @param _event_map A vector of 'event_map' vectors for each stratum.
 * @param _start_map A vector of 'start_map' vectors for each stratum.
 * @param _exp_w_buffer A buffer to store the exponentiated weighted eta for each stratum.
 * @param _T_1_term A buffer for the T_1_term for each stratum.
 * @param _T_2_term A buffer for the T_2_term for each stratum.
 * @param _grad_buffer A buffer to store the gradient for each stratum.
 * @param _diag_hessian_buffer A buffer to store the diagonal of the Hessian for each stratum.
 * @param _diag_part_buffer A buffer for the diag_part for each stratum.
 * @param _w_avg_buffer A buffer for the w_avg for each stratum.
 * @param _event_reorder_buffers A buffer for event reordering for each stratum.
 * @param _risk_sum_buffers A buffer for risk sums for each stratum.
 * @param _forward_cumsum_buffers A buffer for forward cumsums for each stratum.
 * @param _forward_scratch_buffer A scratch buffer for forward calculations for each stratum.
 * @param _reverse_cumsum_buffers A buffer for reverse cumsums for each stratum.
 * @param _have_start_times A boolean indicating if start times are present.
 * @param _efron_stratum A vector of booleans indicating if Efron's method should be used for each stratum.
 * @param grad The output gradient vector for all samples.
 * @param diag_hess The output diagonal Hessian vector for all samples.
 * * @return The total deviance.
 */
double cox_dev_wrapper(
    const EIGEN_REF<Eigen::VectorXd> linear_predictor,
    const EIGEN_REF<Eigen::VectorXd> sample_weight,
    const std::vector<Eigen::VectorXi>& stratum_indices,
    const std::vector<Eigen::VectorXi>& _first,
    const std::vector<Eigen::VectorXi>& _last,
    const std::vector<Eigen::VectorXi>& _event_order,
    const std::vector<Eigen::VectorXi>& _start_order,
    const std::vector<Eigen::VectorXi>& _status_list,
    const std::vector<Eigen::VectorXd>& _scaling,
    const std::vector<Eigen::VectorXi>& _event_map,
    const std::vector<Eigen::VectorXi>& _start_map,
    std::vector<Eigen::VectorXd>& _exp_w_buffer,
    std::vector<Eigen::VectorXd>& _T_1_term,
    std::vector<Eigen::VectorXd>& _T_2_term,
    std::vector<Eigen::VectorXd>& _grad_buffer,
    std::vector<Eigen::VectorXd>& _diag_hessian_buffer,
    std::vector<Eigen::VectorXd>& _diag_part_buffer,
    std::vector<Eigen::VectorXd>& _w_avg_buffer,
    std::vector<py::list>& _event_reorder_buffers,
    std::vector<py::list>& _risk_sum_buffers,
    std::vector<py::list>& _forward_cumsum_buffers,
    std::vector<Eigen::VectorXd>& _forward_scratch_buffer,
    std::vector<py::list>& _reverse_cumsum_buffers,
    bool _have_start_times,
    const std::vector<bool>& _efron_stratum,
    EIGEN_REF<Eigen::VectorXd> grad,
    EIGEN_REF<Eigen::VectorXd> diag_hess)
{
    double deviance = 0.0;
    double loglik_sat = 0.0;

    for (int i = 0; i < stratum_indices.size(); ++i) {
        const Eigen::VectorXi& idx = stratum_indices[i];
        
        // Extract eta and weight for the current stratum
        Eigen::VectorXd eta(idx.size());
        Eigen::VectorXd weight(idx.size());
        for (int j = 0; j < idx.size(); ++j) {
            eta(j) = linear_predictor(idx(j));
            weight(j) = sample_weight(idx(j));
        }

        // Center eta
        eta.array() -= eta.mean();

        // Compute exp_w_buffer
        _exp_w_buffer[i] = (weight.array() * eta.array().min(30).exp());

        // Compute saturated log-likelihood for the stratum
        double loglik_sat_i = compute_sat_loglik(
            _first[i], _last[i], weight, _event_order[i], _status_list[i], _exp_w_buffer[i]
        );
        loglik_sat += loglik_sat_i;

        // Call cox_dev for this stratum
        double dev = cox_dev(
            eta, weight, _exp_w_buffer[i],
            _event_order[i], _start_order[i], _status_list[i],
            _first[i], _last[i], _scaling[i],
            _event_map[i], _start_map[i],
            loglik_sat_i,
            _T_1_term[i], _T_2_term[i],
            _grad_buffer[i], _diag_hessian_buffer[i],
            _diag_part_buffer[i], _w_avg_buffer[i],
            _event_reorder_buffers[i], _risk_sum_buffers[i],
            _forward_cumsum_buffers[i], _forward_scratch_buffer[i],
            _reverse_cumsum_buffers[i], _have_start_times, _efron_stratum[i]
        );

        deviance += dev;

        // Update grad and diag_hess from buffers
        for (int j = 0; j < idx.size(); ++j) {
            grad(idx(j)) = _grad_buffer[i](j);
            diag_hess(idx(j)) = _diag_hessian_buffer[i](j);
        }
    }

    return deviance;
}

// R export
#ifdef R_INTERFACE
// [[Rcpp::export(.cox_dev_stratified_wrapper)]]
double cox_dev_stratified_wrapper(
    const EIGEN_REF<Eigen::VectorXd> linear_predictor,
    const EIGEN_REF<Eigen::VectorXd> sample_weight,
    const std::vector<Eigen::VectorXi>& stratum_indices,
    const std::vector<Eigen::VectorXi>& _first,
    const std::vector<Eigen::VectorXi>& _last,
    const std::vector<Eigen::VectorXi>& _event_order,
    const std::vector<Eigen::VectorXi>& _start_order,
    const std::vector<Eigen::VectorXi>& _status_list,
    const std::vector<Eigen::VectorXd>& _scaling,
    const std::vector<Eigen::VectorXi>& _event_map,
    const std::vector<Eigen::VectorXi>& _start_map,
    std::vector<Eigen::VectorXd>& _exp_w_buffer,
    std::vector<Eigen::VectorXd>& _T_1_term,
    std::vector<Eigen::VectorXd>& _T_2_term,
    std::vector<Eigen::VectorXd>& _grad_buffer,
    std::vector<Eigen::VectorXd>& _diag_hessian_buffer,
    std::vector<Eigen::VectorXd>& _diag_part_buffer,
    std::vector<Eigen::VectorXd>& _w_avg_buffer,
    std::vector<Rcpp::List>& _event_reorder_buffers,
    std::vector<Rcpp::List>& _risk_sum_buffers,
    std::vector<Rcpp::List>& _forward_cumsum_buffers,
    std::vector<Eigen::VectorXd>& _forward_scratch_buffer,
    std::vector<Rcpp::List>& _reverse_cumsum_buffers,
    bool _have_start_times,
    const std::vector<bool>& _efron_stratum,
    EIGEN_REF<Eigen::VectorXd> grad,
    EIGEN_REF<Eigen::VectorXd> diag_hess) {
    
    return cox_dev_wrapper(
        linear_predictor, sample_weight, stratum_indices,
        _first, _last, _event_order, _start_order, _status_list, _scaling,
        _event_map, _start_map, _exp_w_buffer, _T_1_term, _T_2_term,
        _grad_buffer, _diag_hessian_buffer, _diag_part_buffer, _w_avg_buffer,
        _event_reorder_buffers, _risk_sum_buffers, _forward_cumsum_buffers,
        _forward_scratch_buffer, _reverse_cumsum_buffers,
        _have_start_times, _efron_stratum, grad, diag_hess
    );
}
#endif

// Python bindings
#ifdef PY_INTERFACE
// Note: This module will be built into coxc, not as a separate module
// The function will be available as coxc.cox_dev_stratified_wrapper
#endif
