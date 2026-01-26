/**
 * Python bindings for coxdev using pybind11.
 *
 * This file provides Python interface to the standalone coxdev.hpp library.
 * Uses Eigen::Ref for zero-copy array access.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "coxdev.hpp"

namespace py = pybind11;

// Convenience type aliases
using value_t = double;
using index_t = int;
using vec_t = Eigen::VectorXd;
using ivec_t = Eigen::VectorXi;

/**
 * Python wrapper class for stratified Cox deviance.
 * Owns a StratifiedCoxData object with pre-allocated workspaces.
 *
 * All data (event, status, strata, start, sample_weight) are specified at
 * construction time. The __call__ method only takes the linear predictor (eta).
 */
class StratifiedCoxDevianceCpp {
public:
    coxdev::StratifiedCoxData<value_t, index_t> strat_data;
    bool have_start_times;

    // Stored sample weights (set at construction)
    vec_t sample_weight_;

    // Pre-allocated output buffers
    vec_t grad_buffer_;
    vec_t diag_hess_buffer_;

    StratifiedCoxDevianceCpp(
        py::array_t<value_t, py::array::c_style | py::array::forcecast> event,
        py::array_t<index_t, py::array::c_style | py::array::forcecast> status,
        py::array_t<index_t, py::array::c_style | py::array::forcecast> strata,
        py::array_t<value_t, py::array::c_style | py::array::forcecast> start,
        py::array_t<value_t, py::array::c_style | py::array::forcecast> sample_weight,
        bool efron)
    {
        auto event_buf = event.request();
        auto status_buf = status.request();
        auto strata_buf = strata.request();
        auto start_buf = start.request();
        auto weight_buf = sample_weight.request();

        int n = static_cast<int>(event_buf.size);
        int strata_size = static_cast<int>(strata_buf.size);

        // Map numpy arrays to Eigen vectors (zero-copy)
        Eigen::Map<const vec_t> event_vec(static_cast<const value_t*>(event_buf.ptr), n);
        Eigen::Map<const ivec_t> status_vec(static_cast<const index_t*>(status_buf.ptr), n);
        Eigen::Map<const ivec_t> strata_vec(static_cast<const index_t*>(strata_buf.ptr), strata_size);
        Eigen::Map<const vec_t> start_vec(static_cast<const value_t*>(start_buf.ptr), n);
        Eigen::Map<const vec_t> weight_vec(static_cast<const value_t*>(weight_buf.ptr), n);

        // Store sample weights
        sample_weight_ = weight_vec;

        // Check for start times
        have_start_times = false;
        for (int i = 0; i < n; ++i) {
            if (start_vec(i) > -1e30 && start_vec(i) != 0.0) {
                have_start_times = true;
                break;
            }
        }

        // Create CoxSurvivalData and preprocess
        if (strata_size > 0) {
            coxdev::CoxSurvivalData<value_t, index_t> surv(start_vec, event_vec, status_vec, strata_vec, efron);
            coxdev::preprocess_stratified(surv, strat_data);
        } else {
            coxdev::CoxSurvivalData<value_t, index_t> surv(start_vec, event_vec, status_vec, efron);
            coxdev::preprocess_stratified(surv, strat_data);
        }

        // Allocate output buffers
        grad_buffer_.resize(strat_data.n_total);
        diag_hess_buffer_.resize(strat_data.n_total);
    }

    /**
     * Compute Cox deviance, gradient, and diagonal Hessian.
     *
     * @param linear_predictor The linear predictor (eta = X @ beta)
     * @return Tuple of (deviance, loglik_sat, gradient, diag_hessian)
     */
    py::tuple call(
        py::array_t<value_t, py::array::c_style | py::array::forcecast> linear_predictor)
    {
        auto eta_buf = linear_predictor.request();
        int n = static_cast<int>(eta_buf.size);

        // Map numpy array to Eigen (zero-copy)
        Eigen::Map<const vec_t> eta_vec(static_cast<const value_t*>(eta_buf.ptr), n);

        // Compute deviance using stored weights
        value_t deviance = coxdev::cox_dev_stratified<value_t, index_t>(
            eta_vec, sample_weight_, strat_data, grad_buffer_, diag_hess_buffer_);

        // Compute total loglik_sat
        value_t total_loglik_sat = 0.0;
        for (int s = 0; s < strat_data.n_strata; ++s) {
            total_loglik_sat += strat_data.loglik_sat[s];
        }

        // Return as numpy arrays (copy to Python)
        py::array_t<value_t> grad_out(n);
        py::array_t<value_t> diag_hess_out(n);

        auto grad_ptr = grad_out.mutable_data();
        auto diag_hess_ptr = diag_hess_out.mutable_data();

        for (int i = 0; i < n; ++i) {
            grad_ptr[i] = grad_buffer_(i);
            diag_hess_ptr[i] = diag_hess_buffer_(i);
        }

        return py::make_tuple(deviance, total_loglik_sat, grad_out, diag_hess_out);
    }

    /**
     * Get stored sample weights.
     */
    py::array_t<value_t> get_sample_weight() {
        int n = sample_weight_.size();
        py::array_t<value_t> result(n);
        auto ptr = result.mutable_data();
        for (int i = 0; i < n; ++i) {
            ptr[i] = sample_weight_(i);
        }
        return result;
    }

    /**
     * Compute Hessian matrix-vector product.
     *
     * IMPORTANT: Uses cached eta/weight/scaling buffers populated by the most
     * recent call to this object's __call__ method. The __call__ must be invoked
     * first to set up the buffers before calling hessian_matvec.
     *
     * @param arg Vector to multiply by the Hessian
     * @return Hessian-vector product
     */
    py::array_t<value_t> hessian_matvec(
        py::array_t<value_t, py::array::c_style | py::array::forcecast> arg)
    {
        auto arg_buf = arg.request();
        int n = static_cast<int>(arg_buf.size);

        Eigen::Map<const vec_t> arg_vec(static_cast<const value_t*>(arg_buf.ptr), n);

        vec_t result(n);

        // Compute hessian matvec for each stratum
        result.setZero();
        for (int s = 0; s < strat_data.n_strata; ++s) {
            const std::vector<int>& idx = strat_data.stratum_indices[s];
            int n_s = static_cast<int>(idx.size());
            auto& preproc = strat_data.preproc[s];
            auto& ws = strat_data.workspace[s];

            // Extract local arg
            vec_t arg_local(n_s);
            for (int i = 0; i < n_s; ++i) {
                arg_local(i) = arg_vec(idx[i]);
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

        // Return as numpy array
        py::array_t<value_t> result_out(n);
        auto result_ptr = result_out.mutable_data();
        for (int i = 0; i < n; ++i) {
            result_ptr[i] = result(i);
        }

        return result_out;
    }

    int n_strata() const { return strat_data.n_strata; }
    int n_total() const { return strat_data.n_total; }
};

/**
 * Python wrapper for IRLS state (coordinate descent integration).
 */
class CoxIRLSStateCpp {
public:
    StratifiedCoxDevianceCpp* cox_dev_ptr;  // Non-owning pointer
    vec_t working_weights_;
    vec_t working_response_;
    vec_t residuals_;
    value_t deviance_;

    CoxIRLSStateCpp(StratifiedCoxDevianceCpp& cox_dev) : cox_dev_ptr(&cox_dev) {
        int n = cox_dev.strat_data.n_total;
        working_weights_.resize(n);
        working_response_.resize(n);
        residuals_.resize(n);
        deviance_ = 0.0;
    }

    /**
     * Recompute all cached quantities for a new outer IRLS iteration.
     *
     * Uses sample weights stored in the parent CoxDeviance object.
     *
     * @param eta The current linear predictor
     * @return The deviance at this eta
     */
    value_t recompute_outer(
        py::array_t<value_t, py::array::c_style | py::array::forcecast> eta)
    {
        auto eta_buf = eta.request();
        int n = static_cast<int>(eta_buf.size);

        Eigen::Map<const vec_t> eta_vec(static_cast<const value_t*>(eta_buf.ptr), n);

        // Compute deviance using stored weights from parent cox_dev object
        deviance_ = coxdev::cox_dev_stratified<value_t, index_t>(
            eta_vec, cox_dev_ptr->sample_weight_, cox_dev_ptr->strat_data,
            cox_dev_ptr->grad_buffer_, cox_dev_ptr->diag_hess_buffer_);

        // Extract working weights: w = -diag_hessian / 2 (make positive)
        // diag_hessian_buffer = -2 * d²(loglik)/d(eta)²
        working_weights_ = cox_dev_ptr->diag_hess_buffer_ / 2.0;

        // Compute working response: z = eta - grad/diag_hess
        for (int i = 0; i < n; ++i) {
            if (std::abs(cox_dev_ptr->diag_hess_buffer_(i)) > 1e-10) {
                working_response_(i) = eta_vec(i) - cox_dev_ptr->grad_buffer_(i) /
                                                     cox_dev_ptr->diag_hess_buffer_(i);
            } else {
                working_response_(i) = eta_vec(i);
            }
        }

        // Initialize residuals: r = w * (z - eta)
        residuals_ = working_weights_.array() * (working_response_ - eta_vec).array();

        return deviance_;
    }

    py::array_t<value_t> working_weights() {
        int n = working_weights_.size();
        py::array_t<value_t> result(n);
        auto ptr = result.mutable_data();
        for (int i = 0; i < n; ++i) {
            ptr[i] = working_weights_(i);
        }
        return result;
    }

    py::array_t<value_t> working_response() {
        int n = working_response_.size();
        py::array_t<value_t> result(n);
        auto ptr = result.mutable_data();
        for (int i = 0; i < n; ++i) {
            ptr[i] = working_response_(i);
        }
        return result;
    }

    py::array_t<value_t> residuals() {
        int n = residuals_.size();
        py::array_t<value_t> result(n);
        auto ptr = result.mutable_data();
        for (int i = 0; i < n; ++i) {
            ptr[i] = residuals_(i);
        }
        return result;
    }

    value_t current_deviance() {
        return deviance_;
    }

    py::tuple weighted_inner_product(
        py::array_t<value_t, py::array::c_style | py::array::forcecast> x_j)
    {
        auto x_buf = x_j.request();
        int n = static_cast<int>(x_buf.size);
        Eigen::Map<const vec_t> x_vec(static_cast<const value_t*>(x_buf.ptr), n);

        value_t grad_j = (x_vec.array() * residuals_.array()).sum();
        value_t hess_jj = (working_weights_.array() * x_vec.array().square()).sum();

        return py::make_tuple(grad_j, hess_jj);
    }

    void update_residuals(
        value_t delta,
        py::array_t<value_t, py::array::c_style | py::array::forcecast> x_j)
    {
        auto x_buf = x_j.request();
        int n = static_cast<int>(x_buf.size);
        Eigen::Map<const vec_t> x_vec(static_cast<const value_t*>(x_buf.ptr), n);

        residuals_.array() -= delta * working_weights_.array() * x_vec.array();
    }

    void reset_residuals(
        py::array_t<value_t, py::array::c_style | py::array::forcecast> eta_current)
    {
        auto eta_buf = eta_current.request();
        int n = static_cast<int>(eta_buf.size);
        Eigen::Map<const vec_t> eta_vec(static_cast<const value_t*>(eta_buf.ptr), n);

        residuals_ = working_weights_.array() * (working_response_ - eta_vec).array();
    }
};

/**
 * Preprocessing function for Python (returns dict with arrays).
 */
py::tuple c_preprocess(
    py::array_t<value_t, py::array::c_style | py::array::forcecast> start_arr,
    py::array_t<value_t, py::array::c_style | py::array::forcecast> event_arr,
    py::array_t<index_t, py::array::c_style | py::array::forcecast> status_arr)
{
    auto start_buf = start_arr.request();
    auto event_buf = event_arr.request();
    auto status_buf = status_arr.request();

    int nevent = static_cast<int>(status_buf.size);
    Eigen::Map<const vec_t> start(static_cast<const value_t*>(start_buf.ptr), nevent);
    Eigen::Map<const vec_t> event(static_cast<const value_t*>(event_buf.ptr), nevent);
    Eigen::Map<const ivec_t> status(static_cast<const index_t*>(status_buf.ptr), nevent);

    // Use coxdev preprocessing
    auto preproc = coxdev::preprocess<value_t, index_t>(start, event, status);

    // Create output arrays
    py::array_t<index_t> event_order_out(nevent);
    py::array_t<index_t> start_order_out(nevent);
    auto eo_ptr = event_order_out.mutable_data();
    auto so_ptr = start_order_out.mutable_data();
    for (int i = 0; i < nevent; ++i) {
        eo_ptr[i] = preproc.event_order(i);
        so_ptr[i] = preproc.start_order(i);
    }

    // Create dict with preprocessing results
    py::dict result;

    py::array_t<index_t> first_out(nevent), last_out(nevent);
    py::array_t<index_t> event_map_out(nevent), start_map_out(nevent);
    py::array_t<value_t> scaling_out(nevent);
    py::array_t<index_t> status_out(nevent);
    py::array_t<value_t> event_out(nevent), start_out(nevent);

    auto first_ptr = first_out.mutable_data();
    auto last_ptr = last_out.mutable_data();
    auto event_map_ptr = event_map_out.mutable_data();
    auto start_map_ptr = start_map_out.mutable_data();
    auto scaling_ptr = scaling_out.mutable_data();
    auto status_ptr_out = status_out.mutable_data();
    auto event_ptr_out = event_out.mutable_data();
    auto start_ptr_out = start_out.mutable_data();

    for (int i = 0; i < nevent; ++i) {
        first_ptr[i] = preproc.first(i);
        last_ptr[i] = preproc.last(i);
        event_map_ptr[i] = preproc.event_map(i);
        start_map_ptr[i] = preproc.start_map(i);
        scaling_ptr[i] = preproc.scaling(i);
        status_ptr_out[i] = preproc.status(i);
        event_ptr_out[i] = preproc.event(i);
        start_ptr_out[i] = preproc.start(i);
    }

    result["first"] = first_out;
    result["last"] = last_out;
    result["event_map"] = event_map_out;
    result["start_map"] = start_map_out;
    result["scaling"] = scaling_out;
    result["status"] = status_out;
    result["event"] = event_out;
    result["start"] = start_out;

    return py::make_tuple(result, event_order_out, start_order_out);
}

// Module initialization
PYBIND11_MODULE(coxc, m) {
    m.doc() = "Cox deviance implementations (unified stratified)";

    // StratifiedCoxDevianceCpp class
    py::class_<StratifiedCoxDevianceCpp>(m, "StratifiedCoxDevianceCpp")
        .def(py::init<py::array_t<value_t>, py::array_t<index_t>, py::array_t<index_t>,
                      py::array_t<value_t>, py::array_t<value_t>, bool>(),
             py::arg("event"), py::arg("status"), py::arg("strata"),
             py::arg("start"), py::arg("sample_weight"), py::arg("efron") = true,
             "Create Cox deviance calculator with all data specified at construction")
        .def("__call__", &StratifiedCoxDevianceCpp::call,
             py::arg("linear_predictor"),
             "Compute deviance, gradient, and diagonal Hessian for given linear predictor")
        .def("hessian_matvec", &StratifiedCoxDevianceCpp::hessian_matvec,
             py::arg("arg"),
             "Compute Hessian-vector product using cached values from last __call__")
        .def_property_readonly("n_strata", &StratifiedCoxDevianceCpp::n_strata)
        .def_property_readonly("n_total", &StratifiedCoxDevianceCpp::n_total)
        .def_property_readonly("sample_weight", &StratifiedCoxDevianceCpp::get_sample_weight);

    // CoxIRLSStateCpp class for efficient IRLS/coordinate descent
    py::class_<CoxIRLSStateCpp>(m, "CoxIRLSStateCpp")
        .def(py::init<StratifiedCoxDevianceCpp&>(), py::arg("cox_dev"))
        .def("recompute_outer", &CoxIRLSStateCpp::recompute_outer,
             py::arg("eta"),
             "Recompute all cached quantities using stored weights (call once per outer IRLS iteration)")
        .def("working_weights", &CoxIRLSStateCpp::working_weights,
             "Get cached working weights")
        .def("working_response", &CoxIRLSStateCpp::working_response,
             "Get cached working response")
        .def("residuals", &CoxIRLSStateCpp::residuals,
             "Get cached residuals r = w * (z - eta)")
        .def("current_deviance", &CoxIRLSStateCpp::current_deviance,
             "Get deviance from last recompute_outer")
        .def("weighted_inner_product", &CoxIRLSStateCpp::weighted_inner_product,
             py::arg("x_j"),
             "Returns (gradient_j, hessian_jj) using cached residuals")
        .def("update_residuals", &CoxIRLSStateCpp::update_residuals,
             py::arg("delta"), py::arg("x_j"),
             "Update residuals: r -= delta * w * x_j")
        .def("reset_residuals", &CoxIRLSStateCpp::reset_residuals,
             py::arg("eta_current"),
             "Reset residuals for new CD pass");

    // Preprocessing function
    m.def("c_preprocess", &c_preprocess, "Preprocess survival data");

    // Reverse cumsums helper (for testing)
    m.def("reverse_cumsums",
        [](py::array_t<value_t, py::array::c_style | py::array::forcecast> sequence,
           py::array_t<value_t, py::array::c_style | py::array::forcecast> event_buffer,
           py::array_t<value_t, py::array::c_style | py::array::forcecast> start_buffer,
           py::array_t<index_t, py::array::c_style | py::array::forcecast> event_order,
           py::array_t<index_t, py::array::c_style | py::array::forcecast> start_order,
           bool do_event,
           bool do_start) {
            auto seq_buf = sequence.request();
            auto event_buf = event_buffer.request();
            auto start_buf = start_buffer.request();
            auto eo_buf = event_order.request();
            auto so_buf = start_order.request();

            int n = static_cast<int>(seq_buf.size);

            Eigen::Map<const vec_t> seq_vec(static_cast<const value_t*>(seq_buf.ptr), n);
            Eigen::Map<vec_t> event_vec(static_cast<value_t*>(event_buf.ptr), n + 1);
            Eigen::Map<vec_t> start_vec(static_cast<value_t*>(start_buf.ptr), n + 1);
            Eigen::Map<const ivec_t> eo_vec(static_cast<const index_t*>(eo_buf.ptr), n);
            Eigen::Map<const ivec_t> so_vec(static_cast<const index_t*>(so_buf.ptr), n);

            coxdev::reverse_cumsums(seq_vec, event_vec, start_vec, eo_vec, so_vec, do_event, do_start);
        },
        py::arg("sequence"), py::arg("event_buffer"), py::arg("start_buffer"),
        py::arg("event_order"), py::arg("start_order"),
        py::arg("do_event"), py::arg("do_start"),
        "Compute reverse cumsums in event and/or start order");
}
