#include <RcppEigen.h>
#include "coxdev.h"

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;

// Helper to wrap compute_hessian_matvec since Rcpp doesn't natively handle the 
// passed-by-reference output vector in the same way pybind11 lambdas can be inline easily.
Eigen::VectorXd compute_hessian_matvec_cox(CoxDeviance* self, const Eigen::VectorXd& arg) {
    Eigen::VectorXd out(arg.size());
    self->compute_hessian_matvec(arg, out);
    return out;
}

Eigen::VectorXd compute_hessian_matvec_strat(StratifiedCoxDeviance* self, const Eigen::VectorXd& arg) {
    Eigen::VectorXd out(arg.size());
    self->compute_hessian_matvec(arg, out);
    return out;
}

RCPP_MODULE(coxdev_module) {
    class_<CoxDeviance>("CoxDeviance")
        .constructor<const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXi&, const Eigen::VectorXd&, bool>()
        .method("compute_deviance", &CoxDeviance::compute_deviance)
        .method("compute_hessian_matvec", &compute_hessian_matvec_cox)
        .property("gradient", &CoxDeviance::get_gradient)
        .property("diag_hessian", &CoxDeviance::get_diag_hessian)
        .property("linear_predictor", &CoxDeviance::get_linear_predictor)
        .property("sample_weight", &CoxDeviance::get_sample_weight)
        .property("loglik_sat", &CoxDeviance::get_loglik_sat)
        .property("event_order", &CoxDeviance::get_event_order)
        .property("start_order", &CoxDeviance::get_start_order)
        .property("first", &CoxDeviance::get_first)
        .property("last", &CoxDeviance::get_last)
        .property("start_map", &CoxDeviance::get_start_map)
        .property("event_map", &CoxDeviance::get_event_map)
        .property("scaling", &CoxDeviance::get_scaling)
        .property("status", &CoxDeviance::get_status)
        .property("event", &CoxDeviance::get_event)
        .property("start", &CoxDeviance::get_start)
    ;

    class_<StratifiedCoxDeviance>("StratifiedCoxDeviance")
        .constructor<const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXd&, bool>()
        .method("compute_deviance", &StratifiedCoxDeviance::compute_deviance)
        .method("compute_hessian_matvec", &compute_hessian_matvec_strat)
        .property("gradient", &StratifiedCoxDeviance::get_gradient)
        .property("diag_hessian", &StratifiedCoxDeviance::get_diag_hessian)
        .property("linear_predictor", &StratifiedCoxDeviance::get_linear_predictor)
        .property("sample_weight", &StratifiedCoxDeviance::get_sample_weight)
        .property("loglik_sat", &StratifiedCoxDeviance::get_loglik_sat)
        // Returning vectors of custom objects requires special handling in Rcpp Modules, 
        // so get_cox_devs might need a custom getter if it returns a std::vector<CoxDeviance>.
        // .property("cox_devs", &StratifiedCoxDeviance::get_cox_devs)
        .property("unique_strata", &StratifiedCoxDeviance::get_unique_strata)
        .property("stratum_indices", &StratifiedCoxDeviance::get_stratum_indices)
    ;
}
