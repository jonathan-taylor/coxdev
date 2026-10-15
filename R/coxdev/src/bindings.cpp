// R bindings of the CoxDeviance and StratifiedCoxDeviance classes of
// cpp/src/coxdev.cpp (copied here by tools/sync_cpp.sh). The R6 classes of
// R/cox_deviance.R hold an external pointer to a CoxModel.

// [[Rcpp::depends(RcppEigen)]]
#include "coxdev.h"

#include <cctype>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

using namespace Rcpp;

namespace {

// One of the two classes: stratified if strata were given.
struct CoxModel {
  std::unique_ptr<CoxDeviance> plain;
  std::unique_ptr<StratifiedCoxDeviance> strat;
  bool computed = false;

  template <class F>
  auto apply(F f) -> decltype(f(*plain)) {
    return plain ? f(*plain) : f(*strat);
  }
};

CoxModel* get_model(SEXP ptr) {
  XPtr<CoxModel> model(ptr);
  if (!model.get()) {
    stop("coxdev: the C++ object is no longer valid");
  }
  return model.get();
}

CoxModel* get_computed(SEXP ptr) {
  CoxModel* model = get_model(ptr);
  if (!model->computed) {
    stop("coxdev: compute the deviance first");
  }
  return model;
}

// seeds are passed as decimal strings, as R has no 64-bit integers
std::uint64_t parse_seed(const std::string& seed) {
  bool ok = !seed.empty() && std::isdigit(static_cast<unsigned char>(seed[0]));
  std::uint64_t value = 0;
  if (ok) {
    try {
      std::size_t pos = 0;
      value = std::stoull(seed, &pos, 10);
      ok = (pos == seed.size());
    } catch (const std::exception&) {
      ok = false;
    }
  }
  if (!ok) {
    stop("seed must be a nonnegative integer below 2^64");
  }
  return value;
}

}  // namespace

// strata of length 0: unstratified
// [[Rcpp::export(.cox_new)]]
SEXP cox_new(const Eigen::VectorXd& start,
             const Eigen::VectorXd& event,
             const Eigen::VectorXi& status,
             const Eigen::VectorXi& strata,
             const Eigen::VectorXd& weight,
             bool efron) {
  std::unique_ptr<CoxModel> model(new CoxModel());
  if (strata.size() == 0) {
    model->plain.reset(new CoxDeviance(start, event, status, weight, efron));
  } else {
    model->strat.reset(new StratifiedCoxDeviance(start, event, status, strata, weight, efron));
  }
  return XPtr<CoxModel>(model.release(), true);
}

// [[Rcpp::export(.cox_valid)]]
bool cox_valid(SEXP ptr) {
  return TYPEOF(ptr) == EXTPTRSXP && R_ExternalPtrAddr(ptr) != nullptr;
}

// [[Rcpp::export(.cox_compute)]]
List cox_compute(SEXP ptr,
                 const Eigen::VectorXd& eta,
                 const Eigen::VectorXd& sample_weight) {
  CoxModel* model = get_model(ptr);
  return model->apply([&](auto& cox) {
    double deviance = cox.compute_deviance(eta, sample_weight);
    model->computed = true;
    return List::create(_["deviance"] = deviance,
                        _["loglik_sat"] = cox.get_loglik_sat(),
                        _["gradient"] = cox.get_gradient(),
                        _["diag_hessian"] = cox.get_diag_hessian());
  });
}

// the information (half the Hessian of the deviance) times each column of arg
// [[Rcpp::export(.cox_information_matmat)]]
Eigen::MatrixXd cox_information_matmat(SEXP ptr, const Eigen::MatrixXd& arg) {
  CoxModel* model = get_computed(ptr);
  Eigen::MatrixXd out(arg.rows(), arg.cols());
  Eigen::VectorXd col_in(arg.rows()), col_out(arg.rows());
  model->apply([&](auto& cox) {
    for (Eigen::Index j = 0; j < arg.cols(); ++j) {
      col_in = arg.col(j);
      cox.compute_hessian_matvec(col_in, col_out);
      out.col(j) = col_out;
    }
    return 0;
  });
  return out;
}

// [[Rcpp::export(.cox_sample_score)]]
Eigen::MatrixXd cox_sample_score(SEXP ptr,
                                 const std::string& method,
                                 int n_draws,
                                 const std::string& seed) {
  CoxModel* model = get_computed(ptr);
  std::uint64_t s = parse_seed(seed);
  return model->apply([&](auto& cox) { return cox.sample_score(method, n_draws, s); });
}

// [[Rcpp::export(.cox_wild_score)]]
Eigen::VectorXd cox_wild_score(SEXP ptr, const Eigen::VectorXd& multipliers) {
  CoxModel* model = get_computed(ptr);
  return model->apply([&](auto& cox) { return cox.wild_score(multipliers); });
}

// [[Rcpp::export(.cox_score_residuals)]]
Eigen::VectorXd cox_score_residuals(SEXP ptr, const Eigen::VectorXd& x) {
  CoxModel* model = get_computed(ptr);
  return model->apply([&](auto& cox) { return cox.score_residuals(x); });
}

// [[Rcpp::export(.cox_schoenfeld_residuals)]]
Eigen::VectorXd cox_schoenfeld_residuals(SEXP ptr, const Eigen::VectorXd& x) {
  CoxModel* model = get_computed(ptr);
  return model->apply([&](auto& cox) { return cox.schoenfeld_residuals(x); });
}

// [[Rcpp::export(.cox_risk_set_variance)]]
Eigen::VectorXd cox_risk_set_variance(SEXP ptr, const Eigen::VectorXd& z) {
  CoxModel* model = get_computed(ptr);
  return model->apply([&](auto& cox) { return cox.risk_set_variance(z); });
}

// [[Rcpp::export(.cox_expected_events)]]
Eigen::VectorXd cox_expected_events(SEXP ptr) {
  CoxModel* model = get_computed(ptr);
  return model->apply([&](auto& cox) { return cox.expected_events(); });
}

// Harrell's C-index as c(numerator, denominator): needs no compute
// [[Rcpp::export(.cox_concordance)]]
Eigen::VectorXd cox_concordance(SEXP ptr, const Eigen::VectorXd& pred,
                                const Eigen::VectorXd& weight) {
  CoxModel* model = get_model(ptr);
  Concordance c = model->apply([&](auto& cox) { return cox.concordance(pred, weight); });
  Eigen::VectorXd out(2);
  out << c.numerator, c.denominator;
  return out;
}

// [[Rcpp::export(.cox_baseline_hazard)]]
List cox_baseline_hazard(SEXP ptr) {
  CoxModel* model = get_computed(ptr);
  BaselineHazard h = model->apply([&](auto& cox) { return cox.baseline_hazard(); });
  return List::create(_["time"] = h.time,
                      _["hazard"] = h.hazard,
                      _["cumulative_hazard"] = h.cumulative_hazard,
                      _["stratum"] = h.stratum);
}
