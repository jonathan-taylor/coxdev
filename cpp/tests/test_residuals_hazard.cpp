// Tests of CoxDeviance::score_residuals and CoxDeviance::baseline_hazard
// (and their stratified versions) against direct O(n^2) computations, and of
// the identities they satisfy with the score and wild_score.
//
// Build and run from the top of the repository:
//   c++ -std=c++17 -O2 -Ieigen -Icpp/include cpp/src/coxdev.cpp \
//       cpp/tests/test_residuals_hazard.cpp -o test_residuals_hazard
//   ./test_residuals_hazard

#include "coxdev.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>

namespace {

int failures = 0;

void check(double error, double tol, const std::string& what) {
  if (!(error <= tol)) {
    std::printf("FAIL %s: error %.3g > %.3g\n", what.c_str(), error, tol);
    ++failures;
  }
}

double max_abs(const Eigen::MatrixXd& A) { return A.cwiseAbs().maxCoeff(); }

struct Data {
  Eigen::VectorXd start, event, weight, eta;
  Eigen::VectorXi status, strata;
  Eigen::MatrixXd X;
};

Data simulate(int n, int p, bool start, bool zeros, int n_strata, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::normal_distribution<double> norm(0.0, 1.0);
  Data d;
  d.X.resize(n, p);
  d.event.resize(n);
  d.start.resize(n);
  d.weight.resize(n);
  d.eta.resize(n);
  d.status.resize(n);
  d.strata.resize(n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < p; ++j) d.X(i, j) = norm(rng);
    d.eta(i) = 0.5 * d.X(i, 0) - 0.3 * d.X(i, 1) + 2.0;   // far from mean zero
    d.event(i) = std::ceil(-std::log(unif(rng)) * 4) / 4;  // a coarse grid: many ties
    d.status(i) = unif(rng) < 0.7;
    d.start(i) = start ? std::floor(unif(rng) * 0.5 * d.event(i) * 8) / 8
                       : -std::numeric_limits<double>::infinity();
    d.weight(i) = 0.5 + unif(rng);
    if (zeros && unif(rng) < 0.15) d.weight(i) = 0.0;
    d.strata(i) = i % n_strata;
  }
  return d;
}

// subjects of the stratum s (or all, s < 0)
bool in(const Data& d, int i, int s) { return s < 0 || d.strata(i) == s; }

// Direct computation: for each failure time tau with failures D (weight > 0),
// d of them, mean weight wbar, risk set R = {start < tau <= event, weight > 0}
// and Efron terms r = 0..d-1 with s = r / d (0 for Breslow),
//   p_r = e (R - s D) / sum(e (R - s D)),  e = w exp(eta),
//   U_i += (w_i 1{i in D} / d - wbar p_ri) (x_i - xbar_r),
//   hazard(tau) += wbar / sum(e (R - s D)).
void reference(const Data& d, bool efron, int s, Eigen::MatrixXd& U,
               std::map<double, double>& hazard) {
  int n = d.event.size(), p = d.X.cols();
  U = Eigen::MatrixXd::Zero(n, p);
  hazard.clear();
  std::set<double> taus;
  for (int i = 0; i < n; ++i) {
    if (in(d, i, s) && d.status(i) == 1 && d.weight(i) > 0) taus.insert(d.event(i));
  }
  for (double tau : taus) {
    std::vector<int> D, R;
    double wsum = 0.0;
    for (int i = 0; i < n; ++i) {
      if (!in(d, i, s) || d.weight(i) <= 0) continue;
      if (d.status(i) == 1 && d.event(i) == tau) {
        D.push_back(i);
        wsum += d.weight(i);
      }
      if (d.start(i) < tau && tau <= d.event(i)) R.push_back(i);
    }
    int nd = D.size();
    double wbar = wsum / nd;
    for (int r = 0; r < nd; ++r) {
      double sc = efron ? double(r) / nd : 0.0;
      Eigen::VectorXd pr = Eigen::VectorXd::Zero(n);
      for (int i : R) pr(i) += d.weight(i) * std::exp(d.eta(i));
      for (int i : D) pr(i) -= sc * d.weight(i) * std::exp(d.eta(i));
      double S = pr.sum();
      hazard[tau] += wbar / S;
      pr /= S;
      Eigen::RowVectorXd xbar = pr.transpose() * d.X;
      for (int i = 0; i < n; ++i) {
        double a = -wbar * pr(i);
        for (int k : D) {
          if (k == i) a += d.weight(i) / nd;
        }
        if (a != 0.0) U.row(i) += a * (d.X.row(i) - xbar);
      }
    }
  }
}

template <class Cox>
void check_model(Cox& cox, const Data& d, bool efron, bool stratified, const std::string& name) {
  int n = d.event.size();
  cox.compute_deviance(d.eta, d.weight);
  // the residuals of the design, a column at a time
  Eigen::MatrixXd U(n, d.X.cols());
  for (Eigen::Index j = 0; j < d.X.cols(); ++j) U.col(j) = cox.score_residuals(Eigen::VectorXd(d.X.col(j)));
  BaselineHazard H = cox.baseline_hazard();

  // the residuals sum to the score, and sum_i G_i U_i = X^T wild_score(G)
  Eigen::VectorXd score = -0.5 * cox.get_gradient();
  Eigen::VectorXd XtM = d.X.transpose() * score;
  check(max_abs(U.colwise().sum().transpose() - XtM), 1e-9 * (1 + max_abs(XtM)), name + ": sum of residuals");
  std::mt19937_64 rng(5);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::VectorXd G(n);
  for (int i = 0; i < n; ++i) G(i) = norm(rng);
  Eigen::VectorXd XtW = d.X.transpose() * cox.wild_score(G);
  check(max_abs(U.transpose() * G - XtW), 1e-9 * (1 + max_abs(XtW)), name + ": wild_score");

  // against the direct computation, stratum by stratum
  Eigen::MatrixXd U_ref = Eigen::MatrixXd::Zero(n, d.X.cols()), U_s;
  std::map<std::pair<int, double>, double> hazard_ref;
  std::set<int> strata;
  for (int i = 0; i < n; ++i) strata.insert(stratified ? d.strata(i) : -1);
  for (int s : strata) {
    std::map<double, double> h;
    reference(d, efron, s, U_s, h);
    U_ref += U_s;
    for (auto& kv : h) hazard_ref[{stratified ? s : 0, kv.first}] = kv.second;
  }
  check(max_abs(U - U_ref), 1e-9 * (1 + max_abs(U_ref)), name + ": residuals");

  bool same_size = (int) hazard_ref.size() == H.time.size();
  check(same_size ? 0.0 : 1.0, 0.0, name + ": number of failure times");
  if (same_size) {
    double err = 0.0, cum = 0.0;
    int k = 0, last_stratum = std::numeric_limits<int>::min();
    for (auto& kv : hazard_ref) {
      if (kv.first.first != last_stratum) cum = 0.0;
      last_stratum = kv.first.first;
      cum += kv.second;
      err = std::max({err,
                      std::abs(H.time(k) - kv.first.second),
                      std::abs(double(H.stratum(k) - kv.first.first)),
                      std::abs(H.hazard(k) - kv.second) / kv.second,
                      std::abs(H.cumulative_hazard(k) - cum) / cum});
      ++k;
    }
    check(err, 1e-10, name + ": baseline hazard");
  }
}

}  // namespace

int main() {
  std::mt19937_64 rng(2024);
  int cases = 0;
  for (int efron = 0; efron < 2; ++efron) {
    for (int start = 0; start < 2; ++start) {
      for (int zeros = 0; zeros < 2; ++zeros) {
        for (int n_strata : {1, 3}) {
          for (int rep = 0; rep < 3; ++rep) {
            Data d = simulate(80, 3, start, zeros, n_strata, rng);
            std::string name = std::string(efron ? "efron" : "breslow") +
                               (start ? " start" : "") + (zeros ? " zeros" : "") +
                               (n_strata > 1 ? " strata" : "") + " rep " + std::to_string(rep);
            if (n_strata == 1) {
              CoxDeviance cox(d.start, d.event, d.status, d.weight, efron);
              check_model(cox, d, efron, false, name);
            } else {
              StratifiedCoxDeviance cox(d.start, d.event, d.status, d.strata, d.weight, efron);
              check_model(cox, d, efron, true, name);
            }
            ++cases;
          }
        }
      }
    }
  }
  std::printf("%d cases, %d failures\n", cases, failures);
  return failures == 0 ? 0 : 1;
}
