// The coxdev C API (coxdev_c_api.h) over the header-only C++ library.
#define COXDEV_C_BUILD 1
#include "coxdev_c_api.h"
#include "coxdev.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

thread_local std::string last_error;

int fail(int code, const char *message) {
    last_error = message;
    return code;
}

// Run f, turning exceptions into error codes and messages.
template <class F>
int guarded(F &&f) {
    try {
        last_error.clear();
        f();
        return COXDEV_OK;
    } catch (const std::invalid_argument &e) {
        return fail(COXDEV_ERR_ARGUMENT, e.what());
    } catch (const std::bad_alloc &) {
        return fail(COXDEV_ERR_ALLOC, "out of memory");
    } catch (const std::exception &e) {
        return fail(COXDEV_ERR_RUNTIME, e.what());
    } catch (...) {
        return fail(COXDEV_ERR_UNKNOWN, "unknown error");
    }
}

void require(bool ok, const char *message) {
    if (!ok) throw std::invalid_argument(message);
}

using MapXd = Eigen::Map<const Eigen::VectorXd>;
using MapXi = Eigen::Map<const Eigen::VectorXi>;

Eigen::VectorXd vec(const double *x, size_t n) {
    return MapXd(x, static_cast<Eigen::Index>(n));
}

}  // namespace

struct coxdev_s {
    size_t n;
    Eigen::VectorXd start, event, weight;
    Eigen::VectorXi status, strata;
    bool stratified;
    bool efron;
    // the last evaluation: eta and its deviance, reused while eta and the
    // weights are unchanged (as coxdev's Python CoxDeviance caches its last
    // call), so that information products do not recompute the deviance
    Eigen::VectorXd last_eta;
    double last_deviance = 0.;
    bool have_last = false;
    std::unique_ptr<CoxDeviance> cox;
    std::unique_ptr<StratifiedCoxDeviance> scox;

    void build() {
        have_last = false;
        if (stratified) {
            scox = std::make_unique<StratifiedCoxDeviance>(start, event, status, strata, weight, efron);
        } else {
            cox = std::make_unique<CoxDeviance>(start, event, status, weight, efron);
        }
    }

    // Apply f to the C++ object, whichever kind it is.
    template <class F>
    auto visit(F &&f) -> decltype(f(*cox)) {
        return stratified ? f(*scox) : f(*cox);
    }
    template <class F>
    auto cvisit(F &&f) const -> decltype(f(std::as_const(*cox))) {
        return stratified ? f(std::as_const(*scox)) : f(std::as_const(*cox));
    }

    // Evaluate at (eta, sample_weight), rebuilding with new weights first if
    // they differ from the handle's (as coxdev's Python CoxDeviance does).
    double evaluate(const double *eta, const double *sample_weight) {
        require(eta != nullptr, "eta must not be NULL");
        const auto N = static_cast<Eigen::Index>(n);
        if (sample_weight != nullptr) {
            MapXd sw(sample_weight, N);          // compared in place, copied only if new
            if (sw != weight) {
                weight = sw;
                build();
            }
        }
        MapXd eta_m(eta, N);
        if (have_last && eta_m == last_eta) return last_deviance;
        last_eta = eta_m;
        last_deviance = visit([&](auto &c) { return c.compute_deviance(last_eta, weight); });
        have_last = true;
        return last_deviance;
    }
};

extern "C" {

int coxdev_api_version(void) { return COXDEV_C_API_VERSION; }

const char *coxdev_last_error(void) { return last_error.c_str(); }

int coxdev_new(size_t n, const double *start, const double *event, const int *status,
               const int *strata, const double *weight, int efron, coxdev_t **out) {
    if (out == nullptr) return fail(COXDEV_ERR_ARGUMENT, "out must not be NULL");
    *out = nullptr;
    return guarded([&] {
        require(n > 0, "n must be positive");
        require(event != nullptr && status != nullptr, "event and status must not be NULL");
        auto h = std::make_unique<coxdev_s>();
        h->n = n;
        h->event = vec(event, n);
        h->status = MapXi(status, static_cast<Eigen::Index>(n));
        for (Eigen::Index i = 0; i < h->status.size(); ++i)
            require(h->status[i] == 0 || h->status[i] == 1, "status must be 0 or 1");
        h->start = start ? vec(start, n)
                         : Eigen::VectorXd::Constant(n, -std::numeric_limits<double>::infinity());
        h->weight = weight ? vec(weight, n) : Eigen::VectorXd::Ones(n);
        h->stratified = strata != nullptr;
        if (h->stratified) h->strata = MapXi(strata, static_cast<Eigen::Index>(n));
        h->efron = efron != 0;
        h->build();
        *out = h.release();
    });
}

void coxdev_free(coxdev_t *h) { delete h; }

size_t coxdev_n(const coxdev_t *h) { return h ? h->n : 0; }

int coxdev_deviance(coxdev_t *h, const double *eta, const double *sample_weight,
                    double *deviance, double *gradient, double *diag_hessian, double *loglik_sat) {
    if (h == nullptr) return fail(COXDEV_ERR_ARGUMENT, "handle must not be NULL");
    return guarded([&] {
        double dev = h->evaluate(eta, sample_weight);
        if (deviance) *deviance = dev;
        h->visit([&](auto &c) {
            if (gradient) std::memcpy(gradient, c.get_gradient().data(), h->n * sizeof(double));
            if (diag_hessian) std::memcpy(diag_hessian, c.get_diag_hessian().data(), h->n * sizeof(double));
            if (loglik_sat) *loglik_sat = c.get_loglik_sat();
            return 0;
        });
    });
}

int coxdev_hessian_matvec(coxdev_t *h, const double *eta, const double *sample_weight,
                          const double *v, double *out) {
    if (h == nullptr) return fail(COXDEV_ERR_ARGUMENT, "handle must not be NULL");
    return guarded([&] {
        require(v != nullptr && out != nullptr, "v and out must not be NULL");
        h->evaluate(eta, sample_weight);
        Eigen::VectorXd arg = vec(v, h->n), res(h->n);
        h->visit([&](auto &c) { c.compute_hessian_matvec(arg, res); return 0; });
        std::memcpy(out, res.data(), h->n * sizeof(double));
    });
}

int coxdev_information_derivative_matvec(coxdev_t *h, const double *eta, const double *sample_weight,
                                         const double *direction, const double *v, double *out) {
    if (h == nullptr) return fail(COXDEV_ERR_ARGUMENT, "handle must not be NULL");
    return guarded([&] {
        require(direction != nullptr && v != nullptr && out != nullptr,
                "direction, v and out must not be NULL");
        h->evaluate(eta, sample_weight);
        Eigen::VectorXd dir = vec(direction, h->n), arg = vec(v, h->n), res(h->n);
        h->visit([&](auto &c) { c.compute_information_derivative_matvec(dir, arg, res); return 0; });
        std::memcpy(out, res.data(), h->n * sizeof(double));
    });
}

int coxdev_baseline_hazard(coxdev_t *h, const double *eta, const double *sample_weight,
                           size_t *m, double *time, double *hazard, double *cumulative_hazard,
                           int *stratum) {
    if (h == nullptr) return fail(COXDEV_ERR_ARGUMENT, "handle must not be NULL");
    return guarded([&] {
        require(m != nullptr, "m must not be NULL");
        h->evaluate(eta, sample_weight);
        BaselineHazard bh = h->visit([&](auto &c) { return c.baseline_hazard(); });
        size_t k = static_cast<size_t>(bh.time.size());
        if (time == nullptr && hazard == nullptr && cumulative_hazard == nullptr && stratum == nullptr) {
            *m = k;
            return;
        }
        require(*m >= k, "buffers are shorter than the baseline hazard (query its length first)");
        if (time) std::memcpy(time, bh.time.data(), k * sizeof(double));
        if (hazard) std::memcpy(hazard, bh.hazard.data(), k * sizeof(double));
        if (cumulative_hazard) std::memcpy(cumulative_hazard, bh.cumulative_hazard.data(), k * sizeof(double));
        if (stratum) for (size_t i = 0; i < k; ++i) stratum[i] = bh.stratum[static_cast<Eigen::Index>(i)];
        *m = k;
    });
}

int coxdev_handle_concordance(const coxdev_t *h, const double *pred, const double *weight,
                              double *numerator, double *denominator, double *value) {
    if (h == nullptr) return fail(COXDEV_ERR_ARGUMENT, "handle must not be NULL");
    return guarded([&] {
        require(pred != nullptr, "pred must not be NULL");
        Eigen::VectorXd p = vec(pred, h->n);
        Eigen::VectorXd w = weight ? vec(weight, h->n) : h->weight;
        Concordance c = h->cvisit([&](const auto &d) { return d.concordance(p, w); });
        if (numerator) *numerator = c.numerator;
        if (denominator) *denominator = c.denominator;
        if (value) *value = c.value();
    });
}

int coxdev_concordance(size_t n, const double *pred, const double *start, const double *event,
                       const int *status, const double *weight,
                       double *numerator, double *denominator, double *value) {
    return guarded([&] {
        require(n > 0, "n must be positive");
        require(pred && event && status, "pred, event and status must not be NULL");
        Eigen::VectorXd s = start ? vec(start, n)
                                  : Eigen::VectorXd::Constant(n, -std::numeric_limits<double>::infinity());
        Eigen::VectorXd w = weight ? vec(weight, n) : Eigen::VectorXd::Ones(n);
        Eigen::VectorXi st = MapXi(status, static_cast<Eigen::Index>(n));
        Concordance c = concordance(vec(pred, n), s, vec(event, n), st, w);
        if (numerator) *numerator = c.numerator;
        if (denominator) *denominator = c.denominator;
        if (value) *value = c.value();
    });
}

int coxdev_score_residuals(coxdev_t *h, const double *eta, const double *sample_weight,
                           const double *x, size_t p, double *out) {
    if (h == nullptr) return fail(COXDEV_ERR_ARGUMENT, "handle must not be NULL");
    return guarded([&] {
        require(x != nullptr && out != nullptr, "x and out must not be NULL");
        h->evaluate(eta, sample_weight);
        for (size_t j = 0; j < p; ++j) {
            Eigen::VectorXd col = vec(x + j * h->n, h->n);
            Eigen::VectorXd r = h->cvisit([&](const auto &c) { return c.score_residuals(col); });
            std::memcpy(out + j * h->n, r.data(), h->n * sizeof(double));
        }
    });
}

int coxdev_wild_score(coxdev_t *h, const double *eta, const double *sample_weight,
                      const double *g, double *out) {
    if (h == nullptr) return fail(COXDEV_ERR_ARGUMENT, "handle must not be NULL");
    return guarded([&] {
        require(g != nullptr && out != nullptr, "g and out must not be NULL");
        h->evaluate(eta, sample_weight);
        Eigen::VectorXd gv = vec(g, h->n);
        Eigen::VectorXd r = h->visit([&](auto &c) { return c.wild_score(gv); });
        std::memcpy(out, r.data(), h->n * sizeof(double));
    });
}

int coxdev_sample_score(coxdev_t *h, const double *eta, const double *sample_weight,
                        const char *method, size_t n_draws, uint64_t seed, double *out) {
    if (h == nullptr) return fail(COXDEV_ERR_ARGUMENT, "handle must not be NULL");
    return guarded([&] {
        require(method != nullptr && out != nullptr, "method and out must not be NULL");
        require(n_draws > 0, "n_draws must be positive");
        h->evaluate(eta, sample_weight);
        std::string m(method);
        Eigen::MatrixXd d = h->visit([&](auto &c) {
            return c.sample_score(m, static_cast<int>(n_draws), seed);
        });
        require(static_cast<size_t>(d.rows()) == h->n && static_cast<size_t>(d.cols()) == n_draws,
                "unexpected shape of the draws");
        std::memcpy(out, d.data(), h->n * n_draws * sizeof(double));  // Eigen is column-major
    });
}

}  // extern "C"
