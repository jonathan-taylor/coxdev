/*
 * coxdev C API: a plain C interface to the header-only coxdev library, for
 * foreign-function interfaces (Julia ccall, Rust, the JVM, ctypes, ...).
 *
 * A coxdev_t is an opaque handle to a Cox model's data: (start, event] times,
 * status, optional strata, and observation weights, with Efron's or Breslow's
 * ties. The functions that take a linear predictor eta evaluate the model at
 * (eta, sample_weight), as coxdev's Python CoxDeviance does: a sample_weight
 * different from the weights of the handle rebuilds it with the new weights.
 * The last evaluation is reused while eta and the weights are unchanged
 * (compared by value), so that, e.g., coxdev_deviance followed by any number
 * of coxdev_hessian_matvec at the same eta evaluates the model once.
 *
 * Conventions
 * - Every function returns COXDEV_OK (0) or a negative error code; the message
 *   of the last error on the calling thread is coxdev_last_error().
 * - Arrays are passed as pointers with explicit lengths, in the native order
 *   of the observations (the order given to coxdev_new). The library copies
 *   what it keeps; the caller owns all arrays. Outputs are written to caller
 *   allocated buffers.
 * - Optional inputs (start, strata, weight, sample_weight) may be NULL:
 *   no start times (left truncation at -infinity), no strata, weights one,
 *   and sample_weight = the handle's weights.
 * - A handle is not safe for concurrent use from several threads (it keeps
 *   work buffers and the state of its last evaluation); distinct handles are.
 */
#ifndef COXDEV_C_API_H
#define COXDEV_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(COXDEV_C_BUILD)
#    define COXDEV_C_EXPORT __declspec(dllexport)
#  else
#    define COXDEV_C_EXPORT __declspec(dllimport)
#  endif
#else
#  define COXDEV_C_EXPORT __attribute__((visibility("default")))
#endif

/* Version of this C API (not of coxdev): raised when functions are added
   or changed. 2 adds coxdev_information_derivative_matvec; 3 adds
   coxdev_weight, coxdev_survival_table and coxdev_expected_events. */
#define COXDEV_C_API_VERSION 3

enum {
    COXDEV_OK = 0,
    COXDEV_ERR_ARGUMENT = -1,   /* invalid argument: NULL handle or array, bad length, bad value */
    COXDEV_ERR_RUNTIME = -2,    /* an error raised by coxdev (std::exception) */
    COXDEV_ERR_ALLOC = -3,      /* out of memory */
    COXDEV_ERR_UNKNOWN = -4     /* any other exception */
};

typedef struct coxdev_s coxdev_t;

/* The C API version the library was built with (COXDEV_C_API_VERSION). */
COXDEV_C_EXPORT int coxdev_api_version(void);

/* The message of the last error on this thread ("" if none). The pointer is
   valid until the next call into the library on this thread. */
COXDEV_C_EXPORT const char *coxdev_last_error(void);

/* Create a handle for n observations. event and status (0/1) are required;
   start, strata (integer labels) and weight may be NULL. efron: 1 for Efron's
   ties, 0 for Breslow's. On success *out is a new handle, to be released with
   coxdev_free. */
COXDEV_C_EXPORT int coxdev_new(size_t n,
                               const double *start,
                               const double *event,
                               const int *status,
                               const int *strata,
                               const double *weight,
                               int efron,
                               coxdev_t **out);

/* Release a handle (NULL is a no-op). */
COXDEV_C_EXPORT void coxdev_free(coxdev_t *h);

/* The number of observations of a handle (0 for NULL). */
COXDEV_C_EXPORT size_t coxdev_n(const coxdev_t *h);

/* The deviance 2 (loglik_sat - loglik) at (eta, sample_weight), with its
   gradient and diagonal Hessian in eta (n-vectors). deviance, gradient,
   diag_hessian and loglik_sat may each be NULL to skip them. */
COXDEV_C_EXPORT int coxdev_deviance(coxdev_t *h,
                                    const double *eta,
                                    const double *sample_weight,
                                    double *deviance,
                                    double *gradient,
                                    double *diag_hessian,
                                    double *loglik_sat);

/* out = I v, the information (half the Hessian of the deviance) at
   (eta, sample_weight) times v (n-vectors). */
COXDEV_C_EXPORT int coxdev_hessian_matvec(coxdev_t *h,
                                          const double *eta,
                                          const double *sample_weight,
                                          const double *v,
                                          double *out);

/* out = dI[direction] v, the derivative of the information at
   (eta, sample_weight) in the direction `direction` of eta,
   d/dt I(eta + t direction) at t = 0, times v (n-vectors). Symmetric in
   direction and v; O(n). For the derivatives of log determinants, as in
   REML. (C API version 2.) */
COXDEV_C_EXPORT int coxdev_information_derivative_matvec(coxdev_t *h,
                                                         const double *eta,
                                                         const double *sample_weight,
                                                         const double *direction,
                                                         const double *v,
                                                         double *out);

/* The baseline hazard at (eta, sample_weight). Call with time = NULL to get
   the number of entries in *m; then call again with buffers of length *m
   (time, hazard, cumulative_hazard, stratum; any may be NULL). On the second
   call *m must be the length from the first; it is set to the number written. */
COXDEV_C_EXPORT int coxdev_baseline_hazard(coxdev_t *h,
                                           const double *eta,
                                           const double *sample_weight,
                                           size_t *m,
                                           double *time,
                                           double *hazard,
                                           double *cumulative_hazard,
                                           int *stratum);

/* The handle's current weights (n-vector): those it was created with, or
   of the last call that passed weights. A NULL sample_weight in the other
   functions means these. (C API version 3.) */
COXDEV_C_EXPORT int coxdev_weight(const coxdev_t *h, double *out);

/* The survival table at (eta, sample_weight), as R's survfit.coxph lays it
   out: a row per distinct event time (of failures and censorings) in
   increasing time within each stratum, with the weight at risk, the weights
   of the failures and of the censorings there, and the baseline cumulative
   hazard (for eta as given, constant between failure times). The cumulative
   hazard of a subject with linear predictor e is cumulative_hazard * exp(e).
   As coxdev_baseline_hazard: call with every buffer NULL to get the number
   of rows in *m, then again with buffers of length *m (any may be NULL).
   (C API version 3.) */
COXDEV_C_EXPORT int coxdev_survival_table(coxdev_t *h,
                                          const double *eta,
                                          const double *sample_weight,
                                          size_t *m,
                                          double *time,
                                          double *n_risk,
                                          double *n_event,
                                          double *n_censor,
                                          double *cumulative_hazard,
                                          int *stratum);

/* The expected number of events of each observation at (eta,
   sample_weight) (n-vector, native order), survival's predict(type =
   "expected"): status - out is the martingale residual. (C API version 3.) */
COXDEV_C_EXPORT int coxdev_expected_events(coxdev_t *h,
                                           const double *eta,
                                           const double *sample_weight,
                                           double *out);

/* Harrell's concordance index of pred for the handle's data (within strata),
   with weights weight (NULL: the handle's): *numerator / *denominator, and
   *value (NaN when no pair is comparable). Any output may be NULL. */
COXDEV_C_EXPORT int coxdev_handle_concordance(const coxdev_t *h,
                                              const double *pred,
                                              const double *weight,
                                              double *numerator,
                                              double *denominator,
                                              double *value);

/* The same for unstratified (start, event] data without a handle. start and
   weight may be NULL. */
COXDEV_C_EXPORT int coxdev_concordance(size_t n,
                                       const double *pred,
                                       const double *start,
                                       const double *event,
                                       const int *status,
                                       const double *weight,
                                       double *numerator,
                                       double *denominator,
                                       double *value);

/* The score residuals of the p columns of x (n x p, column-major) at
   (eta, sample_weight), into out (n x p, column-major). */
COXDEV_C_EXPORT int coxdev_score_residuals(coxdev_t *h,
                                           const double *eta,
                                           const double *sample_weight,
                                           const double *x,
                                           size_t p,
                                           double *out);

/* The wild draw for multipliers g (n-vector) at (eta, sample_weight). */
COXDEV_C_EXPORT int coxdev_wild_score(coxdev_t *h,
                                      const double *eta,
                                      const double *sample_weight,
                                      const double *g,
                                      double *out);

/* n_draws draws of the score at (eta, sample_weight), into out (n x n_draws,
   column-major). method: "multinomial", "bernoulli", "fisher" or "wild". The
   random numbers depend only on seed, not on the platform, so the draws equal
   those of coxdev's Python and R packages for the same seed, to rounding: they
   go through exp, log, sin and cos, whose last bits can differ between math
   libraries. */
COXDEV_C_EXPORT int coxdev_sample_score(coxdev_t *h,
                                        const double *eta,
                                        const double *sample_weight,
                                        const char *method,
                                        size_t n_draws,
                                        uint64_t seed,
                                        double *out);

#ifdef __cplusplus
}
#endif

#endif /* COXDEV_C_API_H */
