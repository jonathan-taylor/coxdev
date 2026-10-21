# The R package against the Python package, as tests/test_R_package.py (which
# runs the comparison from Python, through rpy2): the deviance, its
# derivatives, the information, the residuals, the baseline hazard, the
# expected events, the C-index and the draws of the score (with the seed
# 2^64 - 1) for Efron's and Breslow's ties, without start times, with them,
# and with strata. The values, in helper-python-reference.R, were made by
# this Python script:
#
#   import numpy as np
#   from coxdev import CoxDeviance
#
#   event = np.array([3, 6, 8, 4, 6, 4, 3, 2, 2, 5, 3, 4, 7, 5, 6, 2.5])
#   status = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1])
#   start = np.array([0, 2, 3, 1, 4, 0, 1, 0, 0.5, 2, 0, 3, 3, 4, 2, 0])
#   strata = np.array([0, 1] * 8)
#   w = np.array([1, 2, 0.5, 1.5, 0, 1, 1, 2, 0.25, 1, 3, 0, 1, 1.5, 2, 0.75])
#   eta = np.round(np.linspace(-1, 1, 16) ** 3 + 0.1 * np.arange(16) % 0.7, 6)
#   X = np.column_stack([np.round(np.sin(np.arange(16)), 6), np.round(np.cos(3 * np.arange(16)), 6)])
#   v = X[:, 0]
#   ones = np.ones(16); ones[[4, 11]] = 0
#
#   def r(x):
#       x = np.atleast_1d(np.asarray(x, float)).ravel(order='F')
#       return 'c(' + ', '.join(repr(float(a)) for a in x) + ')'
#
#   out = []
#   for ties in ['efron', 'breslow']:
#       for label, kw in [('plain', dict()), ('start', dict(start=start)), ('strata', dict(start=start, strata=strata))]:
#           cox = CoxDeviance(event=event, status=status, sample_weight=w, tie_breaking=ties, **kw)
#           res = cox(eta, w)
#           I = cox.information(eta, w)
#           H = cox.baseline_hazard(eta, sample_weight=w)
#           d = dict(deviance=r(res.deviance), loglik_sat=r(res.loglik_sat), gradient=r(res.gradient),
#                    diag_hessian=r(res.diag_hessian), Iv=r(I @ v),
#                    score_residuals=r(cox.score_residuals(eta, X, sample_weight=w)),
#                    schoenfeld=r(cox.schoenfeld_residuals(eta, X, sample_weight=w)),
#                    risk_var=r(cox.risk_set_variance(eta, X, sample_weight=w)),
#                    expected=r(cox.expected_events(eta, sample_weight=w)),
#                    hazard_time=r(H.time), cumulative_hazard=r(H.cumulative_hazard),
#                    c_index=r(cox.c_index(eta, sample_weight=w)))
#           cox1 = CoxDeviance(event=event, status=status, sample_weight=ones, tie_breaking=ties, **kw)
#           for m in ['multinomial', 'bernoulli', 'fisher', 'wild']:
#               d['draws_' + m] = r(cox1.sample_score(eta, sample_weight=ones, method=m, n_draws=2,
#                                                     seed=2**64 - 1))
#           body = ',\n      '.join(f'{k} = {val}' for k, val in d.items())
#           out.append(f'  "{ties} {label}" = list(\n      {body})')
#   print('python_reference <- list(\n' + ',\n'.join(out) + ')')
#   print('# eta', r(eta)); print('# X', r(X))

test_that("the results are those of the Python package", {
  d <- python_data
  ones <- rep(1, 16)
  ones[c(5, 12)] <- 0
  close <- function(a, b, tol = 1e-12) {
    expect_equal(length(a), length(b))
    expect_lt(max(abs(as.vector(a) - b)), tol * max(1, max(abs(b))))
  }
  for (ties in c("efron", "breslow")) {
    for (label in c("plain", "start", "strata")) {
      ref <- python_reference[[paste(ties, label)]]
      start <- if (label != "plain") d$start
      strata <- if (label == "strata") d$strata
      cox <- CoxDeviance$new(event = d$event, status = d$status, start = start, strata = strata,
                             sample_weight = d$weight, tie_breaking = ties)
      r <- cox$compute(d$eta, d$weight)
      close(r$deviance, ref$deviance)
      close(r$loglik_sat, ref$loglik_sat)
      close(r$gradient, ref$gradient)
      close(r$diag_hessian, ref$diag_hessian)
      close(cox$information(d$eta, d$weight)$multiply(d$X[, 1]), ref$Iv)
      close(cox$score_residuals(d$eta, d$X, sample_weight = d$weight), ref$score_residuals)
      close(cox$schoenfeld_residuals(d$eta, d$X, sample_weight = d$weight), ref$schoenfeld)
      close(cox$risk_set_variance(d$eta, d$X, sample_weight = d$weight), ref$risk_var)
      close(cox$expected_events(d$eta, sample_weight = d$weight), ref$expected)
      H <- cox$baseline_hazard(d$eta, sample_weight = d$weight)
      expect_identical(H$time, ref$hazard_time)
      close(H$cumulative_hazard, ref$cumulative_hazard)
      close(cox$c_index(d$eta, sample_weight = d$weight), ref$c_index)

      cox1 <- CoxDeviance$new(event = d$event, status = d$status, start = start, strata = strata,
                              sample_weight = ones, tie_breaking = ties)
      for (method in c("multinomial", "bernoulli", "fisher", "wild")) {
        draws <- cox1$sample_score(d$eta, sample_weight = ones, method = method, n_draws = 2,
                                   seed = "18446744073709551615")
        expect_equal(dim(draws), c(16L, 2L))
        close(draws, ref[[paste0("draws_", method)]])
      }
    }
  }
})
