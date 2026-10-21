# The residuals, baseline hazard, expected events and risk-set variances
# against direct computations (helper-reference.R), with weights all one,
# unequal and with zeros, as tests/test_residuals_hazard.py.
# test-residuals-hazard.R compares the same with the survival package.

WEIGHTS <- c("ones", "unequal", "zeros")

for_settings <- function(f) {
  for (tie_breaking in c("efron", "breslow")) {
    for (start in c(FALSE, TRUE)) {
      for (weights in WEIGHTS) {
        d <- residual_setting(start, weights)
        cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start,
                               sample_weight = d$w, tie_breaking = tie_breaking)
        f(d, cox, tie_breaking == "efron", start, tie_breaking)
      }
    }
  }
}

test_that("score residuals agree with the direct computation", {
  set.seed(40)
  for_settings(function(d, cox, efron, start, tie_breaking) {
    U <- cox$score_residuals(d$eta, d$X, sample_weight = d$w)
    expect_equal(dim(U), dim(d$X))
    keep <- d$w > 0
    U_ref <- matrix(0, nrow(d$X), ncol(d$X))
    U_ref[keep, ] <- reference_residuals(d$X[keep, ], d$eta[keep], d$event[keep], d$status[keep],
                                         d$start[keep], d$w[keep], efron)
    expect_lt(max(abs(U - U_ref)), 1e-10)
    score <- -cox$compute(d$eta, d$w)$gradient / 2
    expect_lt(max(abs(colSums(U) - drop(crossprod(d$X, score)))), 1e-10)
    G <- stats::rnorm(length(d$eta))
    expect_lt(max(abs(drop(G %*% U) -
                      drop(crossprod(d$X, cox$wild_score(d$eta, G, sample_weight = d$w))))), 1e-10)
    expect_equal(cox$score_residuals(d$eta, d$X[, 1], sample_weight = d$w), U[, 1])
  })
})

test_that("stratified score residuals are those of each stratum", {
  set.seed(41)
  d <- residual_setting(TRUE, "unequal")
  n <- length(d$event)
  strata <- seq_len(n) %% 3
  cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start, strata = strata,
                         sample_weight = d$w)
  U <- cox$score_residuals(d$eta, d$X, sample_weight = d$w)
  for (s in 0:2) {
    k <- strata == s
    single <- CoxDeviance$new(event = d$event[k], status = d$status[k], start = d$start[k],
                              sample_weight = d$w[k])
    expect_lt(max(abs(U[k, ] - single$score_residuals(d$eta[k], d$X[k, ], sample_weight = d$w[k]))),
              1e-12)
  }
  expect_error(cox$score_residuals(d$eta, d$X[-1, ]), "one per observation")
})

test_that("the baseline hazard agrees with the direct computation", {
  set.seed(42)
  for_settings(function(d, cox, efron, start, tie_breaking) {
    H <- cox$baseline_hazard(d$eta, sample_weight = d$w)
    ref <- reference_hazard(d$eta, d$event, d$status, d$start, d$w, efron)
    expect_identical(H$time, ref$time)
    expect_equal(H$hazard, ref$hazard, tolerance = 1e-10)
    expect_equal(H$cumulative_hazard, cumsum(ref$hazard), tolerance = 1e-10)
    expect_null(H$stratum)
    # for the linear predictor as given: a shift by c scales it by exp(-c)
    H2 <- cox$baseline_hazard(d$eta + 1.5, sample_weight = d$w)
    expect_equal(H2$cumulative_hazard, H$cumulative_hazard * exp(-1.5), tolerance = 1e-10)
  })
})

test_that("the stratified baseline hazard is that of each stratum", {
  set.seed(43)
  d <- residual_setting(TRUE, "ones")
  n <- length(d$event)
  strata <- ifelse(seq_len(n) %% 2 == 1, 7, 3)
  cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start, strata = strata)
  H <- cox$baseline_hazard(d$eta)
  expect_equal(sort(unique(H$stratum)), c(3L, 7L))
  for (s in c(3, 7)) {
    k <- strata == s
    single <- CoxDeviance$new(event = d$event[k], status = d$status[k],
                              start = d$start[k])$baseline_hazard(d$eta[k])
    expect_equal(H$time[H$stratum == s], single$time)
    expect_equal(H$cumulative_hazard[H$stratum == s], single$cumulative_hazard, tolerance = 1e-12)
  }
})

test_that("Schoenfeld residuals agree with the direct computation", {
  set.seed(44)
  for_settings(function(d, cox, efron, start, tie_breaking) {
    R <- cox$schoenfeld_residuals(d$eta, d$X, sample_weight = d$w)
    expect_equal(dim(R), dim(d$X))
    R_ref <- reference_schoenfeld(d$X, d$eta, d$event, d$status, d$start, d$w, efron)
    expect_lt(max(abs(R - R_ref)), 1e-10)
    expect_true(all(R[d$status == 0 | d$w == 0, ] == 0))
    score <- -cox$compute(d$eta, d$w)$gradient / 2
    expect_lt(max(abs(drop(crossprod(R, d$w)) - drop(crossprod(d$X, score)))), 1e-10)
    # a constant covariate has no residuals; a vector design gives a vector
    expect_lt(max(abs(cox$schoenfeld_residuals(d$eta, rep(1, length(d$eta)), sample_weight = d$w))),
              1e-12)
    expect_equal(cox$schoenfeld_residuals(d$eta, d$X[, 1], sample_weight = d$w), R[, 1])
    # a product with the information in between leaves them unchanged
    cox$information(d$eta, d$w)$multiply(d$X[, 2])
    expect_equal(cox$schoenfeld_residuals(d$eta, d$X, sample_weight = d$w), R, tolerance = 1e-14)
  })
})

test_that("stratified Schoenfeld residuals are those of each stratum", {
  set.seed(45)
  d <- residual_setting(TRUE, "unequal")
  n <- length(d$event)
  strata <- seq_len(n) %% 3
  cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start, strata = strata,
                         sample_weight = d$w)
  R <- cox$schoenfeld_residuals(d$eta, d$X, sample_weight = d$w)
  for (s in 0:2) {
    k <- strata == s
    single <- CoxDeviance$new(event = d$event[k], status = d$status[k], start = d$start[k],
                              sample_weight = d$w[k])
    expect_lt(max(abs(R[k, ] - single$schoenfeld_residuals(d$eta[k], d$X[k, ],
                                                           sample_weight = d$w[k]))), 1e-12)
  }
  expect_error(cox$schoenfeld_residuals(d$eta, d$X[-1, ]), "one per observation")
})

test_that("expected events agree with the direct computation", {
  set.seed(46)
  for_settings(function(d, cox, efron, start, tie_breaking) {
    E <- cox$expected_events(d$eta, sample_weight = d$w)
    E_ref <- reference_expected(d$eta, d$event, d$status, d$start, d$w, efron)
    expect_equal(E, E_ref, tolerance = 1e-10)
    expect_equal(cox$survival_at_exit(d$eta, sample_weight = d$w), exp(-E_ref), tolerance = 1e-10)
    # for positive weights, w (status - E) is the score -gradient / 2
    keep <- d$w > 0
    score <- -cox$compute(d$eta, d$w)$gradient / 2
    expect_lt(max(abs((d$w * (d$status - E))[keep] - score[keep])), 1e-10)
    # without start times and with Breslow's ties, exp(eta) times the
    # cumulative hazard at the exit time
    if (!start && tie_breaking == "breslow") {
      H <- cox$baseline_hazard(d$eta, sample_weight = d$w)
      at <- findInterval(d$event, H$time)
      cumhaz <- ifelse(at > 0, H$cumulative_hazard[pmax(at, 1)], 0)
      expect_equal(E, exp(d$eta) * cumhaz, tolerance = 1e-10)
    }
  })
})

test_that("stratified expected events are those of each stratum", {
  set.seed(47)
  d <- residual_setting(TRUE, "unequal")
  n <- length(d$event)
  strata <- ifelse(seq_len(n) %% 2 == 1, 7, 3)
  cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start, strata = strata,
                         sample_weight = d$w)
  E <- cox$expected_events(d$eta, sample_weight = d$w)
  for (s in c(3, 7)) {
    k <- strata == s
    single <- CoxDeviance$new(event = d$event[k], status = d$status[k], start = d$start[k],
                              sample_weight = d$w[k])
    expect_equal(E[k], single$expected_events(d$eta[k], sample_weight = d$w[k]), tolerance = 1e-12)
  }
})

test_that("risk-set variances agree with the direct computation", {
  set.seed(48)
  for_settings(function(d, cox, efron, start, tie_breaking) {
    Z <- d$X + 5                                  # far from mean zero
    V <- cox$risk_set_variance(d$eta, Z, sample_weight = d$w)
    V_ref <- reference_variance(Z, d$eta, d$event, d$status, d$start, d$w, efron)
    expect_lt(max(abs(V - V_ref)), 1e-10)
    expect_true(all(V[d$status == 0 | d$w == 0, ] == 0))
    # weighted, they sum to the diagonal of the information
    I <- coef_information(cox, Z, d$eta, d$w)
    expect_equal(drop(crossprod(V, d$w)), diag(I), tolerance = 1e-10)
    # covariances by polarization: their weighted sum is the information
    C01 <- (cox$risk_set_variance(d$eta, Z[, 1] + Z[, 2], sample_weight = d$w) -
            cox$risk_set_variance(d$eta, Z[, 1] - Z[, 2], sample_weight = d$w)) / 4
    expect_equal(sum(d$w * C01), I[1, 2], tolerance = 1e-8)
    # a constant has no variance; a shift changes nothing
    expect_lt(max(abs(cox$risk_set_variance(d$eta, rep(1, length(d$eta)), sample_weight = d$w))),
              1e-12)
    expect_lt(max(abs(cox$risk_set_variance(d$eta, d$X[, 1], sample_weight = d$w) - V[, 1])), 1e-10)
  })
})

test_that("stratified risk-set variances are those of each stratum, also for a sparse Z", {
  set.seed(49)
  d <- residual_setting(TRUE, "unequal")
  n <- length(d$event)
  strata <- seq_len(n) %% 3
  cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start, strata = strata,
                         sample_weight = d$w)
  V <- cox$risk_set_variance(d$eta, d$X, sample_weight = d$w)
  for (s in 0:2) {
    k <- strata == s
    single <- CoxDeviance$new(event = d$event[k], status = d$status[k], start = d$start[k],
                              sample_weight = d$w[k])
    expect_lt(max(abs(V[k, ] - single$risk_set_variance(d$eta[k], d$X[k, ],
                                                        sample_weight = d$w[k]))), 1e-12)
  }
  skip_if_not_installed("Matrix")
  Xs <- d$X * (abs(d$X) > 1)
  expect_equal(unname(cox$risk_set_variance(d$eta, Matrix::Matrix(Xs, sparse = TRUE),
                                            sample_weight = d$w)),
               cox$risk_set_variance(d$eta, Xs, sample_weight = d$w), tolerance = 1e-14)
})
