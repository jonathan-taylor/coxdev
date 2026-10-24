# Strata, as tests/test_stratified.py: a stratified model is the sum of a
# model per stratum, its information is block diagonal, one stratum is no
# strata, and strata may be any labels. test-survival.R compares stratified
# models with coxph.

strat_data <- function(n = 20) {
  start <- stats::runif(n, 0, 5)
  list(status = sample(0:1, n, replace = TRUE),
       start = start,
       event = stats::runif(n, 0, 10) + start,
       eta = stats::rnorm(n),
       weight = stats::runif(n, 0.5, 2),
       n = n)
}

multi_strata <- function() sample(c(rep(0L, 7), rep(1L, 7), rep(2L, 6)))

expect_same_fit <- function(a, b, eta, w, tol = 1e-12) {
  ra <- a$compute(eta, w)
  rb <- b$compute(eta, w)
  expect_equal(ra$deviance, rb$deviance, tolerance = tol)
  expect_equal(ra$loglik_sat, rb$loglik_sat, tolerance = tol)
  expect_equal(ra$gradient, rb$gradient, tolerance = tol)
  expect_equal(ra$diag_hessian, rb$diag_hessian, tolerance = tol)
  v <- stats::rnorm(length(eta))
  expect_equal(a$information(eta, w)$multiply(v), b$information(eta, w)$multiply(v),
               tolerance = tol)
}

test_that("one stratum is the unstratified model", {
  set.seed(42)
  d <- strat_data()
  for (tie_breaking in c("efron", "breslow")) {
    for (start in list(NULL, d$start)) {
      a <- CoxDeviance$new(event = d$event, status = d$status, start = start,
                           tie_breaking = tie_breaking)
      b <- CoxDeviance$new(event = d$event, status = d$status, start = start,
                           strata = rep(0L, d$n), tie_breaking = tie_breaking)
      expect_true(b$stratified)
      expect_same_fit(a, b, d$eta, d$weight)
      expect_same_fit(a, b, d$eta, NULL)
    }
  }
})

test_that("the stratified model is the sum of the models of its strata", {
  set.seed(43)
  d <- strat_data()
  strata <- multi_strata()
  for (tie_breaking in c("efron", "breslow")) {
    for (start in list(NULL, d$start)) {
      cox <- CoxDeviance$new(event = d$event, status = d$status, start = start, strata = strata,
                             tie_breaking = tie_breaking)
      r <- cox$compute(d$eta, d$weight)
      deviance <- loglik_sat <- 0
      gradient <- diag_hessian <- numeric(d$n)
      for (s in 0:2) {
        k <- strata == s
        single <- CoxDeviance$new(event = d$event[k], status = d$status[k], start = start[k],
                                  tie_breaking = tie_breaking)
        rs <- single$compute(d$eta[k], d$weight[k])
        deviance <- deviance + rs$deviance
        loglik_sat <- loglik_sat + rs$loglik_sat
        gradient[k] <- rs$gradient
        diag_hessian[k] <- rs$diag_hessian
      }
      expect_equal(r$deviance, deviance, tolerance = 1e-12)
      expect_equal(r$loglik_sat, loglik_sat, tolerance = 1e-12)
      expect_equal(r$gradient, gradient, tolerance = 1e-12)
      expect_equal(r$diag_hessian, diag_hessian, tolerance = 1e-12)
    }
  }
})

test_that("the information is block diagonal, with the blocks of the strata", {
  # three strata of sizes 15, 13 and 23, not contiguous
  set.seed(2024)
  sizes <- c(15, 13, 23)
  n <- sum(sizes)
  strata <- sample(rep(0:2, sizes))
  start <- stats::runif(n, 0, 5)
  event <- stats::runif(n, 0, 10) + start
  status <- sample(0:1, n, replace = TRUE)
  eta <- stats::rnorm(n)
  weight <- stats::runif(n, 0.5, 2)

  cox <- CoxDeviance$new(event = event, status = status, start = start, strata = strata)
  I <- cox$information(eta, weight)
  D <- I$to_dense()
  expect_equal(D, t(D), tolerance = 1e-12)
  v <- stats::rnorm(n)
  u <- stats::rnorm(n)
  expect_equal(sum(v * I$multiply(u)), sum(u * I$multiply(v)), tolerance = 1e-10)
  expected <- numeric(n)
  for (s in 0:2) {
    k <- strata == s
    block <- CoxDeviance$new(event = event[k], status = status[k],
                             start = start[k])$information(eta[k], weight[k])
    expected[k] <- block$multiply(v[k])
    expect_equal(D[k, k], block$to_dense(), tolerance = 1e-12)
    # no interaction between strata
    expect_true(all(D[k, !k] == 0))
    vs <- ifelse(k, v, 0)
    out <- I$multiply(vs)
    expect_true(all(out[!k] == 0))
    expect_equal(out[k], block$multiply(v[k]), tolerance = 1e-12)
  }
  expect_equal(I$multiply(v), expected, tolerance = 1e-12)
})

test_that("strata given as labels fit the model of their codes", {
  set.seed(5)
  n <- 120
  event <- ceiling(stats::rexp(n) * 4) / 4
  status <- stats::rbinom(n, 1, 0.7)
  eta <- stats::rnorm(n) * 0.5
  codes <- sample(0:2, n, replace = TRUE)
  by_code <- CoxDeviance$new(event = event, status = status, strata = codes)
  r_code <- by_code$compute(eta)
  H_code <- by_code$baseline_hazard(eta)
  for (names in list(c("north", "south", "west"), c(-1.5, 0.25, 7), c(10L, 20L, 30L))) {
    labels <- names[codes + 1]
    for (strata in list(labels, factor(labels))) {
      by_label <- CoxDeviance$new(event = event, status = status, strata = strata)
      r_label <- by_label$compute(eta)
      expect_equal(r_label$deviance, r_code$deviance, tolerance = 1e-14)
      expect_equal(r_label$gradient, r_code$gradient, tolerance = 1e-14)
      H_label <- by_label$baseline_hazard(eta)
      expect_equal(H_label$cumulative_hazard, H_code$cumulative_hazard, tolerance = 1e-14)
      expect_equal(as.character(H_label$stratum), as.character(names[H_code$stratum + 1]))
      expect_setequal(as.character(H_label$stratum), as.character(names))
    }
  }
})

test_that("stratified models with ties, small and large weights", {
  set.seed(6)
  d <- strat_data()
  strata <- multi_strata()
  tied <- rep(c(1, 1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 6), 2)[1:d$n]
  for (tie_breaking in c("efron", "breslow")) {
    cox <- CoxDeviance$new(event = tied, status = d$status, strata = strata,
                           tie_breaking = tie_breaking)
    r <- cox$compute(d$eta, d$weight)
    expect_gt(r$deviance, 0)
    expect_length(r$gradient, d$n)
    expect_length(cox$information(d$eta, d$weight)$multiply(stats::rnorm(d$n)), d$n)
    # the deviance, gradient and information scale with a constant weight
    for (c in c(0.001, 1000)) {
      rc <- cox$compute(d$eta, rep(c, d$n))
      r1 <- cox$compute(d$eta, rep(1, d$n))
      expect_equal(rc$deviance, c * r1$deviance, tolerance = 1e-10)
      expect_equal(rc$gradient, c * r1$gradient, tolerance = 1e-10)
      expect_equal(rc$diag_hessian, c * r1$diag_hessian, tolerance = 1e-10)
    }
  }
})

test_that("stratified inputs are checked", {
  set.seed(7)
  d <- strat_data()
  expect_error(CoxDeviance$new(event = d$event, status = d$status, strata = c(0, 1)),
               "strata must have length 20")
  expect_error(CoxDeviance$new(event = d$event, status = d$status, strata = c(NaN, rep(0, 19))),
               "missing")
  status <- d$status
  status[1] <- 1.2
  expect_error(CoxDeviance$new(event = d$event, status = status, strata = rep(0, 20)), "binary")
  expect_error(CoxDeviance$new(event = d$event, status = rep(c(0.5, 1.5, 0, 1), 5)), "binary")
})

test_that("strata of unequal sizes agree with coxph", {
  skip_if_survival_broken()   # see helper-survival.R
  skip_if_not_installed("survival")
  # the data of create_stratified_data in tests/test_compareR.py, with
  # continuous times
  set.seed(42)
  n <- 200
  stratum <- rep(0:2, c(50, 80, 70))
  status <- stats::rbinom(n, 1, 0.7)
  start <- stats::rexp(n, 2)
  event <- stats::rexp(n) + start
  X <- matrix(stats::rnorm(n * 3), n, 3)
  beta <- stats::rnorm(3) / sqrt(n)
  weight <- stats::runif(n, 0.5, 2)
  eta <- drop(X %*% beta)
  for (tie_breaking in c("efron", "breslow")) {
    for (s in list(NULL, start)) {
      cox <- CoxDeviance$new(event = event, status = status, start = s, strata = stratum,
                             tie_breaking = tie_breaking)
      C <- cox$compute(eta, weight)
      y <- if (is.null(s)) survival::Surv(event, status) else survival::Surv(s, event, status)
      fit <- survival::coxph(y ~ X + survival::strata(stratum), init = beta, weights = weight,
                             ties = tie_breaking, robust = FALSE,
                             control = survival::coxph.control(iter.max = 0, timefix = FALSE))
      expect_equal(C$deviance - 2 * C$loglik_sat, -2 * fit$loglik[1], tolerance = 1e-10)
      G <- -2 * colSums(survival::coxph.detail(fit)$score)
      expect_lt(rel_diff(drop(crossprod(X, C$gradient)), unname(G)), 1e-10)
      expect_lt(rel_diff(solve(coef_information(cox, X, eta, weight)), unname(stats::vcov(fit))),
                1e-10)
    }
  }
})
