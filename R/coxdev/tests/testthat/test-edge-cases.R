# Edge cases: no failures, one failure, everyone failing at once, entries at
# failure times, zero weights everywhere, strata without failures, extreme
# linear predictors; and finite-difference checks of the derivatives under
# every option (tests/test_diag_hessian.py checks diag_hessian so).

coxph_check <- function(event, status, X, beta, weight = NULL, start = NULL, strata = NULL,
                        ties = "efron", tol = 1e-10) {
  n <- length(event)
  weight <- if (is.null(weight)) rep(1, n) else weight
  cox <- CoxDeviance$new(event = event, status = status, start = start, strata = strata,
                         tie_breaking = ties)
  eta <- drop(X %*% beta)
  C <- cox$compute(eta, weight)
  y <- if (is.null(start)) survival::Surv(event, status) else survival::Surv(start, event, status)
  ctrl <- survival::coxph.control(iter.max = 0, timefix = FALSE)
  fit <- if (is.null(strata)) {
    survival::coxph(y ~ X, init = beta, weights = weight, ties = ties, robust = FALSE,
                    control = ctrl)
  } else {
    stratum <- strata
    survival::coxph(y ~ X + survival::strata(stratum), init = beta, weights = weight,
                    ties = ties, robust = FALSE, control = ctrl)
  }
  expect_equal(C$deviance - 2 * C$loglik_sat, -2 * fit$loglik[1], tolerance = tol)
  # coxph's score residuals are per unit weight (coxph.detail fails with a
  # single failure time)
  G <- -2 * colSums(weight * stats::residuals(fit, type = "score"))
  expect_equal(drop(crossprod(X, C$gradient)), unname(G), tolerance = tol)
  expect_equal(coef_information(cox, X, eta, weight), unname(solve(stats::vcov(fit))),
               tolerance = 1e-8)
  invisible(C)
}

test_that("without failures everything is zero", {
  set.seed(70)
  n <- 10
  event <- ceiling(stats::rexp(n) * 3) / 3
  eta <- stats::rnorm(n)
  for (tie_breaking in c("efron", "breslow")) {
    for (start in list(NULL, event / 2)) {
      cox <- CoxDeviance$new(event = event, status = rep(0, n), start = start,
                             tie_breaking = tie_breaking)
      r <- cox$compute(eta)
      expect_identical(c(r$deviance, r$loglik_sat), c(0, 0))
      expect_true(all(r$gradient == 0) && all(r$diag_hessian == 0))
      expect_true(all(cox$information(eta)$to_dense() == 0))
      expect_equal(nrow(cox$baseline_hazard(eta)), 0L)
      expect_true(all(cox$expected_events(eta) == 0))
      expect_true(all(cox$survival_at_exit(eta) == 1))
      X <- matrix(stats::rnorm(2 * n), n, 2)
      expect_true(all(cox$score_residuals(eta, X) == 0))
      expect_true(all(cox$schoenfeld_residuals(eta, X) == 0))
      expect_true(all(cox$risk_set_variance(eta, X) == 0))
      expect_true(is.nan(cox$c_index(eta)))
      for (method in c("multinomial", "bernoulli", "fisher", "wild")) {
        expect_true(all(cox$sample_score(eta, method = method, n_draws = 3, seed = 1) == 0))
      }
    }
  }
})

test_that("zero weights everywhere give zero", {
  event <- c(2, 3, 3, 5, 6)
  cox <- CoxDeviance$new(event = event, status = c(1, 1, 0, 1, 1))
  r <- cox$compute(c(0.1, -0.2, 0.3, 0, 0.5), rep(0, 5))
  expect_identical(c(r$deviance, r$loglik_sat), c(0, 0))
  expect_true(all(r$gradient == 0) && all(r$diag_hessian == 0))
})

test_that("one observation, and one failure", {
  r <- CoxDeviance$new(event = 1, status = 1)$compute(0.7)
  expect_identical(c(r$deviance, r$gradient, r$diag_hessian), c(0, 0, 0))

  # one failure, at time 3, with 2, 3, 4 and 5 at risk: the log partial
  # likelihood is eta_3 - log sum over the risk set of exp(eta)
  event <- c(2, 3, 3, 5, 6)
  status <- c(0, 0, 1, 0, 0)
  eta <- c(0.1, -0.2, 0.3, 0, 0.5)
  for (tie_breaking in c("efron", "breslow")) {
    r <- CoxDeviance$new(event = event, status = status, tie_breaking = tie_breaking)$compute(eta)
    risk <- event >= 3
    p <- exp(eta) * risk / sum(exp(eta[risk]))
    expect_equal(r$loglik_sat, 0)
    expect_equal(r$deviance, -2 * (eta[3] - log(sum(exp(eta[risk])))), tolerance = 1e-12)
    expect_equal(r$gradient, -2 * (status - p), tolerance = 1e-12)
    expect_equal(r$diag_hessian, 2 * p * (1 - p), tolerance = 1e-12)
  }
})

test_that("ties of every failure, and entries at failure times, agree with coxph", {
  skip_if_survival_broken()   # see helper-survival.R
  skip_if_not_installed("survival")
  set.seed(71)
  for (ties in c("efron", "breslow")) {
    # everyone fails at once
    n <- 6
    X <- matrix(stats::rnorm(2 * n), n, 2)
    coxph_check(rep(1, n), rep(1, n), X, c(0.3, -0.4), ties = ties)
    # entries at failure times: one entering at 2 is not at risk at 2
    event <- c(2, 2, 3, 4, 4, 5, 6, 3)
    start <- c(0, 1, 2, 2, 3, 2, 4, 0)
    status <- c(1, 1, 1, 0, 1, 1, 0, 1)
    X <- matrix(stats::rnorm(2 * 8), 8, 2)
    coxph_check(event, status, X, c(0.5, -0.2), weight = stats::runif(8, 0.5, 2),
                start = start, ties = ties)
  }
})

test_that("strata without failures, and of one observation, agree with coxph", {
  skip_if_survival_broken()   # see helper-survival.R
  skip_if_not_installed("survival")
  set.seed(72)
  n <- 40
  event <- ceiling(stats::rexp(n) * 3) / 3
  status <- stats::rbinom(n, 1, 0.7)
  strata <- rep(1:4, c(15, 15, 9, 1))
  status[strata == 3] <- 0
  X <- matrix(stats::rnorm(2 * n), n, 2)
  for (ties in c("efron", "breslow")) {
    C <- coxph_check(event, status, X, c(0.3, -0.4), weight = stats::runif(n, 0.5, 2),
                     strata = strata, ties = ties)
    expect_true(all(C$gradient[strata == 3] == 0))
  }
})

test_that("extreme linear predictors change nothing", {
  set.seed(73)
  n <- 30
  event <- ceiling(stats::rexp(n) * 3) / 3
  status <- stats::rbinom(n, 1, 0.7)
  eta <- stats::rnorm(n)
  for (tie_breaking in c("efron", "breslow")) {
    cox <- CoxDeviance$new(event = event, status = status, start = floor(event * 3 * stats::runif(n)) / 3,
                           tie_breaking = tie_breaking)
    r <- cox$compute(eta)
    I <- cox$information(eta)$to_dense()
    for (shift in c(-800, 800)) {
      s <- cox$compute(eta + shift)
      expect_equal(s$deviance, r$deviance, tolerance = 1e-10)
      expect_equal(s$gradient, r$gradient, tolerance = 1e-10)
      expect_equal(s$diag_hessian, r$diag_hessian, tolerance = 1e-10)
      expect_equal(cox$information(eta + shift)$to_dense(), I, tolerance = 1e-10)
    }
  }
})

test_that("the gradient and information are the derivatives, under every option", {
  set.seed(74)
  h <- 1e-6
  for (tie_breaking in c("efron", "breslow")) {
    for (start in c(FALSE, TRUE)) {
      for (strata in c(FALSE, TRUE)) {
        for (weight in c("unequal", "zeros")) {
          m <- make_cox(n = 25, start = start, strata = strata, weight = weight,
                        tie_breaking = tie_breaking)
          r <- m$cox$compute(m$eta, m$w)
          D <- m$cox$information(m$eta, m$w)$to_dense()
          fd_dev <- fd_grad <- NULL
          for (i in seq_along(m$eta)) {
            e <- m$eta
            e[i] <- e[i] + h
            plus <- m$cox$compute(e, m$w)
            e[i] <- e[i] - 2 * h
            minus <- m$cox$compute(e, m$w)
            fd_dev <- c(fd_dev, (plus$deviance - minus$deviance) / (2 * h))
            fd_grad <- cbind(fd_grad, (plus$gradient - minus$gradient) / (2 * h))
          }
          expect_lt(max(abs(fd_dev - r$gradient)), 1e-6)
          expect_lt(max(abs(fd_grad / 2 - D)), 1e-6)
          # diag_hessian is the diagonal of the Hessian of the deviance
          expect_lt(max(abs(diag(fd_grad) - r$diag_hessian)), 1e-6)
        }
      }
    }
  }
})
