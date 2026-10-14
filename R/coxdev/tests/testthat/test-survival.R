# Comparisons with coxph of the survival package: the deviance, the score of
# the coefficients and their covariance, the inverse of the information.

skip_if_not_installed("survival")

coxph_at <- function(event, status, X, beta, weight, start, ties, strata = NULL) {
  y <- if (is.null(start)) survival::Surv(event, status) else survival::Surv(start, event, status)
  fit <- if (is.null(strata)) {
    survival::coxph(y ~ X, init = beta, weights = weight, ties = ties, robust = FALSE,
                    control = survival::coxph.control(iter.max = 0, timefix = FALSE))
  } else {
    stratum <- as.integer(factor(strata))
    survival::coxph(y ~ X + survival::strata(stratum), init = beta, weights = weight,
                    ties = ties, robust = FALSE,
                    control = survival::coxph.control(iter.max = 0, timefix = FALSE))
  }
  list(D = -2 * fit$loglik[1],
       G = -2 * colSums(survival::coxph.detail(fit)$score),
       cov = stats::vcov(fit))
}

check_coxph <- function(data, tie_breaking, weight, have_start, strata = NULL, tol = 1e-10) {
  n <- nrow(data)
  p <- max(1, n %/% 4)
  X <- matrix(stats::rnorm(n * p), n, p)
  beta <- stats::rnorm(p) / sqrt(n)
  eta <- drop(X %*% beta)
  start <- if (have_start) data$start else NULL

  cox <- CoxDeviance$new(event = data$event, status = data$status, start = start,
                         strata = strata, sample_weight = weight,
                         tie_breaking = tie_breaking)
  C <- cox$compute(eta, weight)
  info <- coef_information(cox, X, eta, weight)
  expect_equal(info, t(info), tolerance = 1e-12)

  keep <- weight > 0
  R <- coxph_at(data$event[keep], data$status[keep], X[keep, , drop = FALSE], beta,
                weight[keep], if (have_start) start[keep], tie_breaking,
                if (!is.null(strata)) strata[keep])
  expect_equal(C$deviance - 2 * C$loglik_sat, R$D, tolerance = tol)
  expect_lt(rel_diff(drop(crossprod(X, C$gradient)), unname(R$G)), tol)
  expect_lt(rel_diff(solve(info), unname(R$cov)), 1e-8)
}

test_that("deviance, score and covariance agree with coxph", {
  set.seed(1)
  for (combo in all_combos[sample(length(all_combos), 25)]) {
    for (tie_breaking in c("efron", "breslow")) {
      for (weights in list(just_ones, sample_weights, sample_weights_zeros)) {
        for (have_start in c(TRUE, FALSE)) {
          data <- simulate_df(combo, nrep = 3, size = 5)
          check_coxph(data, tie_breaking, weights(nrow(data)), have_start)
        }
      }
    }
  }
})

test_that("stratified models agree with coxph", {
  set.seed(2)
  for (tie_breaking in c("efron", "breslow")) {
    for (have_start in c(TRUE, FALSE)) {
      for (n_strata in c(2, 3, 5)) {
        data <- simulate_df(all_combos[[length(all_combos)]], nrep = 2, size = 5)
        n <- nrow(data)
        strata <- sample(n_strata, n, replace = TRUE)
        check_coxph(data, tie_breaking, sample_weights(n), have_start, strata = strata)
        # strata given as labels
        check_coxph(data, tie_breaking, sample_weights(n), have_start,
                    strata = letters[strata])
      }
    }
  }
})
