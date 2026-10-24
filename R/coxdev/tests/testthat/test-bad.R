# The problematic case of tests/test_bad.py and tests/test_bad.R: 50
# observations with continuous times and 10 covariates, against coxph and
# glmnet.

problematic_data <- function() {
  set.seed(42)
  n <- 50
  p <- 10
  list(event = stats::rexp(n, rate = 0.5),
       status = sample(c(0, 1), n, replace = TRUE, prob = c(0.3, 0.7)),
       sample_weight = stats::runif(n, 0.5, 2),
       X = matrix(stats::rnorm(n * p), n, p),
       beta = stats::rnorm(p) * 0.1)
}

test_that("the result is complete and finite", {
  d <- problematic_data()
  cox <- CoxDeviance$new(event = d$event, status = d$status)
  eta <- drop(d$X %*% d$beta)
  C <- cox$compute(eta, d$sample_weight)
  expect_true(is.finite(C$deviance) && C$deviance > 0)
  expect_length(C$gradient, 50)
  expect_length(C$diag_hessian, 50)
  expect_true(all(is.finite(C$gradient)) && all(is.finite(C$diag_hessian)))
})

test_that("the information of the coefficients is symmetric positive definite", {
  d <- problematic_data()
  cox <- CoxDeviance$new(event = d$event, status = d$status)
  I <- coef_information(cox, d$X, drop(d$X %*% d$beta), d$sample_weight)
  expect_equal(dim(I), c(10L, 10L))
  expect_equal(I, t(I), tolerance = 1e-10)
  expect_true(all(eigen(I, symmetric = TRUE, only.values = TRUE)$values > 0))
})

test_that("the problematic case agrees with coxph", {
  skip_if_not_installed("survival")
  d <- problematic_data()
  eta <- drop(d$X %*% d$beta)
  for (tie_breaking in c("efron", "breslow")) {
    cox <- CoxDeviance$new(event = d$event, status = d$status, tie_breaking = tie_breaking)
    C <- cox$compute(eta, d$sample_weight)
    X <- d$X
    fit <- survival::coxph(survival::Surv(d$event, d$status) ~ X, init = d$beta,
                           weights = d$sample_weight, ties = tie_breaking, robust = FALSE,
                           control = survival::coxph.control(iter.max = 0, timefix = FALSE))
    expect_equal(C$deviance - 2 * C$loglik_sat, -2 * fit$loglik[1], tolerance = 1e-10)
    G <- -2 * colSums(survival::coxph.detail(fit)$score)
    expect_lt(rel_diff(drop(crossprod(d$X, C$gradient)), unname(G)), 1e-10)
    cov <- solve(coef_information(cox, d$X, eta, d$sample_weight))
    expect_lt(rel_diff(unname(stats::vcov(fit)), cov), 1e-10)
  }
})

test_that("the problematic case agrees with glmnet", {
  skip_if_not_installed("survival")
  skip_if_not_installed("glmnet", minimum_version = "5.0")
  d <- problematic_data()
  eta <- drop(d$X %*% d$beta)
  cox <- CoxDeviance$new(event = d$event, status = d$status, tie_breaking = "breslow")
  C <- cox$compute(eta, d$sample_weight)
  y <- survival::Surv(d$event, d$status)
  D_R <- glmnet::coxnet.deviance(pred = eta, y = y, weights = d$sample_weight,
                                 std.weights = FALSE, cox.ties = "breslow")
  G_R <- glmnet:::coxgrad(eta, y, d$sample_weight, std.weights = FALSE, diag.hessian = TRUE,
                          cox.ties = "breslow")
  expect_lt(abs(D_R - C$deviance) / abs(D_R), 1e-10)
  expect_lt(rel_diff(C$gradient, -2 * as.numeric(G_R)), 1e-10)
  expect_lt(rel_diff(C$diag_hessian, attr(G_R, "diag_hessian")), 1e-10)
})
