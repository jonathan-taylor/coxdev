test_that("stratified cox deviance matches survival::coxph", {
  skip_if_not_installed("survival")
  library(survival)

  set.seed(42)
  n <- 100
  x <- rnorm(n)
  event <- sort(rexp(n, exp(0.5 * x)))
  status <- rbinom(n, 1, 0.7)
  strata <- sample(1:3, n, replace = TRUE)

  # Fit with survival::coxph
  surv_fit <- coxph(Surv(event, status) ~ x + strata(strata), method = "efron")
  beta <- coef(surv_fit)
  eta <- x * beta

  # Use coxdev at the MLE
  strat <- make_stratified_cox_deviance(event = event, status = status, strata = strata)
  result <- strat$coxdev(eta)

  loglik_from_coxdev <- result$loglik_sat - result$deviance / 2

  # Log-likelihood should match
  expect_equal(loglik_from_coxdev, surv_fit$loglik[2], tolerance = 1e-10)

  # Gradient at MLE should be zero
  grad_at_mle <- sum(result$gradient * x)
  expect_lt(abs(grad_at_mle), 1e-10)
})

test_that("stratified cox deviance with left truncation matches survival::coxph", {
  skip_if_not_installed("survival")
  library(survival)

  set.seed(123)
  n <- 100
  start <- runif(n, 0, 2)
  event <- start + rexp(n, 0.5)
  status <- rbinom(n, 1, 0.7)
  strata <- sample(1:3, n, replace = TRUE)
  x <- rnorm(n)

  # Fit with survival::coxph
  surv_fit <- coxph(Surv(start, event, status) ~ x + strata(strata), method = "efron")
  beta <- coef(surv_fit)
  eta <- x * beta

  # Use coxdev
  strat <- make_stratified_cox_deviance(
    event = event,
    start = start,
    status = status,
    strata = strata
  )
  result <- strat$coxdev(eta)

  loglik_from_coxdev <- result$loglik_sat - result$deviance / 2

  # Log-likelihood should match
  expect_equal(loglik_from_coxdev, surv_fit$loglik[2], tolerance = 1e-10)

  # Gradient at MLE should be zero
  grad_at_mle <- sum(result$gradient * x)
  expect_lt(abs(grad_at_mle), 1e-10)
})

test_that("stratified cox deviance with Breslow tie-breaking", {
  skip_if_not_installed("survival")
  library(survival)

  set.seed(99)
  n <- 100
  event <- round(runif(n, 1, 20))  # Create ties
  status <- rbinom(n, 1, 0.6)
  strata <- sample(1:2, n, replace = TRUE)
  x <- rnorm(n)

  # Fit with Breslow
  surv_fit <- coxph(Surv(event, status) ~ x + strata(strata), method = "breslow")
  beta <- coef(surv_fit)
  eta <- x * beta

  # Use coxdev with Breslow
  strat <- make_stratified_cox_deviance(
    event = event,
    status = status,
    strata = strata,
    tie_breaking = "breslow"
  )
  result <- strat$coxdev(eta)

  loglik_from_coxdev <- result$loglik_sat - result$deviance / 2

  # Log-likelihood should match
  expect_equal(loglik_from_coxdev, surv_fit$loglik[2], tolerance = 1e-10)
})

test_that("stratified hessian matvec is symmetric", {
  set.seed(42)
  n <- 50
  event <- sort(runif(n, 0, 10))
  status <- rbinom(n, 1, 0.5)
  strata <- sample(1:3, n, replace = TRUE)
  eta <- rnorm(n)

  strat <- make_stratified_cox_deviance(event = event, status = status, strata = strata)

  # Compute to populate buffers
  result <- strat$coxdev(eta)
  matvec <- strat$information(eta)

  # Test symmetry: v' H w = w' H v
  v <- rnorm(n)
  w <- rnorm(n)

  Hv <- matvec(v)
  Hw <- matvec(w)

  expect_equal(sum(w * Hv), sum(v * Hw), tolerance = 1e-10)
})

test_that("stratified information returned correct n_strata and n_total", {
  set.seed(42)
  n <- 100
  event <- sort(runif(n, 0, 10))
  status <- rbinom(n, 1, 0.5)
  strata <- sample(1:5, n, replace = TRUE)

  strat <- make_stratified_cox_deviance(event = event, status = status, strata = strata)

  expect_equal(strat$n_total, n)
  expect_equal(strat$n_strata, length(unique(strata)))
})

# Tests comparing against glmnet with nonzero lambda

test_that("stratified cox deviance matches glmnet at nonzero lambda", {
  skip_if_not_installed("glmnet")
  skip_if_not_installed("survival")
  library(glmnet)
  library(survival)

  set.seed(42)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  event <- rexp(n)
  status <- rbinom(n, 1, 0.7)
  strata <- sample(1:3, n, replace = TRUE)

  # Create stratified survival object for glmnet
  y <- stratifySurv(Surv(event, status), strata)

  # Fit glmnet with several lambda values
  fit <- glmnet(x, y, family = "cox", lambda = c(0.1, 0.05, 0.01))

  # Test at each lambda value
  for (lam in fit$lambda) {
    beta <- as.numeric(coef(fit, s = lam))
    eta <- as.numeric(x %*% beta)

    # Get deviance from glmnet (uses Breslow)
    dev_glmnet <- coxnet.deviance(pred = eta, y = y, weights = rep(1, n), std.weights = FALSE)

    # Get deviance from coxdev (must use Breslow to match glmnet)
    strat <- make_stratified_cox_deviance(
      event = event,
      status = status,
      strata = strata,
      tie_breaking = "breslow"
    )
    result <- strat$coxdev(eta)

    expect_equal(result$deviance, dev_glmnet, tolerance = 1e-10,
                 info = sprintf("Deviance mismatch at lambda = %g", lam))
  }
})

test_that("stratified cox gradient matches glmnet at nonzero lambda", {
  skip_if_not_installed("glmnet")
  skip_if_not_installed("survival")
  library(glmnet)
  library(survival)

  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  event <- rexp(n)
  status <- rbinom(n, 1, 0.7)
  strata <- sample(1:3, n, replace = TRUE)

  # Create stratified survival object for glmnet
  y <- stratifySurv(Surv(event, status), strata)

  # Fit glmnet
  fit <- glmnet(x, y, family = "cox", lambda = c(0.1, 0.05, 0.01))

  # Test at each lambda value
  for (lam in fit$lambda) {
    beta <- as.numeric(coef(fit, s = lam))
    eta <- as.numeric(x %*% beta)

    # Get gradient from glmnet (uses Breslow)
    # glmnet returns gradient of log partial likelihood, coxdev returns gradient of deviance
    # gradient_deviance = -2 * gradient_loglik
    grad_glmnet <- coxgrad(eta, y, w = rep(1, n), std.weights = FALSE, diag.hessian = TRUE)
    grad_glmnet_deviance <- -2 * as.numeric(grad_glmnet)
    diag_hess_glmnet <- -2 * attr(grad_glmnet, "diag_hessian")

    # Get gradient from coxdev (must use Breslow to match glmnet)
    strat <- make_stratified_cox_deviance(
      event = event,
      status = status,
      strata = strata,
      tie_breaking = "breslow"
    )
    result <- strat$coxdev(eta)

    expect_equal(result$gradient, grad_glmnet_deviance, tolerance = 1e-10,
                 info = sprintf("Gradient mismatch at lambda = %g", lam))
    expect_equal(result$diag_hessian, diag_hess_glmnet, tolerance = 1e-10,
                 info = sprintf("Diagonal Hessian mismatch at lambda = %g", lam))
  }
})

test_that("stratified cox with left truncation matches glmnet at nonzero lambda", {
  skip_if_not_installed("glmnet")
  skip_if_not_installed("survival")
  library(glmnet)
  library(survival)

  set.seed(456)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  start <- runif(n, 0, 2)
  event <- start + rexp(n, 0.5)
  status <- rbinom(n, 1, 0.7)
  strata <- sample(1:3, n, replace = TRUE)

  # Create stratified survival object for glmnet with start/stop times
  y <- stratifySurv(Surv(start, event, status), strata)

  # Fit glmnet
  fit <- glmnet(x, y, family = "cox", lambda = c(0.1, 0.05, 0.01))

  # Test at each lambda value
  for (lam in fit$lambda) {
    beta <- as.numeric(coef(fit, s = lam))
    eta <- as.numeric(x %*% beta)

    # Get deviance and gradient from glmnet
    dev_glmnet <- coxnet.deviance(pred = eta, y = y, weights = rep(1, n), std.weights = FALSE)
    grad_glmnet <- coxgrad(eta, y, w = rep(1, n), std.weights = FALSE, diag.hessian = TRUE)
    grad_glmnet_deviance <- -2 * as.numeric(grad_glmnet)

    # Get from coxdev (must use Breslow to match glmnet)
    strat <- make_stratified_cox_deviance(
      event = event,
      start = start,
      status = status,
      strata = strata,
      tie_breaking = "breslow"
    )
    result <- strat$coxdev(eta)

    expect_equal(result$deviance, dev_glmnet, tolerance = 1e-10,
                 info = sprintf("Deviance mismatch at lambda = %g", lam))
    expect_equal(result$gradient, grad_glmnet_deviance, tolerance = 1e-10,
                 info = sprintf("Gradient mismatch at lambda = %g", lam))
  }
})

test_that("stratified cox with weights matches glmnet at nonzero lambda", {
  skip_if_not_installed("glmnet")
  skip_if_not_installed("survival")
  library(glmnet)
  library(survival)

  set.seed(789)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  event <- rexp(n)
  status <- rbinom(n, 1, 0.7)
  strata <- sample(1:3, n, replace = TRUE)
  weights <- runif(n, 0.5, 2)

  # Create stratified survival object for glmnet
  y <- stratifySurv(Surv(event, status), strata)

  # Fit glmnet with weights
  fit <- glmnet(x, y, family = "cox", weights = weights, lambda = c(0.1, 0.05, 0.01))

  # Test at each lambda value
  for (lam in fit$lambda) {
    beta <- as.numeric(coef(fit, s = lam))
    eta <- as.numeric(x %*% beta)

    # Get deviance and gradient from glmnet
    dev_glmnet <- coxnet.deviance(pred = eta, y = y, weights = weights, std.weights = FALSE)
    grad_glmnet <- coxgrad(eta, y, w = weights, std.weights = FALSE, diag.hessian = TRUE)
    grad_glmnet_deviance <- -2 * as.numeric(grad_glmnet)

    # Get from coxdev (must use Breslow to match glmnet)
    strat <- make_stratified_cox_deviance(
      event = event,
      status = status,
      strata = strata,
      tie_breaking = "breslow"
    )
    result <- strat$coxdev(eta, weights)

    expect_equal(result$deviance, dev_glmnet, tolerance = 1e-10,
                 info = sprintf("Deviance mismatch at lambda = %g", lam))
    expect_equal(result$gradient, grad_glmnet_deviance, tolerance = 1e-10,
                 info = sprintf("Gradient mismatch at lambda = %g", lam))
  }
})
