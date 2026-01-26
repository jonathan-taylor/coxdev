# Tests for IRLS/coordinate descent interface

test_that("make_cox_irls_state creates valid state object", {
  set.seed(42)
  n <- 50
  time <- rexp(n)
  status <- rbinom(n, 1, 0.7)

  cox <- make_cox_deviance(event = time, status = status)
  irls <- make_cox_irls_state(cox)

  expect_type(irls, "list")
  expect_true(all(c("recompute_outer", "working_weights", "working_response",
                    "residuals", "current_deviance", "weighted_inner_product",
                    "update_residuals", "reset_residuals") %in% names(irls)))
})

test_that("recompute_outer returns deviance and computes working quantities", {
  set.seed(42)
  n <- 50
  time <- rexp(n)
  status <- rbinom(n, 1, 0.7)

  cox <- make_cox_deviance(event = time, status = status)
  irls <- make_cox_irls_state(cox)

  eta <- rnorm(n, sd = 0.5)
  deviance <- irls$recompute_outer(eta)

  expect_type(deviance, "double")
  expect_true(is.finite(deviance))
  expect_true(deviance >= 0)

  # Compare with direct coxdev call
  result <- cox$coxdev(eta)
  expect_equal(deviance, result$deviance, tolerance = 1e-10)
})

test_that("working_weights are positive and match -diag_hessian/2", {
  set.seed(42)
  n <- 50
  time <- rexp(n)
  status <- rbinom(n, 1, 0.7)

  cox <- make_cox_deviance(event = time, status = status)
  irls <- make_cox_irls_state(cox)

  eta <- rnorm(n, sd = 0.5)
  irls$recompute_outer(eta)

  w <- irls$working_weights()
  expect_length(w, n)
  expect_true(all(w >= 0))

  # Compare with direct computation
  result <- cox$coxdev(eta)
  expected_w <- result$diag_hessian / 2  # diag_hessian is negative, / 2 gives positive
  expect_equal(w, expected_w, tolerance = 1e-10)
})

test_that("working_response satisfies z = eta - grad/diag_hess", {
  set.seed(42)
  n <- 50
  time <- rexp(n)
  status <- rbinom(n, 1, 0.7)

  cox <- make_cox_deviance(event = time, status = status)
  irls <- make_cox_irls_state(cox)

  eta <- rnorm(n, sd = 0.5)
  irls$recompute_outer(eta)

  z <- irls$working_response()
  expect_length(z, n)

  # Compare with direct computation
  result <- cox$coxdev(eta)
  expected_z <- eta - result$gradient / result$diag_hessian
  # Handle near-zero hessian entries
  safe_idx <- abs(result$diag_hessian) > 1e-10
  expect_equal(z[safe_idx], expected_z[safe_idx], tolerance = 1e-10)
})

test_that("residuals are r = w * (z - eta)", {
  set.seed(42)
  n <- 50
  time <- rexp(n)
  status <- rbinom(n, 1, 0.7)

  cox <- make_cox_deviance(event = time, status = status)
  irls <- make_cox_irls_state(cox)

  eta <- rnorm(n, sd = 0.5)
  irls$recompute_outer(eta)

  r <- irls$residuals()
  w <- irls$working_weights()
  z <- irls$working_response()

  expect_length(r, n)
  expect_equal(r, w * (z - eta), tolerance = 1e-10)
})

test_that("weighted_inner_product computes correct gradient and hessian", {
  set.seed(42)
  n <- 50
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(1, -0.5, 0.3)
  eta_true <- X %*% beta_true
  time <- rexp(n, exp(eta_true))
  status <- rbinom(n, 1, 0.7)

  cox <- make_cox_deviance(event = time, status = status)
  irls <- make_cox_irls_state(cox)

  eta <- rep(0, n)
  irls$recompute_outer(eta)

  w <- irls$working_weights()
  r <- irls$residuals()

  for (j in 1:p) {
    x_j <- X[, j]
    gh <- irls$weighted_inner_product(x_j)

    expect_type(gh, "double")
    expect_length(gh, 2)
    expect_true("gradient" %in% names(gh))
    expect_true("hessian" %in% names(gh))

    # Verify formulas
    expected_grad <- sum(x_j * r)
    expected_hess <- sum(w * x_j^2)

    expect_equal(unname(gh["gradient"]), expected_grad, tolerance = 1e-10)
    expect_equal(unname(gh["hessian"]), expected_hess, tolerance = 1e-10)
  }
})

test_that("update_residuals correctly updates r -= delta * w * x_j", {
  set.seed(42)
  n <- 50
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  time <- rexp(n)
  status <- rbinom(n, 1, 0.7)

  cox <- make_cox_deviance(event = time, status = status)
  irls <- make_cox_irls_state(cox)

  eta <- rep(0, n)
  irls$recompute_outer(eta)

  w <- irls$working_weights()
  r_before <- irls$residuals()

  # Update for coordinate 1
  delta <- 0.5
  x_j <- X[, 1]
  irls$update_residuals(delta, x_j)

  r_after <- irls$residuals()
  expected_r <- r_before - delta * w * x_j

  expect_equal(r_after, expected_r, tolerance = 1e-10)
})

test_that("reset_residuals resets to w * (z - eta_current)", {
  set.seed(42)
  n <- 50
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  time <- rexp(n)
  status <- rbinom(n, 1, 0.7)

  cox <- make_cox_deviance(event = time, status = status)
  irls <- make_cox_irls_state(cox)

  eta0 <- rep(0, n)
  irls$recompute_outer(eta0)

  w <- irls$working_weights()
  z <- irls$working_response()

  # Do some updates
  for (j in 1:p) {
    irls$update_residuals(0.1, X[, j])
  }

  # Reset with new eta
  eta_new <- X %*% c(0.1, 0.1, 0.1)
  irls$reset_residuals(eta_new)

  r <- irls$residuals()
  expected_r <- w * (z - eta_new)

  expect_equal(r, as.vector(expected_r), tolerance = 1e-10)
})

test_that("one coordinate descent pass decreases deviance", {
  set.seed(42)
  n <- 100
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(1, -0.5, 0.3, 0, 0)
  eta_true <- X %*% beta_true
  time <- rexp(n, exp(eta_true))
  status <- rbinom(n, 1, 0.7)

  cox <- make_cox_deviance(event = time, status = status)
  irls <- make_cox_irls_state(cox)

  # Initialize
  beta <- rep(0, p)
  eta <- X %*% beta

  # Outer iteration
  dev_before <- irls$recompute_outer(eta)

  # One CD pass
  for (j in 1:p) {
    x_j <- X[, j]
    gh <- irls$weighted_inner_product(x_j)
    if (gh["hessian"] > 1e-10) {
      delta <- gh["gradient"] / gh["hessian"]
      beta[j] <- beta[j] + delta
      irls$update_residuals(delta, x_j)
    }
  }

  # Recompute
  eta <- X %*% beta
  dev_after <- irls$recompute_outer(eta)

  expect_true(dev_after < dev_before)
})

test_that("IRLS converges to reasonable solution", {
  set.seed(42)
  n <- 100
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(0.5, -0.3, 0.2)
  eta_true <- X %*% beta_true
  time <- rexp(n, exp(eta_true))
  status <- rbinom(n, 1, 0.8)

  cox <- make_cox_deviance(event = time, status = status)
  irls <- make_cox_irls_state(cox)

  # Initialize
  beta <- rep(0, p)
  eta <- X %*% beta

  # Multiple outer iterations
  deviances <- numeric(10)
  for (outer in 1:10) {
    irls$recompute_outer(eta)
    deviances[outer] <- irls$current_deviance()

    # Multiple CD passes
    for (pass in 1:3) {
      for (j in 1:p) {
        x_j <- X[, j]
        gh <- irls$weighted_inner_product(x_j)
        if (gh["hessian"] > 1e-10) {
          delta <- gh["gradient"] / gh["hessian"]
          beta[j] <- beta[j] + delta
          irls$update_residuals(delta, x_j)
        }
      }
      eta <- X %*% beta
      irls$reset_residuals(eta)
    }

    eta <- X %*% beta
  }

  # Check convergence
  expect_true(all(diff(deviances) <= 0))  # Monotonically decreasing
  expect_true(deviances[10] < deviances[1])  # Final < initial

  # Check coefficients are reasonable (correct sign at least)
  expect_true(sign(beta[1]) == sign(beta_true[1]))
})

test_that("stratified IRLS state works", {
  set.seed(42)
  n <- 100
  time <- rexp(n)
  status <- rbinom(n, 1, 0.7)
  strata <- rep(1:2, each = n / 2)

  cox <- make_stratified_cox_deviance(event = time, status = status, strata = strata)
  irls <- make_cox_irls_state(cox)

  eta <- rnorm(n, sd = 0.5)
  deviance <- irls$recompute_outer(eta)

  expect_true(is.finite(deviance))
  expect_true(deviance >= 0)

  w <- irls$working_weights()
  expect_length(w, n)
  expect_true(all(w >= 0))
})

test_that("IRLS with sample weights works correctly", {
  set.seed(42)
  n <- 50
  time <- rexp(n)
  status <- rbinom(n, 1, 0.7)
  weights <- runif(n, 0.5, 2.0)

  cox <- make_cox_deviance(event = time, status = status, sample_weight = weights)
  irls <- make_cox_irls_state(cox)

  eta <- rnorm(n, sd = 0.5)
  deviance <- irls$recompute_outer(eta)

  expect_true(is.finite(deviance))

  # Verify sample weights were used
  result <- cox$coxdev(eta)
  expect_equal(deviance, result$deviance, tolerance = 1e-10)
})
