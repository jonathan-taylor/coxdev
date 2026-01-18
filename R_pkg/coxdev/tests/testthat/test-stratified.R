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
