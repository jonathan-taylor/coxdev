##context("Check zero-weight handling")

# Tests to verify that zero-weight observations don't affect Cox model results.
# When some observations have zero weights, the deviance, gradient, and
# information matrix should match the results computed using only the
# non-zero weight observations.

tol <- 1e-10

# Function to generate weights with some zeros
sample_weights_with_zeros <- function(size, n_zero = NULL) {
  if (is.null(n_zero)) {
    n_zero <- max(1, size %/% 5)
  }
  n_zero <- min(n_zero, size %/% 3)

  weights <- runif(size, 0.5, 2.0)
  zero_idx <- sample(size, n_zero)
  weights[zero_idx] <- 0.0
  weights
}

check_zero_weights <- function(tie_types,
                               tie_breaking,
                               have_start_times,
                               nrep = 3,
                               size = 5,
                               tol = 1e-10) {

  data <- simulate_df(tie_types, nrep, size, noinfo = TRUE)
  n <- nrow(data)

  if (have_start_times) {
    start <- data$start
  } else {
    start <- NA
  }

  # Generate weights with some zeros
  weights <- sample_weights_with_zeros(n)
  nonzero_idx <- weights > 0

  # Linear predictor
  eta <- rnorm(n) * 0.5

  # Full model with zero weights
  cox_full <- make_cox_deviance(
    event = data$event,
    start = start,
    status = data$status,
    weight = weights,
    tie_breaking = tie_breaking
  )
  result_full <- cox_full$coxdev(eta, weights)

  # Subset model (non-zero weights only)
  if (have_start_times) {
    start_subset <- data$start[nonzero_idx]
  } else {
    start_subset <- NA
  }
  cox_subset <- make_cox_deviance(
    event = data$event[nonzero_idx],
    start = start_subset,
    status = data$status[nonzero_idx],
    weight = weights[nonzero_idx],
    tie_breaking = tie_breaking
  )
  result_subset <- cox_subset$coxdev(eta[nonzero_idx], weights[nonzero_idx])

  # Compare deviance
  expect_true(
    all_close(result_full$deviance, result_subset$deviance, rtol = tol),
    info = sprintf("Deviance mismatch for %s, have_start=%s: %f vs %f",
                   tie_breaking, have_start_times, result_full$deviance, result_subset$deviance)
  )

  # Compare saturated log-likelihood
  expect_true(
    all_close(result_full$loglik_sat, result_subset$loglik_sat, rtol = tol),
    info = sprintf("Loglik_sat mismatch for %s: %f vs %f",
                   tie_breaking, result_full$loglik_sat, result_subset$loglik_sat)
  )

  # Compare gradient for non-zero weight observations
  expect_true(
    all_close(result_full$gradient[nonzero_idx], result_subset$gradient, rtol = tol),
    info = sprintf("Gradient mismatch for %s", tie_breaking)
  )

  # Gradient for zero-weight observations should be zero
  expect_true(
    all(abs(result_full$gradient[!nonzero_idx]) < 1e-10),
    info = "Gradient should be zero for zero-weight observations"
  )

  # Compare diagonal Hessian for non-zero weight observations
  expect_true(
    all_close(result_full$diag_hessian[nonzero_idx], result_subset$diag_hessian, rtol = tol),
    info = sprintf("Diagonal Hessian mismatch for %s", tie_breaking)
  )

  # Compare information matrix action on a test vector
  info_full <- cox_full$information(eta, weights)
  info_subset <- cox_subset$information(eta[nonzero_idx], weights[nonzero_idx])

  # Test vector
  v_full <- rnorm(n)
  v_subset <- v_full[nonzero_idx]

  # Information matrix-vector product
  Iv_full <- info_full(v_full)
  Iv_subset <- info_subset(v_subset)

  # The result for non-zero weight observations should match
  expect_true(
    all_close(Iv_full[nonzero_idx], Iv_subset, rtol = tol),
    info = sprintf("Information matrix-vector product mismatch for %s", tie_breaking)
  )
}

# Run tests for various tie patterns and tie-breaking methods
test_that("zero weights work correctly for efron with start times", {
  for (tie_type in all_combos[1:min(5, length(all_combos))]) {
    check_zero_weights(tie_type, "efron", TRUE, nrep = 3, size = 5, tol = tol)
  }
})

test_that("zero weights work correctly for efron without start times", {
  for (tie_type in all_combos[1:min(5, length(all_combos))]) {
    check_zero_weights(tie_type, "efron", FALSE, nrep = 3, size = 5, tol = tol)
  }
})

test_that("zero weights work correctly for breslow with start times", {
  for (tie_type in all_combos[1:min(5, length(all_combos))]) {
    check_zero_weights(tie_type, "breslow", TRUE, nrep = 3, size = 5, tol = tol)
  }
})

test_that("zero weights work correctly for breslow without start times", {
  for (tie_type in all_combos[1:min(5, length(all_combos))]) {
    check_zero_weights(tie_type, "breslow", FALSE, nrep = 3, size = 5, tol = tol)
  }
})
