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
    sample_weight = weights,
    tie_breaking = tie_breaking
  )
  result_full <- cox_full$coxdev(eta)

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
    sample_weight = weights[nonzero_idx],
    tie_breaking = tie_breaking
  )
  result_subset <- cox_subset$coxdev(eta[nonzero_idx])

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
  info_full <- cox_full$information(eta)
  info_subset <- cox_subset$information(eta[nonzero_idx])

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

# Regression test for bug where first observation in event order has zero weight
# This tests that the stratified C++ implementation correctly handles the case
# where effective_cluster_sizes(0) = 0 because the first event-ordered observation
# has zero weight. Previously, the code checked effective_cluster_sizes(0) > 0
# to decide whether to use zero-weight handling, which failed in this case.
test_that("zero weight at first event-ordered position works correctly", {
  set.seed(42)  # For reproducibility

  # Create data where first observation in event order will have zero weight
  # Use ties to ensure Efron correction is needed
  n <- 20
  event <- c(1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
  start <- event - runif(n, 0.1, 0.5)
  status <- as.integer(c(1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1))

  # Get the event order (order by event time, then by status descending for ties)
  # This matches the preprocessing logic: events before censoring at same time
  event_order <- order(event, -status) - 1L  # -1L for 0-based indexing to match C++

  # Create weights where the first observation in event order has zero weight
  weights <- rep(1.0, n)
  weights[event_order[1] + 1L] <- 0  # +1 for R 1-based indexing

  # Verify our setup: first event-ordered observation should have zero weight
  w_event <- weights[event_order + 1L]
  expect_equal(w_event[1], 0, info = "First event-ordered observation should have zero weight")

  eta <- rnorm(n) * 0.5

  # Create cox deviance with Efron (which exercises the problematic code path)
  cox <- make_cox_deviance(event = event, start = start, status = status,
                            sample_weight = weights, tie_breaking = "efron")
  result_full <- cox$coxdev(eta)

  # Compare with subset (non-zero weights only)
  nonzero_idx <- weights > 0
  cox_subset <- make_cox_deviance(
    event = event[nonzero_idx],
    start = start[nonzero_idx],
    status = status[nonzero_idx],
    sample_weight = weights[nonzero_idx],
    tie_breaking = "efron"
  )
  result_subset <- cox_subset$coxdev(eta[nonzero_idx])

  # The key test: deviance should match
  expect_true(
    all_close(result_full$deviance, result_subset$deviance, rtol = 1e-10),
    info = sprintf("Deviance mismatch: full=%f, subset=%f",
                   result_full$deviance, result_subset$deviance)
  )

  # loglik_sat should also match
  expect_true(
    all_close(result_full$loglik_sat, result_subset$loglik_sat, rtol = 1e-10),
    info = "Loglik_sat mismatch"
  )

  # Gradient for non-zero weights should match
  expect_true(
    all_close(result_full$gradient[nonzero_idx], result_subset$gradient, rtol = 1e-10),
    info = "Gradient mismatch"
  )
})

# =============================================================================
# Direct tests of saturated log-likelihood formula
# =============================================================================

#' Compute expected saturated log-likelihood using the formula
#'
#' For Breslow: LL_sat = -sum(W_C * log(W_C))
#' For Efron:   LL_sat = -sum(W_C * [log(W_C) + (1/K_C+) * (log(K_C+!) - K_C+ * log(K_C+))])
#'
#' Where:
#'   W_C = sum of weights for events at time t
#'   K_C+ = count of events with positive weight at time t
compute_expected_sat_loglik <- function(event, status, weight, tie_breaking = "breslow") {
  event_times <- event[status == 1]
  event_weights <- weight[status == 1]
  unique_times <- unique(event_times)

  loglik_sat <- 0
  for (t in unique_times) {
    mask <- event_times == t
    w_c <- sum(event_weights[mask])
    if (w_c > 0) {
      # Breslow term
      loglik_sat <- loglik_sat - w_c * log(w_c)

      # Efron penalty term
      if (tie_breaking == "efron") {
        k_c_plus <- sum(event_weights[mask] > 0)
        if (k_c_plus > 0) {
          efron_penalty <- (w_c / k_c_plus) * (lgamma(k_c_plus + 1) - k_c_plus * log(k_c_plus))
          loglik_sat <- loglik_sat - efron_penalty
        }
      }
    }
  }
  loglik_sat
}

test_that("saturated log-likelihood matches formula for Breslow (no ties)", {
  # No ties case: Breslow and Efron should be identical
  event <- c(1, 2, 3, 4, 5)
  status <- c(1L, 1L, 1L, 1L, 1L)
  weight <- c(1.0, 2.0, 1.5, 3.0, 0.5)
  eta <- rep(0, 5)

  cox <- make_cox_deviance(event = event, status = status, sample_weight = weight, tie_breaking = "breslow")
  result <- cox$coxdev(eta)

  expected <- compute_expected_sat_loglik(event, status, weight, "breslow")

  expect_true(
    all_close(result$loglik_sat, expected, rtol = 1e-10),
    info = sprintf("Breslow sat loglik: got %f, expected %f", result$loglik_sat, expected)
  )
})

test_that("saturated log-likelihood matches formula for Efron (no ties)", {
  # No ties case: K_C+ = 1 for each cluster, so Efron penalty = 0
  event <- c(1, 2, 3, 4, 5)
  status <- c(1L, 1L, 1L, 1L, 1L)
  weight <- c(1.0, 2.0, 1.5, 3.0, 0.5)
  eta <- rep(0, 5)

  cox <- make_cox_deviance(event = event, status = status, sample_weight = weight, tie_breaking = "efron")
  result <- cox$coxdev(eta)

  expected <- compute_expected_sat_loglik(event, status, weight, "efron")

  expect_true(
    all_close(result$loglik_sat, expected, rtol = 1e-10),
    info = sprintf("Efron sat loglik (no ties): got %f, expected %f", result$loglik_sat, expected)
  )
})

test_that("saturated log-likelihood matches formula for Breslow (with ties)", {
  # Ties at t=1 and t=2
  event <- c(1, 1, 2, 2, 3)
  status <- c(1L, 1L, 1L, 1L, 1L)
  weight <- c(1.0, 2.0, 1.5, 0.5, 3.0)
  eta <- rep(0, 5)

  cox <- make_cox_deviance(event = event, status = status, sample_weight = weight, tie_breaking = "breslow")
  result <- cox$coxdev(eta)

  expected <- compute_expected_sat_loglik(event, status, weight, "breslow")

  expect_true(
    all_close(result$loglik_sat, expected, rtol = 1e-10),
    info = sprintf("Breslow sat loglik (ties): got %f, expected %f", result$loglik_sat, expected)
  )
})

test_that("saturated log-likelihood matches formula for Efron (with ties)", {
  # Ties at t=1 and t=2 - Efron should differ from Breslow
  event <- c(1, 1, 2, 2, 3)
  status <- c(1L, 1L, 1L, 1L, 1L)
  weight <- c(1.0, 2.0, 1.5, 0.5, 3.0)
  eta <- rep(0, 5)

  cox <- make_cox_deviance(event = event, status = status, sample_weight = weight, tie_breaking = "efron")
  result <- cox$coxdev(eta)

  expected <- compute_expected_sat_loglik(event, status, weight, "efron")

  expect_true(
    all_close(result$loglik_sat, expected, rtol = 1e-10),
    info = sprintf("Efron sat loglik (ties): got %f, expected %f", result$loglik_sat, expected)
  )
})

test_that("saturated log-likelihood matches formula with zero weights (Breslow)", {
  event <- c(1, 1, 2, 2, 3)
  status <- c(1L, 1L, 1L, 1L, 1L)
  weight <- c(0.0, 2.0, 1.5, 0.0, 3.0)  # Zero weights at positions 1 and 4
  eta <- rep(0, 5)

  cox <- make_cox_deviance(event = event, status = status, sample_weight = weight, tie_breaking = "breslow")
  result <- cox$coxdev(eta)

  expected <- compute_expected_sat_loglik(event, status, weight, "breslow")

  expect_true(
    all_close(result$loglik_sat, expected, rtol = 1e-10),
    info = sprintf("Breslow sat loglik (zero weights): got %f, expected %f", result$loglik_sat, expected)
  )
})

test_that("saturated log-likelihood matches formula with zero weights (Efron)", {
  event <- c(1, 1, 2, 2, 3)
  status <- c(1L, 1L, 1L, 1L, 1L)
  weight <- c(0.0, 2.0, 1.5, 0.0, 3.0)  # Zero weights at positions 1 and 4
  eta <- rep(0, 5)

  cox <- make_cox_deviance(event = event, status = status, sample_weight = weight, tie_breaking = "efron")
  result <- cox$coxdev(eta)

  expected <- compute_expected_sat_loglik(event, status, weight, "efron")

  expect_true(
    all_close(result$loglik_sat, expected, rtol = 1e-10),
    info = sprintf("Efron sat loglik (zero weights): got %f, expected %f", result$loglik_sat, expected)
  )
})

test_that("saturated log-likelihood: all zero weights at one time point", {
  # All events at t=1 have zero weight
  event <- c(1, 1, 2, 3)
  status <- c(1L, 1L, 1L, 1L)
  weight <- c(0.0, 0.0, 2.0, 3.0)
  eta <- rep(0, 4)

  for (tie_breaking in c("breslow", "efron")) {
    cox <- make_cox_deviance(event = event, status = status, sample_weight = weight, tie_breaking = tie_breaking)
    result <- cox$coxdev(eta)

    expected <- compute_expected_sat_loglik(event, status, weight, tie_breaking)

    expect_true(
      all_close(result$loglik_sat, expected, rtol = 1e-10),
      info = sprintf("%s sat loglik (all zero at t=1): got %f, expected %f",
                     tie_breaking, result$loglik_sat, expected)
    )
  }
})

test_that("Efron differs from Breslow when there are ties", {
  event <- c(1, 1, 1, 2, 2, 3)  # 3 events at t=1, 2 at t=2

  status <- c(1L, 1L, 1L, 1L, 1L, 1L)
  weight <- c(1.0, 2.0, 1.5, 1.0, 2.0, 3.0)
  eta <- rep(0, 6)

  cox_breslow <- make_cox_deviance(event = event, status = status, sample_weight = weight, tie_breaking = "breslow")
  cox_efron <- make_cox_deviance(event = event, status = status, sample_weight = weight, tie_breaking = "efron")

  result_breslow <- cox_breslow$coxdev(eta)
  result_efron <- cox_efron$coxdev(eta)

  # They should differ when there are ties

  expect_false(
    all_close(result_breslow$loglik_sat, result_efron$loglik_sat, rtol = 1e-6),
    info = "Efron and Breslow sat loglik should differ with ties"
  )

  # Efron should be larger (less negative) due to the penalty term being subtracted
  expect_true(
    result_efron$loglik_sat > result_breslow$loglik_sat,
    info = sprintf("Efron (%f) should be > Breslow (%f) with ties",
                   result_efron$loglik_sat, result_breslow$loglik_sat)
  )
})
