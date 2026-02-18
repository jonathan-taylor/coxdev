## context("Check against glmnet package")
tol  <- 1e-10

get_glmnet_result <- function(event,
                              status,
                              start,
                              eta,
                              weight) {
  event <- as.numeric(event)
  status <- as.numeric(status)
  weight <- as.numeric(weight)
  eta <- as.numeric(eta)
  if (length(start) == length(status)) {
    y <- survival::Surv(start, event, status)
    D_R <- glmnet:::coxnet.deviance3(pred=eta, y=y, weight=weight, std.weights=FALSE)
    G_R <- glmnet:::coxgrad3(eta, y, weight, std.weights=FALSE, diag.hessian=TRUE)
    H_R <- attr(G_R, 'diag_hessian')
  } else {
    y <- Surv(event, status)
    D_R <- glmnet:::coxnet.deviance2(pred=eta, y=y, weight=weight, std.weights=FALSE)
    G_R <- glmnet:::coxgrad2(eta, y, weight, std.weights=FALSE, diag.hessian=TRUE)
    H_R <- attr(G_R, 'diag_hessian')
  }
  list(D = D_R, G = as.numeric(-2 * G_R), H = -2 * H_R)
}


check_glmnet <- function(tie_types,
                        sample_weight,
                        have_start_times,
                        nrep=5,
                        size=5,
                        tol=1e-10) {

  data <- simulate_df(tie_types,
                      nrep,
                      size)

  n <- nrow(data)
  eta <- rnorm(n)
  weight <- sample_weight(n)
  if (have_start_times) {
    start <- data$start
  } else {
    start <- NA
  }
  glmnet_result <- get_glmnet_result(data$event,
                                     data$status,
                                     start,
                                     eta,
                                     weight)
  D_glmnet  <- glmnet_result$D
  G_glmnet <- glmnet_result$G
  H_glmnet <- glmnet_result$H

  cox_deviance <- make_cox_deviance(event = data$event, start = start, status = data$status,
                                    sample_weight = weight, tie_breaking = 'breslow')
  C <- cox_deviance$coxdev(eta)

  expect_true(all_close(D_glmnet, C$deviance),
              info = "Deviance mismatch")
  expect_true(rel_diff_norm(G_glmnet, C$gradient) < tol,
              info = "Gradient mismatch")
  expect_true(rel_diff_norm(H_glmnet, C$diag_hessian) < tol,
              info = "Covariance mismatch")
  ## stopifnot(all_close(D_glmnet, C$deviance))
  ## stopifnot(rel_diff_norm(G_glmnet, C$gradient) < tol)
  ## stopifnot(rel_diff_norm(H_glmnet, C$diag_hessian) < tol)
}

for (tie_type in all_combos) {
  for (sample_weight in list(just_ones, sample_weights)) {
    for (have_start_time in c(TRUE, FALSE)) {
      check_glmnet(tie_type,
                   sample_weight,
                   have_start_time,
                   nrep=5,
                   size=5,
                   tol=1e-10)
    }
  }
}

# =============================================================================
# Tests for saturated log-likelihood (Breslow formula)
# glmnet uses Breslow tie-breaking, so we verify our Breslow saturated
# log-likelihood matches the expected formula from the glmnet paper.
# =============================================================================

#' Compute Breslow saturated log-likelihood using the formula from glmnet paper
#' LL_sat = -sum(W_C * log(W_C)) where W_C is the sum of weights at each unique event time
compute_breslow_sat_loglik <- function(event, status, weight) {
  event_times <- event[status == 1]
  event_weights <- weight[status == 1]
  unique_times <- unique(event_times)

  loglik_sat <- 0
  for (t in unique_times) {
    w_c <- sum(event_weights[event_times == t])
    if (w_c > 0) {
      loglik_sat <- loglik_sat - w_c * log(w_c)
    }
  }
  loglik_sat
}

test_that("Breslow saturated log-likelihood matches glmnet formula (unit weights)", {
  # Test with ties
  event <- c(1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 8, 9, 10, 11, 12, 13, 14, 15)
  status <- rep(1L, 20)
  weight <- rep(1, 20)
  eta <- rep(0, 20)

  expected <- compute_breslow_sat_loglik(event, status, weight)

  cox <- make_cox_deviance(event = event, status = status, sample_weight = weight, tie_breaking = "breslow")
  result <- cox$coxdev(eta)

  expect_true(
    all_close(result$loglik_sat, expected, rtol = 1e-10),
    info = sprintf("Breslow sat loglik mismatch: got %f, expected %f",
                   result$loglik_sat, expected)
  )

  # Also verify deviance matches glmnet
  y <- Surv(event, status)
  dev_glmnet <- glmnet:::coxnet.deviance2(pred = eta, y = y, weight = weight, std.weights = FALSE)
  expect_true(
    all_close(result$deviance, dev_glmnet, rtol = 1e-10),
    info = "Deviance should match glmnet"
  )
})

test_that("Breslow saturated log-likelihood matches glmnet formula (non-unit weights)", {
  event <- c(1, 1, 2, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11)
  status <- rep(1L, 15)
  weight <- c(1.5, 2.0, 1.0, 0.5, 3.0, 2.5, 1.0, 1.5, 2.0, 1.0, 0.5, 3.0, 2.0, 1.5, 1.0)
  set.seed(456)
  eta <- rnorm(15) * 0.5

  expected <- compute_breslow_sat_loglik(event, status, weight)

  cox <- make_cox_deviance(event = event, status = status, sample_weight = weight, tie_breaking = "breslow")
  result <- cox$coxdev(eta)

  expect_true(
    all_close(result$loglik_sat, expected, rtol = 1e-10),
    info = sprintf("Breslow sat loglik mismatch (weighted): got %f, expected %f",
                   result$loglik_sat, expected)
  )

  # Also verify deviance matches glmnet
  y <- Surv(event, status)
  dev_glmnet <- glmnet:::coxnet.deviance2(pred = eta, y = y, weight = weight, std.weights = FALSE)
  expect_true(
    all_close(result$deviance, dev_glmnet, rtol = 1e-10),
    info = "Deviance should match glmnet (weighted)"
  )
})

test_that("Breslow saturated log-likelihood matches glmnet formula (with start times)", {
  set.seed(789)
  n <- 15
  event <- c(1, 1, 2, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11)
  start <- event - runif(n, 0.1, 0.5)
  status <- rep(1L, n)
  weight <- c(1.5, 2.0, 1.0, 0.5, 3.0, 2.5, 1.0, 1.5, 2.0, 1.0, 0.5, 3.0, 2.0, 1.5, 1.0)
  eta <- rnorm(n) * 0.5

  # Saturated log-likelihood only depends on event times and weights, not start times
  expected <- compute_breslow_sat_loglik(event, status, weight)

  cox <- make_cox_deviance(event = event, start = start, status = status, sample_weight = weight, tie_breaking = "breslow")
  result <- cox$coxdev(eta)

  expect_true(
    all_close(result$loglik_sat, expected, rtol = 1e-10),
    info = sprintf("Breslow sat loglik mismatch (start times): got %f, expected %f",
                   result$loglik_sat, expected)
  )

  # Also verify deviance matches glmnet
  y <- Surv(start, event, status)
  dev_glmnet <- glmnet:::coxnet.deviance3(pred = eta, y = y, weight = weight, std.weights = FALSE)
  expect_true(
    all_close(result$deviance, dev_glmnet, rtol = 1e-10),
    info = "Deviance should match glmnet (start times)"
  )
})
