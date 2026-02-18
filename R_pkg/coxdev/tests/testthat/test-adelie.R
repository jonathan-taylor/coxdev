# Tests comparing coxdev saturated log-likelihood against R adelie package
# Note: R adelie interface differs from Python - we can only test loss_full()

library(adelie)

# Helper to compute expected saturated loglik from adelie
get_adelie_sat_loglik <- function(adelie_glm, weight_sum, weighted_events) {
  loss_full <- adelie_glm$loss_full()
  -weight_sum * loss_full - log(weight_sum) * weighted_events
}

test_that("coxdev sat loglik matches adelie (efron, with ties)", {
  set.seed(789)
  n <- 30
  # Create ties
  event <- rep(c(1, 2, 3, 4, 5), 6)
  status <- rbinom(n, 1, 0.8)
  weights <- runif(n, 0.5, 2.0)
  eta <- rnorm(n) * 0.3
  start <- rep(0, n)

  weight_sum <- sum(weights)
  weighted_events <- sum(status * weights)

  # coxdev
  cox <- make_cox_deviance(event = event, status = status, sample_weight = weights,
                           tie_breaking = "efron")
  result <- cox$coxdev(eta)

  # adelie
  adelie_glm <- glm.cox(stop = event, status = status, start = start,
                        weights = weights, tie_method = "efron")
  sat_loglik_adelie <- get_adelie_sat_loglik(adelie_glm, weight_sum, weighted_events)

  expect_true(
    all.equal(result$loglik_sat, sat_loglik_adelie, tolerance = 1e-9),
    info = sprintf("Efron sat loglik mismatch: coxdev=%f, adelie=%f",
                   result$loglik_sat, sat_loglik_adelie)
  )
})

test_that("coxdev sat loglik matches adelie (breslow, with ties)", {
  set.seed(321)
  n <- 30
  # Create ties
  event <- rep(c(1, 2, 3, 4, 5), 6)
  status <- rbinom(n, 1, 0.8)
  weights <- runif(n, 0.5, 2.0)
  eta <- rnorm(n) * 0.3
  start <- rep(0, n)

  weight_sum <- sum(weights)
  weighted_events <- sum(status * weights)

  # coxdev
  cox <- make_cox_deviance(event = event, status = status, sample_weight = weights,
                           tie_breaking = "breslow")
  result <- cox$coxdev(eta)

  # adelie
  adelie_glm <- glm.cox(stop = event, status = status, start = start,
                        weights = weights, tie_method = "breslow")
  sat_loglik_adelie <- get_adelie_sat_loglik(adelie_glm, weight_sum, weighted_events)

  expect_true(
    all.equal(result$loglik_sat, sat_loglik_adelie, tolerance = 1e-9),
    info = sprintf("Breslow sat loglik mismatch: coxdev=%f, adelie=%f",
                   result$loglik_sat, sat_loglik_adelie)
  )
})

test_that("Efron differs from Breslow sat loglik when there are ties", {
  set.seed(42)
  n <- 10
  # Every pair tied
  event <- c(1, 1, 2, 2, 3, 3, 4, 4, 5, 5)
  status <- rep(1L, n)
  weights <- rep(1.0, n)
  eta <- rep(0, n)

  cox_efron <- make_cox_deviance(event = event, status = status, sample_weight = weights,
                                  tie_breaking = "efron")
  cox_breslow <- make_cox_deviance(event = event, status = status, sample_weight = weights,
                                    tie_breaking = "breslow")

  result_efron <- cox_efron$coxdev(eta)
  result_breslow <- cox_breslow$coxdev(eta)

  # They should differ
  expect_false(
    isTRUE(all.equal(result_efron$loglik_sat, result_breslow$loglik_sat, tolerance = 1e-6)),
    info = "Efron and Breslow sat loglik should differ with ties"
  )

  # Verify both match adelie
  weight_sum <- sum(weights)
  weighted_events <- sum(status * weights)
  start <- rep(0, n)

  adelie_efron <- glm.cox(stop = event, status = status, start = start,
                          weights = weights, tie_method = "efron")
  adelie_breslow <- glm.cox(stop = event, status = status, start = start,
                            weights = weights, tie_method = "breslow")

  sat_efron_adelie <- get_adelie_sat_loglik(adelie_efron, weight_sum, weighted_events)
  sat_breslow_adelie <- get_adelie_sat_loglik(adelie_breslow, weight_sum, weighted_events)

  expect_true(
    all.equal(result_efron$loglik_sat, sat_efron_adelie, tolerance = 1e-9),
    info = "Efron sat loglik should match adelie"
  )
  expect_true(
    all.equal(result_breslow$loglik_sat, sat_breslow_adelie, tolerance = 1e-9),
    info = "Breslow sat loglik should match adelie"
  )
})

test_that("coxdev sat loglik matches adelie (no ties)", {
  set.seed(111)
  n <- 20
  event <- sort(runif(n, 1, 10))  # No ties
  status <- rbinom(n, 1, 0.7)
  weights <- runif(n, 0.5, 2.0)
  eta <- rnorm(n) * 0.3
  start <- rep(0, n)

  weight_sum <- sum(weights)
  weighted_events <- sum(status * weights)

  for (tie_method in c("efron", "breslow")) {
    cox <- make_cox_deviance(event = event, status = status, sample_weight = weights,
                             tie_breaking = tie_method)
    result <- cox$coxdev(eta)

    adelie_glm <- glm.cox(stop = event, status = status, start = start,
                          weights = weights, tie_method = tie_method)
    sat_loglik_adelie <- get_adelie_sat_loglik(adelie_glm, weight_sum, weighted_events)

    expect_true(
      all.equal(result$loglik_sat, sat_loglik_adelie, tolerance = 1e-9),
      info = sprintf("%s sat loglik mismatch (no ties): coxdev=%f, adelie=%f",
                     tie_method, result$loglik_sat, sat_loglik_adelie)
    )
  }
})

test_that("coxdev sat loglik matches adelie (all events at same time)", {
  set.seed(222)
  n <- 20
  event <- rep(5.0, n)  # All at same time
  status <- rep(1L, n)  # All events
  weights <- rep(1.0, n)
  eta <- rep(0, n)
  start <- rep(0, n)

  weight_sum <- sum(weights)
  weighted_events <- sum(status * weights)

  for (tie_method in c("efron", "breslow")) {
    cox <- make_cox_deviance(event = event, status = status, sample_weight = weights,
                             tie_breaking = tie_method)
    result <- cox$coxdev(eta)

    adelie_glm <- glm.cox(stop = event, status = status, start = start,
                          weights = weights, tie_method = tie_method)
    sat_loglik_adelie <- get_adelie_sat_loglik(adelie_glm, weight_sum, weighted_events)

    expect_true(
      all.equal(result$loglik_sat, sat_loglik_adelie, tolerance = 1e-9),
      info = sprintf("%s sat loglik mismatch (all tied): coxdev=%f, adelie=%f",
                     tie_method, result$loglik_sat, sat_loglik_adelie)
    )
  }
})
