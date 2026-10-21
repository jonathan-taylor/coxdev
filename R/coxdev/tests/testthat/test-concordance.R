# CoxDeviance$c_index, Harrell's concordance index, as tests/test_concordance.py:
# against the O(n^2) count of its definition, its properties, and glmnet's
# Cindex (test-residuals-hazard.R compares it with survival::concordance).

concordance_data <- function(n) {
  # few distinct times and rounded predictions, so that both have ties
  time <- as.numeric(sample(1:9, n, replace = TRUE))
  status <- stats::rbinom(n, 1, 0.6)
  start <- pmin(floor(time * stats::runif(n)), time - 1)
  strata <- sample(0:2, n, replace = TRUE)
  w <- stats::runif(n, 0, 2) * (stats::runif(n) > 0.1)
  pred <- round(stats::rnorm(n), 1)
  list(time = time, status = status, start = start, strata = strata, w = w, pred = pred)
}

test_that("the C-index is the count of its definition", {
  for (seed in 0:29) {
    set.seed(seed)
    n <- sample(2:79, 1)
    d <- concordance_data(n)
    for (setting in list(list(S = FALSE, G = FALSE, W = FALSE), list(S = TRUE, G = FALSE, W = TRUE),
                         list(S = FALSE, G = TRUE, W = TRUE), list(S = TRUE, G = TRUE, W = TRUE))) {
      cox <- CoxDeviance$new(event = d$time, status = d$status,
                             start = if (setting$S) d$start,
                             strata = if (setting$G) d$strata)
      C <- cox$c_index(d$pred, sample_weight = if (setting$W) d$w)
      C_bf <- brute_force_c_index(d$pred, d$time, d$status,
                                  if (setting$S) d$start else rep(-Inf, n),
                                  if (setting$G) d$strata else rep(0, n),
                                  if (setting$W) d$w else rep(1, n))
      if (is.nan(C_bf)) {
        expect_true(is.nan(C))
      } else {
        expect_lt(abs(C - C_bf), 1e-12)
      }
    }
  }
})

test_that("the C-index needs no compute, and a compute does not change it", {
  set.seed(1)
  d <- concordance_data(100)
  cox <- CoxDeviance$new(event = d$time, status = d$status, start = d$start)
  P <- cbind(d$pred, -d$pred, 0)
  C <- cox$c_index(P, sample_weight = d$w)
  expect_length(C, 3)
  expect_equal(C, sapply(1:3, function(j) cox$c_index(P[, j], sample_weight = d$w)))
  expect_equal(C[1] + C[2], 1, tolerance = 1e-12)
  expect_identical(C[3], 0.5)
  cox$compute(d$pred * 0.3)
  expect_identical(cox$c_index(d$pred, sample_weight = d$w), C[1])
  # no comparable pair
  expect_true(is.nan(CoxDeviance$new(event = d$time, status = rep(0, 100))$c_index(d$pred)))
  expect_error(cox$c_index(d$pred[-1]), "one per observation")
})

test_that("the C-index agrees with glmnet's Cindex", {
  skip_if_not_installed("survival")
  skip_if_not_installed("glmnet", minimum_version = "5.0")
  set.seed(2)
  for (start in c(FALSE, TRUE)) {
    n <- 150
    time <- ceiling(stats::rexp(n) * 4) / 4 + 0.25
    status <- stats::rbinom(n, 1, 0.7)
    entry <- if (start) floor(stats::runif(n, 0, 0.5) * time * 8) / 8
    w <- stats::runif(n, 0.2, 3)
    pred <- round(stats::rnorm(n), 1)
    y <- if (start) survival::Surv(entry, time, status) else survival::Surv(time, status)
    cox <- CoxDeviance$new(event = time, status = status, start = entry)
    expect_lt(abs(cox$c_index(pred, sample_weight = w) - glmnet::Cindex(pred, y, weights = w)),
              1e-12)
  }
})
