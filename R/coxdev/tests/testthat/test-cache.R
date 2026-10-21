# Caching, as tests/test_cache.py: CoxDeviance makes a new C++ object only
# for new weights, and reuses its result at the linear predictor of the last
# call, comparing the values.

cache_data <- function(stratified = FALSE, n = 50) {
  event <- ceiling(stats::rexp(n) * 3) / 3      # a coarse grid: many ties
  status <- stats::rbinom(n, 1, 0.7)
  entry <- floor(stats::runif(n, 0, 0.5) * event * 6) / 6
  w <- stats::runif(n, 0.5, 2)
  w[1:4] <- 0
  list(args = list(event = event, status = status, start = entry,
                   strata = if (stratified) seq_len(n) %% 2),
       eta = stats::rnorm(n) * 0.3, w = w)
}

new_cox <- function(d, w) do.call(CoxDeviance$new, c(d$args, list(sample_weight = w)))

# count the calls of the C++ constructor and of the deviance computation
local_counts <- function(env = parent.frame()) {
  counts <- new.env()
  counts$new <- 0
  counts$compute <- 0
  cox_new <- coxdev:::.cox_new
  cox_compute <- coxdev:::.cox_compute
  local_mocked_bindings(
    .cox_new = function(...) {
      counts$new <- counts$new + 1
      cox_new(...)
    },
    .cox_compute = function(...) {
      counts$compute <- counts$compute + 1
      cox_compute(...)
    },
    .package = "coxdev", .env = env)
  counts
}

test_that("a result is reused at the same linear predictor and weights", {
  for (stratified in c(FALSE, TRUE)) {
    set.seed(60)
    d <- cache_data(stratified)
    counts <- local_counts()
    cox <- new_cox(d, d$w)
    expect_equal(counts$new, 1)
    first <- cox$compute(d$eta, d$w)
    # equal values (copies): no new object, and the result is reused
    expect_identical(cox$compute(d$eta + 0, d$w * 1), first)
    expect_equal(c(counts$new, counts$compute), c(1, 1))
    # so are the information, residuals and draws at the same point
    cox$information(d$eta, d$w)$multiply(d$eta)
    cox$score_residuals(d$eta, cbind(d$eta), sample_weight = d$w)
    cox$sample_score(d$eta, d$w, method = "wild", seed = 1)
    expect_equal(c(counts$new, counts$compute), c(1, 1))
    # a new linear predictor is computed
    cox$compute(2 * d$eta, d$w)
    expect_equal(c(counts$new, counts$compute), c(1, 2))
  }
})

test_that("new weights give a new object, and the results of a fresh one", {
  for (stratified in c(FALSE, TRUE)) {
    set.seed(61)
    d <- cache_data(stratified)
    other <- d$w
    other[10] <- 2 * other[10]
    calls <- list(list(d$eta, d$w), list(d$eta, other), list(d$eta, d$w / sum(d$w)),
                  list(2 * d$eta, d$w / sum(d$w)), list(d$eta, NULL))
    expected <- lapply(calls, function(a) new_cox(d, a[[2]])$compute(a[[1]], a[[2]]))
    cox <- new_cox(d, d$w)
    counts <- local_counts()
    for (k in seq_along(calls)) {
      a <- cox$compute(calls[[k]][[1]], calls[[k]][[2]])
      b <- expected[[k]]
      expect_equal(a$deviance, b$deviance, tolerance = 1e-12)
      expect_equal(a$gradient, b$gradient, tolerance = 1e-12)
      expect_equal(a$diag_hessian, b$diag_hessian, tolerance = 1e-12)
    }
    # new weights at each call but the first and the fourth
    expect_equal(counts$new, 3)
  }
})

test_that("changed values are seen", {
  set.seed(62)
  d <- cache_data()
  cox <- new_cox(d, d$w)
  e <- d$eta
  weights <- d$w
  cox$compute(e, weights)
  e <- e * 2
  expect_equal(cox$compute(e, weights)$deviance, new_cox(d, d$w)$compute(2 * d$eta, d$w)$deviance,
               tolerance = 1e-12)
  weights[10] <- 0
  expect_equal(cox$compute(e, weights)$deviance,
               new_cox(d, weights)$compute(2 * d$eta, weights)$deviance, tolerance = 1e-12)
})
