# Input checks, caching, weights, saved objects and make_cox_deviance.

event <- c(3, 6, 8, 4, 6, 4, 3, 2, 2, 5, 3, 4)
status <- c(1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1)
eta <- seq(-1, 1, length.out = 12)

test_that("inputs are checked", {
  expect_error(CoxDeviance$new(event = event, status = status[-1]), "status must have length 12")
  expect_error(CoxDeviance$new(event = event, status = status + 1), "binary")
  expect_error(CoxDeviance$new(event = c(event[-1], NA), status = status), "missing")
  expect_error(CoxDeviance$new(event = event, status = status, start = 1:3), "start must have length")
  expect_error(CoxDeviance$new(event = event, status = status, strata = c(1:11, NA)), "missing")
  expect_error(CoxDeviance$new(event = event, status = status, tie_breaking = "exact"))
  cox <- CoxDeviance$new(event = event, status = status)
  expect_error(cox$compute(eta[-1]), "linear_predictor must have length 12")
  expect_error(cox$compute(replace(eta, 1, Inf)), "finite")
  expect_error(cox$compute(eta, rep(-1, 12)), "nonnegative")
  expect_error(cox$wild_score(eta, 1:3), "multipliers must have length 12")
})

test_that("logical status and integer event times are accepted", {
  a <- CoxDeviance$new(event = as.integer(event), status = status == 1)$compute(eta)
  b <- CoxDeviance$new(event = event, status = status)$compute(eta)
  expect_equal(a$deviance, b$deviance)
})

test_that("the result has the fields of the Python result", {
  cox <- CoxDeviance$new(event = event, status = status)
  r <- cox$compute(eta)
  expect_s3_class(r, "CoxDevianceResult")
  expect_named(r, c("linear_predictor", "sample_weight", "loglik_sat", "deviance",
                    "gradient", "diag_hessian"))
  expect_equal(r$sample_weight, rep(1, 12))
  expect_output(print(r), "deviance")
  expect_output(print(cox), "12 observations, 8 failures, efron ties")
  expect_equal(cox$n, 12L)
  expect_false(cox$stratified)
})

test_that("a constant shift of the linear predictor changes nothing", {
  cox <- CoxDeviance$new(event = event, status = status)
  a <- cox$compute(eta)
  b <- cox$compute(eta + 3)
  expect_equal(a$deviance, b$deviance)
  expect_equal(a$gradient, b$gradient)
})

test_that("weights other than those at construction are used", {
  w <- c(rep(0, 3), runif(9))
  a <- CoxDeviance$new(event = event, status = status)$compute(eta, w)
  b <- CoxDeviance$new(event = event, status = status, sample_weight = w)$compute(eta, w)
  expect_equal(a$deviance, b$deviance)
  expect_equal(a$gradient, b$gradient)
  expect_equal(a$diag_hessian, b$diag_hessian)
})

test_that("Breslow and Efron agree without ties", {
  set.seed(19)
  ev <- rexp(12)
  a <- CoxDeviance$new(event = ev, status = status, tie_breaking = "efron")$compute(eta)
  b <- CoxDeviance$new(event = ev, status = status, tie_breaking = "breslow")$compute(eta)
  expect_equal(a$deviance, b$deviance)
})

test_that("a saved and reloaded object still works", {
  cox <- CoxDeviance$new(event = event, status = status)
  r <- cox$compute(eta)
  I <- cox$information(eta)
  v <- I$multiply(eta)
  f <- tempfile(fileext = ".rds")
  saveRDS(list(cox = cox, I = I), f)
  saved <- readRDS(f)
  expect_equal(saved$cox$compute(eta + 1)$deviance, r$deviance)
  expect_equal(saved$I$multiply(eta), v)
})

test_that("make_cox_deviance still works", {
  set.seed(20)
  start <- event - runif(12)
  for (s in list(NA, start)) {
    old <- make_cox_deviance(event = event, start = s, status = status, tie_breaking = "breslow")
    cox <- CoxDeviance$new(event = event, status = status,
                           start = if (length(s) == 12) s, tie_breaking = "breslow")
    expect_equal(old$coxdev(eta)$deviance, cox$compute(eta)$deviance)
    x <- matrix(rnorm(24), 12, 2)
    expect_equal(old$information(eta)(x), cox$information(eta)$multiply(x))
    expect_true(is.matrix(old$information(eta)(x[, 1])))
  }
})

test_that("the response can be a Surv object", {
  skip_if_not_installed("survival")
  set.seed(21)
  start <- event - runif(12)
  w <- runif(12)
  same <- function(a, b) {
    expect_equal(a$compute(eta, w)$deviance, b$compute(eta, w)$deviance)
    expect_equal(a$compute(eta, w)$gradient, b$compute(eta, w)$gradient)
  }
  same(CoxDeviance$new(survival::Surv(event, status)),
       CoxDeviance$new(event = event, status = status))
  same(CoxDeviance$new(survival::Surv(start, event, status), tie_breaking = "breslow"),
       CoxDeviance$new(event = event, status = status, start = start, tie_breaking = "breslow"))
  # status coded 1/2 or logical, as Surv allows
  same(CoxDeviance$new(survival::Surv(event, status + 1)),
       CoxDeviance$new(event = event, status = status))
  strata <- rep(c("a", "b", "c"), 4)
  same(CoxDeviance$new(survival::Surv(event, status), strata = strata),
       CoxDeviance$new(event = event, status = status, strata = strata))

  expect_error(CoxDeviance$new(survival::Surv(event, status), status = status), "not both")
  expect_error(CoxDeviance$new(survival::Surv(event, status), start = start), "not both")
  expect_error(CoxDeviance$new(survival::Surv(event, status, type = "left")), "not left")
})

test_that("the strata of a Surv object (as glmnet::stratifySurv sets them) are used", {
  skip_if_not_installed("survival")
  strata <- rep(1:3, 4)
  # glmnet::stratifySurv() keeps the strata in the attribute "strata"
  y <- survival::Surv(event, status)
  attr(y, "strata") <- strata
  class(y) <- c("stratifySurv", class(y))
  a <- CoxDeviance$new(y)
  expect_true(a$stratified)
  b <- CoxDeviance$new(event = event, status = status, strata = strata)
  expect_equal(a$compute(eta)$deviance, b$compute(eta)$deviance)
  expect_equal(a$sample_score(eta, seed = 1), b$sample_score(eta, seed = 1))
  expect_error(CoxDeviance$new(y, strata = strata), "both")
})
