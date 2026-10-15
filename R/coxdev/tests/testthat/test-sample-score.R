# CoxDeviance$sample_score and $wild_score, as tests/test_sample_score.py.

METHODS <- c("multinomial", "bernoulli", "fisher", "wild")

# score residuals by direct computation: U_i = sum over the failure terms t of
# (w_i 1{i in D_t} / d - wbar p_ti) (x_i - xbar_t), with Efron's weights for ties
reference_residuals <- function(X, eta, event, status, start, weight, efron) {
  if (is.null(start)) start <- rep(-Inf, length(event))
  e <- weight * exp(eta - max(eta))
  U <- matrix(0, nrow(X), ncol(X))
  for (t in unique(event[status == 1 & weight > 0])) {
    D <- event == t & status == 1 & weight > 0
    R <- event >= t & start < t & weight > 0
    d <- sum(D)
    wbar <- mean(weight[D])
    for (r in seq_len(d) - 1) {
      s <- if (efron) r / d else 0
      w_r <- e * (R - s * D)
      p_r <- w_r / sum(w_r)
      xbar <- drop(p_r %*% X)
      U <- U + (weight * D / d - wbar * p_r) * sweep(X, 2, xbar)
    }
  }
  U
}

breslow_information <- function(X, eta, event, status, start, weight = NULL) {
  cox <- CoxDeviance$new(event = event, status = status, start = start,
                         sample_weight = weight, tie_breaking = "breslow")
  coef_information(cox, X, eta, weight)
}

eta_of <- function(d) drop(d$X %*% c(0.3, -0.2, 0.1))

test_that("the wild score is a sum of the score residuals", {
  set.seed(10)
  for (tie_breaking in c("efron", "breslow")) {
    for (start in c(FALSE, TRUE)) {
      for (weighted in c(FALSE, TRUE)) {
        d <- cox_data(start = start)
        n <- length(d$event)
        weight <- if (weighted) stats::runif(n, 0.5, 2) else rep(1, n)
        eta <- eta_of(d)
        cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start,
                               tie_breaking = tie_breaking, sample_weight = weight)
        U <- reference_residuals(d$X, eta, d$event, d$status, d$start, weight,
                                 tie_breaking == "efron")
        ones <- cox$wild_score(eta, rep(1, n), sample_weight = weight)
        expect_equal(ones, -cox$compute(eta, weight)$gradient / 2, tolerance = 1e-10)
        expect_equal(drop(crossprod(d$X, ones)), colSums(U), tolerance = 1e-8)
        G <- stats::rnorm(n)
        draw <- cox$wild_score(eta, G, sample_weight = weight)
        expect_equal(drop(crossprod(d$X, draw)), drop(G %*% U), tolerance = 1e-8)
      }
    }
  }
})

test_that("draws sum to zero and are reproducible", {
  set.seed(11)
  for (start in c(FALSE, TRUE)) {
    d <- cox_data(start = start)
    n <- length(d$event)
    eta <- eta_of(d)
    cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start)
    for (method in METHODS) {
      draws <- cox$sample_score(eta, method = method, n_draws = 20, seed = 3)
      expect_equal(dim(draws), c(n, 20L))
      expect_equal(colSums(draws), rep(0, 20), tolerance = 1e-9)
      expect_identical(draws, cox$sample_score(eta, method = method, n_draws = 20, seed = 3))
      expect_identical(draws, cox$sample_score(eta, method = method, n_draws = 20, seed = "3"))
      expect_false(isTRUE(all.equal(draws, cox$sample_score(eta, method = method,
                                                            n_draws = 20, seed = 4))))
      one <- cox$sample_score(eta, method = method, seed = 3)
      expect_true(is.vector(one) && length(one) == n)
      # without a seed, a fresh one each time, from R's generator
      expect_false(isTRUE(all.equal(cox$sample_score(eta, method = method),
                                    cox$sample_score(eta, method = method))))
      set.seed(1)
      a <- cox$sample_score(eta, method = method)
      set.seed(1)
      expect_identical(cox$sample_score(eta, method = method), a)
    }
  }
})

test_that("the draws are those of the Python package", {
  # from python: CoxDeviance(event=event, status=status).sample_score(eta, method=m, seed=2024)
  event <- c(3, 6, 8, 4, 6, 4, 3, 2, 2, 5, 3, 4)
  status <- c(1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1)
  eta <- seq(-1, 1, length.out = 12)
  cox <- CoxDeviance$new(event = event, status = status)
  expect_equal(cox$compute(eta)$deviance, 25.194282096268275, tolerance = 1e-13)
  python <- list(
    multinomial = c(-0.0886959871713917, -0.6508380235839233, 0.2193874114570279,
                    0.6055423359250375, -0.12295102733753693, 0.4325525291093102,
                    -0.264046107823554, -0.09055234051730049, -0.10860812424663115,
                    -0.6960866802158419, 0.4535753252430438, 0.310720689161758),
    bernoulli = c(-0.0570283439324455, 0.44308510058170514, -0.6679615595277318,
                  -0.2593444589838083, -0.9608967758176988, -0.3730802332995341,
                  -0.16977219298419535, -0.09055234051730049, -0.10860812424663115,
                  0.7061425192962638, 1.6486684992298573, -0.11065208979848129),
    fisher = c(-0.0971869426169536, -0.7322236836940971, 1.1308326006220737,
               0.5120295973274303, -0.21741377526033379, -0.6562774944431109,
               -0.2754989129463248, -0.0905523405173005, -0.10860812424663115,
               0.3924748715672237, 0.4759456528905565, -0.3335214486825313),
    wild = c(0.2883115723740829, -0.05899880977053942, 0.9625294246243029,
             0.8045867762614034, -2.046399792090064, 1.0730068301135267,
             -0.15645398708124209, 0.1417341419547601, -1.3314714705435222,
             -0.16631942249169576, 0.6831542414451688, -0.19367950479618107))
  for (method in METHODS) {
    expect_equal(cox$sample_score(eta, method = method, seed = 2024), python[[method]],
                 tolerance = 1e-12)
  }
})

test_that("multinomial covariance is the Breslow information", {
  set.seed(12)
  for (start in c(FALSE, TRUE)) {
    d <- cox_data(n = 200, start = start)
    n <- length(d$event)
    weight <- rep(1 / n, n)
    eta <- eta_of(d)
    cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start,
                           sample_weight = weight)
    draws <- crossprod(d$X, cox$sample_score(eta, sample_weight = weight,
                                             method = "multinomial", n_draws = 6000, seed = 1))
    # the score of the weighted loss has covariance w0 times its Hessian
    target <- breslow_information(d$X, eta, d$event, d$status, d$start, weight) / n
    expect_lt(max(abs(rowMeans(draws))), 4 * max(sqrt(diag(target) / 6000)))
    expect_lt(max(abs(stats::cov(t(draws)) - target) / (abs(target) + max(diag(target)))), 0.08)
  }
})

test_that("bernoulli has less variance under heavy ties", {
  set.seed(13)
  d <- cox_data(n = 200)
  eta <- eta_of(d)
  cox <- CoxDeviance$new(event = d$event, status = d$status)
  draws <- crossprod(d$X, cox$sample_score(eta, method = "bernoulli", n_draws = 6000, seed = 2))
  ratio <- diag(stats::cov(t(draws))) / diag(breslow_information(d$X, eta, d$event, d$status, NULL))
  expect_true(all(ratio < 1) && all(ratio > 0.5))
})

test_that("fisher draws the noncentral hypergeometric distribution", {
  # one failure time, d = 2 of m = 5 at risk: P(A) is proportional to the
  # product of exp(eta) over A
  eta <- c(0.8, -0.3, 0.2, 1.1, -1.0)
  cox <- CoxDeviance$new(event = rep(1, 5), status = c(1, 1, 0, 0, 0))
  draws <- cox$sample_score(eta, method = "fisher", n_draws = 50000, seed = 11)
  sets <- utils::combn(5, 2, simplify = FALSE)
  prob <- sapply(sets, function(A) exp(sum(eta[A])))
  prob <- prob / sum(prob)
  mean <- sapply(1:5, function(j) sum(prob[sapply(sets, function(A) j %in% A)]))
  delta <- draws + mean
  expect_equal(delta, round(delta), tolerance = 1e-10)
  expect_equal(colSums(delta), rep(2, ncol(delta)))
  key <- apply(round(delta), 2, function(col) paste(which(col == 1), collapse = ","))
  counts <- table(factor(key, levels = sapply(sets, paste, collapse = ",")))
  expect_gt(stats::chisq.test(as.vector(counts), p = prob)$p.value, 1e-3)
})

test_that("fisher draws are 0 when everyone at risk fails", {
  cox <- CoxDeviance$new(event = rep(1, 3), status = rep(1, 3))
  draws <- cox$sample_score(c(0.5, -0.2, 0.1), method = "fisher", n_draws = 5, seed = 0)
  expect_equal(draws, matrix(0, 3, 5), tolerance = 1e-12)
})

test_that("without ties the model-based methods have the information as covariance", {
  set.seed(14)
  d <- cox_data(n = 150, ties = FALSE)
  eta <- eta_of(d)
  cox <- CoxDeviance$new(event = d$event, status = d$status)
  info <- breslow_information(d$X, eta, d$event, d$status, NULL)
  for (method in c("multinomial", "bernoulli", "fisher")) {
    draws <- crossprod(d$X, cox$sample_score(eta, method = method, n_draws = 6000, seed = 3))
    expect_lt(max(abs(stats::cov(t(draws)) - info) / (abs(info) + max(diag(info)))), 0.08)
  }
})

test_that("the wild covariance is the sandwich", {
  set.seed(15)
  d <- cox_data(n = 200)
  eta <- eta_of(d)
  cox <- CoxDeviance$new(event = d$event, status = d$status)
  draws <- crossprod(d$X, cox$sample_score(eta, method = "wild", n_draws = 6000, seed = 5))
  U <- reference_residuals(d$X, eta, d$event, d$status, NULL, rep(1, length(eta)), TRUE)
  sandwich <- crossprod(U)
  expect_lt(max(abs(stats::cov(t(draws)) - sandwich) / (abs(sandwich) + max(diag(sandwich)))), 0.08)
})

test_that("a subject never at risk has draws 0", {
  set.seed(16)
  d <- cox_data(start = TRUE)
  last <- max(d$event[d$status == 1])
  event <- c(d$event, last + 2)
  status <- c(d$status, 0)
  start <- c(d$start, last + 1)
  cox <- CoxDeviance$new(event = event, status = status, start = start)
  for (method in METHODS) {
    draws <- cox$sample_score(rep(0, length(event)), method = method, n_draws = 50, seed = 0)
    expect_equal(draws[length(event), ], rep(0, 50))
  }
})

test_that("unequal weights need method wild; zero weights drop out", {
  set.seed(17)
  d <- cox_data()
  n <- length(d$event)
  eta <- eta_of(d)
  weight <- stats::runif(n, 0.5, 2)
  cox <- CoxDeviance$new(event = d$event, status = d$status, sample_weight = weight)
  expect_error(cox$sample_score(eta, sample_weight = weight, method = "multinomial"),
               "equal positive weights")
  expect_length(cox$sample_score(eta, sample_weight = weight, method = "wild"), n)
  weight <- rep(1, n)
  weight[1:10] <- 0
  draws <- cox$sample_score(eta, sample_weight = weight, method = "multinomial", n_draws = 20)
  expect_equal(draws[1:10, ], matrix(0, 10, 20))
})

test_that("strata are sampled within themselves", {
  set.seed(18)
  d <- cox_data(n = 200)
  n <- length(d$event)
  strata <- seq_len(n) %% 3
  eta <- eta_of(d)
  cox <- CoxDeviance$new(event = d$event, status = d$status, strata = strata)
  expect_equal(cox$wild_score(eta, rep(1, n)), -cox$compute(eta)$gradient / 2, tolerance = 1e-10)
  for (method in METHODS) {
    draws <- cox$sample_score(eta, method = method, n_draws = 10, seed = 0)
    for (s in 0:2) {
      expect_equal(colSums(draws[strata == s, ]), rep(0, 10), tolerance = 1e-9)
    }
  }
  s0 <- strata == 0
  U <- reference_residuals(d$X[s0, ], eta[s0], d$event[s0], d$status[s0], NULL, rep(1, sum(s0)), TRUE)
  G <- stats::rnorm(n)
  expect_equal(drop(crossprod(d$X[s0, ], cox$wild_score(eta, G)[s0])), drop(G[s0] %*% U),
               tolerance = 1e-8)
})

test_that("seeds are checked", {
  cox <- CoxDeviance$new(event = c(1, 2, 3), status = c(1, 1, 0))
  eta <- c(0, 0.5, 1)
  expect_length(cox$sample_score(eta, seed = "18446744073709551615"), 3)
  expect_error(cox$sample_score(eta, seed = "18446744073709551616"), "below 2\\^64")
  expect_error(cox$sample_score(eta, seed = "-1"), "below 2\\^64")
  expect_error(cox$sample_score(eta, seed = -1), "whole number")
  expect_error(cox$sample_score(eta, seed = 1.5), "whole number")
  expect_error(cox$sample_score(eta, seed = 2^60), "as strings")
  expect_error(cox$sample_score(eta, n_draws = -1), "nonnegative")
  expect_equal(dim(cox$sample_score(eta, n_draws = 0, seed = 1)), c(3L, 0L))
  expect_error(cox$sample_score(eta, method = "wallenius"), "should be one of")
})
