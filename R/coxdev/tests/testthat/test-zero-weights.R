# Zero weights drop the observations, as tests/test_zero_weights.py: the
# deviance, the gradient and the information are those of the data without
# them, for every kind of ties of helper-simulate.R.

test_that("zero weights are as if the observations were dropped", {
  set.seed(50)
  for (combo in all_combos[sample(length(all_combos), 40)]) {
    for (tie_breaking in c("efron", "breslow")) {
      for (have_start in c(TRUE, FALSE)) {
        data <- simulate_df(combo, nrep = 5, size = 10)
        n <- nrow(data)
        weight <- sample_weights_zeros(n)
        keep <- weight > 0
        start <- if (have_start) data$start
        eta <- stats::rnorm(n)

        cox <- CoxDeviance$new(event = data$event, status = data$status, start = start,
                               tie_breaking = tie_breaking)
        C <- cox$compute(eta, weight)
        H <- cox$information(eta, weight)$to_dense()

        dropped <- CoxDeviance$new(event = data$event[keep], status = data$status[keep],
                                   start = start[keep], tie_breaking = tie_breaking)
        C0 <- dropped$compute(eta[keep], weight[keep])
        H0 <- dropped$information(eta[keep], weight[keep])$to_dense()

        expect_equal(C$deviance, C0$deviance, tolerance = 1e-10)
        expect_equal(C$loglik_sat, C0$loglik_sat, tolerance = 1e-10)
        expect_equal(C$gradient[keep], C0$gradient, tolerance = 1e-10)
        expect_true(all(C$gradient[!keep] == 0))
        expect_equal(C$diag_hessian[keep], C0$diag_hessian, tolerance = 1e-10)
        expect_false(anyNA(H))
        expect_equal(H[keep, keep], H0, tolerance = 1e-10)
        expect_true(all(H[!keep, ] == 0) && all(H[, !keep] == 0))
      }
    }
  }
})
