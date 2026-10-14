# Comparisons with glmnet (>= 5.0): the deviance, its gradient and the
# diagonal of its Hessian with respect to the linear predictor.

skip_if_not_installed("survival")
skip_if_not_installed("glmnet", minimum_version = "5.0")

test_that("deviance, gradient and diag_hessian agree with glmnet", {
  set.seed(3)
  for (combo in all_combos[sample(length(all_combos), 25)]) {
    for (tie_breaking in c("efron", "breslow")) {
      for (weights in list(just_ones, sample_weights)) {
        for (have_start in c(TRUE, FALSE)) {
          data <- simulate_df(combo, nrep = 3, size = 5)
          n <- nrow(data)
          eta <- stats::rnorm(n)
          weight <- weights(n)
          # glmnet needs start times >= 0 and stop times > 0; the partial
          # likelihood depends on the times only through their order
          low <- min(data$event, if (have_start) data$start)
          shift <- if (low > 0) 0 else ceiling(1 - low)
          y <- if (have_start) survival::Surv(data$start + shift, data$event + shift, data$status)
               else survival::Surv(data$event + shift, data$status)
          D_R <- glmnet::coxnet.deviance(pred = eta, y = y, weights = weight,
                                         std.weights = FALSE, cox.ties = tie_breaking)
          G_R <- glmnet:::coxgrad(eta, y, weight, std.weights = FALSE,
                                  diag.hessian = TRUE, cox.ties = tie_breaking)
          cox <- CoxDeviance$new(event = data$event, status = data$status,
                                 start = if (have_start) data$start,
                                 tie_breaking = tie_breaking)
          C <- cox$compute(eta, weight)
          expect_equal(C$deviance, D_R, tolerance = 1e-10)
          expect_lt(rel_diff(C$gradient, -2 * as.numeric(G_R)), 1e-10)
          # glmnet's diag_hessian is on the scale of coxdev's; with Efron's
          # ties neither is the diagonal of the Hessian at tied failures (and
          # they differ), so compare it only for Breslow's
          if (tie_breaking == "breslow") {
            expect_lt(rel_diff(C$diag_hessian, attr(G_R, "diag_hessian")), 1e-10)
          }
        }
      }
    }
  }
})
