# CoxDeviance$survfit, $martingale_residuals, $deviance_residuals and the
# loglik of compute(): against survival's survfit.coxph, residuals.coxph
# (types "martingale" and "deviance") and coxph's loglik.

skip_if_not_installed("survival")

test_that("survfit, the residuals and the loglik agree with survival", {
  skip_if_survival_broken()   # see helper-survival.R
  set.seed(31)
  for (tie_breaking in c("efron", "breslow")) {
    for (start in c(FALSE, TRUE)) {
      for (weighted in c(FALSE, TRUE)) {
        d <- cox_data(n = 150, start = start)
        n <- length(d$event)
        w <- if (weighted) stats::runif(n, 0.5, 2) else rep(1, n)
        eta <- drop(d$X %*% c(0.3, -0.2, 0.1))
        new_eta <- c(-1, 0, 0.7)
        y <- if (is.null(d$start)) survival::Surv(d$event, d$status)
             else survival::Surv(d$start, d$event, d$status)
        fit <- survival::coxph(y ~ eta, init = 1, weights = w, ties = tie_breaking,
                               control = survival::coxph.control(iter.max = 0, timefix = FALSE))
        S <- survival::survfit(fit, newdata = data.frame(eta = new_eta), se.fit = FALSE)
        S0 <- survival::survfit(fit, se.fit = FALSE)
        cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start,
                               sample_weight = w, tie_breaking = tie_breaking)

        curves <- cox$survfit(eta, w, new_linear_predictor = new_eta)
        expect_equal(curves$time, S$time)
        expect_equal(curves$n_risk, S$n.risk, tolerance = 1e-10)
        expect_equal(curves$n_event, S$n.event, tolerance = 1e-10)
        expect_equal(curves$n_censor, S$n.censor, tolerance = 1e-10)
        expect_equal(unname(curves$cumulative_hazard), unname(S$cumhaz), tolerance = 1e-8)
        expect_equal(unname(curves$survival), unname(S$surv), tolerance = 1e-8)
        # survfit's default curve is at coxph's means: weighted for
        # right-censored data, unweighted for (start, stop] data
        center <- if (start) mean(eta) else sum(w * eta) / sum(w)
        default <- cox$survfit(eta, w, center = center)
        expect_equal(drop(default$cumulative_hazard), S0$cumhaz, tolerance = 1e-8)

        expect_equal(cox$martingale_residuals(eta, w),
                     unname(stats::residuals(fit, type = "martingale")), tolerance = 1e-10)
        expect_equal(cox$deviance_residuals(eta, w),
                     unname(stats::residuals(fit, type = "deviance")), tolerance = 1e-10)
        expect_equal(cox$compute(eta, w)$loglik, fit$loglik[2], tolerance = 1e-10)
      }
    }
  }
})

test_that("survfit with strata: each curve on its own stratum", {
  set.seed(32)
  d <- cox_data(n = 150, start = TRUE)
  n <- length(d$event)
  w <- stats::runif(n, 0.5, 2)
  eta <- drop(d$X %*% c(0.3, -0.2, 0.1))
  strata <- sample(c("a", "b"), n, replace = TRUE)
  cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start,
                         strata = strata, sample_weight = w)
  curves <- cox$survfit(eta, w, new_linear_predictor = c(0.5, -0.5), new_strata = c("a", "b"))
  center <- sum(w * eta) / sum(w)
  for (j in 1:2) {
    s <- c("a", "b")[j]
    k <- strata == s
    rows <- as.character(curves$stratum) == s
    single <- CoxDeviance$new(event = d$event[k], status = d$status[k], start = d$start[k],
                              sample_weight = w[k])$survfit(eta[k], w[k], center = center)
    expect_equal(curves$time[rows], single$time)
    expect_equal(curves$cumulative_hazard[rows, j],
                 drop(single$cumulative_hazard) * exp(c(0.5, -0.5)[j] - center), tolerance = 1e-10)
    expect_true(all(is.na(curves$cumulative_hazard[!rows, j])))
  }
  expect_error(cox$survfit(eta, w, new_linear_predictor = 0), "new_strata is needed")
  expect_error(cox$survfit(eta, w, new_linear_predictor = 0, new_strata = "z"), "not seen")
})
