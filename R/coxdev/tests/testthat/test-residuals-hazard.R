# CoxDeviance$score_residuals, $schoenfeld_residuals, $baseline_hazard and
# $expected_events: the identities with the score and wild_score, and
# survival's residuals(type = "score" and "schoenfeld"), basehaz and
# predict(type = "expected" and "survival").

skip_if_not_installed("survival")

coxph_at <- function(d, X, beta, weight, ties, strata = NULL) {
  y <- if (is.null(d$start)) survival::Surv(d$event, d$status)
       else survival::Surv(d$start, d$event, d$status)
  ctrl <- survival::coxph.control(iter.max = 0, timefix = FALSE)
  if (is.null(strata)) {
    survival::coxph(y ~ X, init = beta, weights = weight, ties = ties, control = ctrl)
  } else {
    stratum <- strata
    survival::coxph(y ~ X + survival::strata(stratum), init = beta, weights = weight,
                    ties = ties, control = ctrl)
  }
}

test_that("score residuals and the baseline hazard agree with survival", {
  set.seed(30)
  beta <- c(0.3, -0.2, 0.1)
  for (tie_breaking in c("efron", "breslow")) {
    for (start in c(FALSE, TRUE)) {
      for (weighted in c(FALSE, TRUE)) {
        d <- cox_data(n = 150, start = start)
        n <- length(d$event)
        w <- if (weighted) stats::runif(n, 0.5, 2) else rep(1, n)
        eta <- drop(d$X %*% beta)
        cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start,
                               sample_weight = w, tie_breaking = tie_breaking)
        fit <- coxph_at(d, d$X, beta, w, tie_breaking)

        # coxph's score residuals are per unit weight
        U <- cox$score_residuals(eta, d$X, sample_weight = w)
        expect_equal(unname(U), w * unname(stats::residuals(fit, type = "score")),
                     tolerance = 1e-10)

        H <- cox$baseline_hazard(eta, sample_weight = w)
        bh <- survival::basehaz(fit, centered = FALSE)
        at <- match(H$time, bh$time)
        expect_false(anyNA(at))
        expect_equal(H$cumulative_hazard, bh$hazard[at], tolerance = 1e-10)
        expect_equal(cumsum(H$hazard), H$cumulative_hazard)
      }
    }
  }
})

test_that("with strata, as labels", {
  set.seed(31)
  beta <- c(0.3, -0.2, 0.1)
  d <- cox_data(n = 200, start = TRUE)
  n <- length(d$event)
  strata <- c("north", "south", "west")[seq_len(n) %% 3 + 1]
  eta <- drop(d$X %*% beta)
  cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start, strata = strata)
  fit <- coxph_at(d, d$X, beta, rep(1, n), "efron", strata = strata)
  expect_equal(unname(cox$score_residuals(eta, d$X)),
               unname(stats::residuals(fit, type = "score")), tolerance = 1e-10)

  H <- cox$baseline_hazard(eta)
  expect_equal(levels(H$stratum), c("north", "south", "west"))
  bh <- survival::basehaz(fit, centered = FALSE)
  for (s in levels(H$stratum)) {
    Hs <- H[H$stratum == s, ]
    bs <- bh[bh$strata == s, ]
    expect_equal(Hs$cumulative_hazard, bs$hazard[match(Hs$time, bs$time)], tolerance = 1e-10)
  }
  # integer strata keep their values
  H2 <- CoxDeviance$new(event = d$event, status = d$status, start = d$start,
                        strata = (seq_len(n) %% 3) * 10)$baseline_hazard(eta)
  expect_equal(sort(unique(H2$stratum)), c(0L, 10L, 20L))
  expect_equal(H2$cumulative_hazard, H$cumulative_hazard)
})

test_that("the residuals sum to the score and give the wild draws", {
  set.seed(32)
  d <- cox_data(n = 150, start = TRUE)
  n <- length(d$event)
  w <- stats::runif(n, 0.5, 2)
  w[1:10] <- 0
  eta <- drop(d$X %*% c(0.3, -0.2, 0.1)) + 2
  X <- d$X
  colnames(X) <- c("a", "b", "c")
  cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start, sample_weight = w)
  U <- cox$score_residuals(eta, X, sample_weight = w)
  expect_equal(colnames(U), c("a", "b", "c"))
  expect_equal(colSums(U), drop(crossprod(X, -cox$compute(eta, w)$gradient / 2)), tolerance = 1e-10)
  G <- stats::rnorm(n)
  expect_equal(drop(G %*% U), drop(crossprod(X, cox$wild_score(eta, G, sample_weight = w))),
               tolerance = 1e-10)
  expect_equal(U[1:10, ], matrix(0, 10, 3, dimnames = list(NULL, colnames(X))))
  expect_equal(cox$score_residuals(eta, X[, 1], sample_weight = w), unname(U[, 1]))
  expect_error(cox$score_residuals(eta, X[-1, ]), "rows")
})

test_that("the baseline hazard is for the linear predictor as given", {
  set.seed(33)
  d <- cox_data(n = 100)
  eta <- drop(d$X %*% c(0.3, -0.2, 0.1))
  cox <- CoxDeviance$new(event = d$event, status = d$status)
  H <- cox$baseline_hazard(eta)
  expect_named(H, c("time", "hazard", "cumulative_hazard"))
  expect_equal(cox$baseline_hazard(eta + 1.5)$cumulative_hazard,
               H$cumulative_hazard * exp(-1.5), tolerance = 1e-12)
  expect_true(all(diff(H$time) > 0))
})

test_that("a sparse X gives the residuals of the dense X", {
  skip_if_not_installed("Matrix")
  set.seed(34)
  d <- cox_data(n = 120, start = TRUE)
  w <- stats::runif(length(d$event), 0.5, 2)
  eta <- drop(d$X %*% c(0.3, -0.2, 0.1))
  X <- d$X * (abs(d$X) > 1)
  X[, 2] <- 0
  colnames(X) <- c("a", "b", "c")
  cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start, sample_weight = w)
  U <- cox$score_residuals(eta, X, sample_weight = w)
  expect_equal(cox$score_residuals(eta, Matrix::Matrix(X, sparse = TRUE), sample_weight = w), U,
               tolerance = 1e-14)
  expect_equal(unname(U[, 2]), rep(0, nrow(X)), tolerance = 1e-14)
})

test_that("Schoenfeld residuals and expected events agree with survival", {
  set.seed(35)
  beta <- c(0.3, -0.2, 0.1)
  for (tie_breaking in c("efron", "breslow")) {
    for (start in c(FALSE, TRUE)) {
      for (weighted in c(FALSE, TRUE)) {
        d <- cox_data(n = 150, start = start)
        n <- length(d$event)
        w <- if (weighted) stats::runif(n, 0.5, 2) else rep(1, n)
        eta <- drop(d$X %*% beta)
        cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start,
                               sample_weight = w, tie_breaking = tie_breaking)
        fit <- coxph_at(d, d$X, beta, w, tie_breaking)

        # coxph has the rows of the failures, in order of failure time; tied
        # failures share their mean, so compare sorted within each time
        R <- cox$schoenfeld_residuals(eta, d$X, sample_weight = w)
        fail <- d$status == 1
        R_fail <- R[fail, , drop = FALSE]
        sch <- as.matrix(stats::residuals(fit, type = "schoenfeld"))
        mine <- order(d$event[fail], R_fail[, 1])
        theirs <- order(sort(d$event[fail]), sch[, 1])
        expect_equal(unname(R_fail[mine, ]), unname(sch[theirs, ]), tolerance = 1e-10)
        expect_true(all(R[!fail, ] == 0))
        expect_equal(drop(crossprod(R, w)), drop(crossprod(d$X, -cox$compute(eta, w)$gradient / 2)),
                     tolerance = 1e-10)

        E <- cox$expected_events(eta, sample_weight = w)
        expect_equal(E, unname(stats::predict(fit, type = "expected")), tolerance = 1e-10)
        expect_equal(d$status - E, unname(stats::residuals(fit, type = "martingale")),
                     tolerance = 1e-10)
        expect_equal(cox$survival_at_exit(eta, sample_weight = w),
                     unname(stats::predict(fit, type = "survival")), tolerance = 1e-10)
      }
    }
  }
})

test_that("Schoenfeld residuals and expected events with strata, sparse X and zero weights", {
  set.seed(36)
  beta <- c(0.3, -0.2, 0.1)
  d <- cox_data(n = 200, start = TRUE)
  n <- length(d$event)
  strata <- c("north", "south", "west")[seq_len(n) %% 3 + 1]
  eta <- drop(d$X %*% beta)
  cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start, strata = strata)
  fit <- coxph_at(d, d$X, beta, rep(1, n), "efron", strata = strata)
  expect_equal(cox$expected_events(eta), unname(stats::predict(fit, type = "expected")),
               tolerance = 1e-10)
  R <- cox$schoenfeld_residuals(eta, d$X)
  for (s in unique(strata)) {
    k <- strata == s
    single <- CoxDeviance$new(event = d$event[k], status = d$status[k], start = d$start[k])
    expect_equal(R[k, ], single$schoenfeld_residuals(eta[k], d$X[k, ]), tolerance = 1e-12)
  }

  # zero weights: as if those observations were dropped, but for their own
  # expected events, which are still those of an observation at risk
  w <- stats::runif(n, 0.5, 2)
  w[1:15] <- 0
  cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start, sample_weight = w)
  keep <- w > 0
  dropped <- CoxDeviance$new(event = d$event[keep], status = d$status[keep],
                             start = d$start[keep], sample_weight = w[keep])
  R <- cox$schoenfeld_residuals(eta, d$X, sample_weight = w)
  expect_equal(R[keep, ], dropped$schoenfeld_residuals(eta[keep], d$X[keep, ], sample_weight = w[keep]),
               tolerance = 1e-12)
  expect_true(all(R[!keep, ] == 0))
  expect_equal(cox$expected_events(eta, w)[keep],
               dropped$expected_events(eta[keep], w[keep]), tolerance = 1e-12)

  skip_if_not_installed("Matrix")
  X <- d$X * (abs(d$X) > 1)
  colnames(X) <- c("a", "b", "c")
  expect_equal(cox$schoenfeld_residuals(eta, Matrix::Matrix(X, sparse = TRUE), sample_weight = w),
               cox$schoenfeld_residuals(eta, X, sample_weight = w), tolerance = 1e-14)
})

test_that("risk-set variances agree with survival's coxph.detail", {
  set.seed(37)
  beta <- c(0.3, -0.2, 0.1)
  for (tie_breaking in c("efron", "breslow")) {
    for (start in c(FALSE, TRUE)) {
      d <- cox_data(n = 150, start = start)
      n <- length(d$event)
      w <- stats::runif(n, 0.5, 2)
      eta <- drop(d$X %*% beta)
      cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start,
                             sample_weight = w, tie_breaking = tie_breaking)
      fit <- coxph_at(d, d$X, beta, w, tie_breaking)
      detail <- survival::coxph.detail(fit)
      V <- cox$risk_set_variance(eta, d$X, sample_weight = w)
      expect_true(all(V[d$status == 0, ] == 0))
      # imat at each failure time is the sum over its failures of w V
      fail_times <- detail$time
      for (j in 1:3) {
        by_time <- tapply((w * V[, j])[d$status == 1], d$event[d$status == 1], sum)
        expect_equal(as.vector(by_time[as.character(fail_times)]),
                     unname(detail$imat[j, j, ]), tolerance = 1e-10)
      }
      # covariances by polarization
      C12 <- (cox$risk_set_variance(eta, d$X[, 1] + d$X[, 2], sample_weight = w) -
              cox$risk_set_variance(eta, d$X[, 1] - d$X[, 2], sample_weight = w)) / 4
      by_time <- tapply((w * C12)[d$status == 1], d$event[d$status == 1], sum)
      expect_equal(as.vector(by_time[as.character(fail_times)]), unname(detail$imat[1, 2, ]),
                   tolerance = 1e-10)
    }
  }
})

test_that("the C-index agrees with survival's concordance", {
  set.seed(38)
  for (start in c(FALSE, TRUE)) {
    for (stratified in c(FALSE, TRUE)) {
      d <- cox_data(n = 150, start = start)
      n <- length(d$event)
      w <- stats::runif(n, 0.2, 3)
      pred <- round(stats::rnorm(n), 1)
      stratum <- if (stratified) sample(1:3, n, replace = TRUE) else NULL
      cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start,
                             strata = stratum)
      y <- if (is.null(d$start)) survival::Surv(d$event, d$status)
           else survival::Surv(d$start, d$event, d$status)
      C_R <- if (stratified) {
        survival::concordance(y ~ I(-pred) + survival::strata(stratum), weights = w,
                              timefix = FALSE)$concordance
      } else {
        survival::concordance(y ~ I(-pred), weights = w, timefix = FALSE)$concordance
      }
      expect_equal(cox$c_index(pred, sample_weight = w), unname(C_R), tolerance = 1e-12)
    }
  }
  # columns, a reversed predictor, a constant one
  P <- cbind(a = pred, b = -pred, c = 0)
  C <- cox$c_index(P, sample_weight = w)
  expect_equal(names(C), c("a", "b", "c"))
  expect_equal(unname(C[1] + C[2]), 1)
  expect_equal(unname(C[3]), 0.5)
  expect_error(cox$c_index(pred[-1]), "rows")
})
