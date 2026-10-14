# CoxDeviance$score_residuals and $baseline_hazard: the identities with the
# score and wild_score, and survival's residuals(type = "score") and basehaz.

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
