# Direct, O(n^2) computations of what CoxDeviance computes in O(n), as the
# reference_* functions of tests/test_sample_score.py and
# tests/test_residuals_hazard.py, and the brute-force C-index of
# tests/test_concordance.py. A NULL start is -Inf for every observation.

.start_or_inf <- function(start, n) if (is.null(start)) rep(-Inf, n) else start

# score residuals: U_i = sum over the failure terms t of
# (w_i 1{i in D_t} / d - wbar p_ti) (x_i - xbar_t), with Efron's weights for ties
reference_residuals <- function(X, eta, event, status, start, weight, efron) {
  start <- .start_or_inf(start, length(event))
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

# the baseline hazard: at each failure time, the sum over the Efron terms of
# wbar / (the sum of w exp(eta) over the risk set, less s of the failures)
reference_hazard <- function(eta, event, status, start, weight, efron) {
  start <- .start_or_inf(start, length(event))
  e <- weight * exp(eta)
  times <- sort(unique(event[status == 1 & weight > 0]))
  hazard <- vapply(times, function(t) {
    D <- event == t & status == 1 & weight > 0
    R <- event >= t & start < t & weight > 0
    d <- sum(D)
    wbar <- mean(weight[D])
    s <- if (efron) (seq_len(d) - 1) / d else rep(0, d)
    sum(vapply(s, function(si) wbar / sum(e * (R - si * D)), 0))
  }, 0)
  list(time = times, hazard = hazard)
}

# the Schoenfeld residuals: for each failure of positive weight, x_i less the
# mean of x over its risk set with weights w exp(eta); with Efron's ties, the
# average over r = 0, ..., d - 1 of the means with the tied failures
# downweighted by r / d. Zero for the other rows.
reference_schoenfeld <- function(X, eta, event, status, start, weight, efron) {
  start <- .start_or_inf(start, length(event))
  e <- weight * exp(eta - max(eta))
  out <- matrix(0, nrow(X), ncol(X))
  for (t in unique(event[status == 1 & weight > 0])) {
    D <- event == t & status == 1 & weight > 0
    R <- event >= t & start < t & weight > 0
    d <- sum(D)
    means <- sapply(seq_len(d) - 1, function(r) {
      p <- e * (R - (if (efron) r / d else 0) * D)
      drop(p %*% X) / sum(p)
    })
    xbar <- if (is.matrix(means)) rowMeans(means) else mean(means)
    out[D, ] <- sweep(X[D, , drop = FALSE], 2, xbar)
  }
  out
}

# the expected events: exp(eta_i) times the sum over the failure times in
# (start_i, event_i] and the Efron terms r of wbar / S_r, with i's own failure
# time downweighted by r / d with Efron's ties. An observation of zero weight
# still gets the expected events of one at risk.
reference_expected <- function(eta, event, status, start, weight, efron) {
  n <- length(event)
  start <- .start_or_inf(start, n)
  e <- weight * exp(eta)
  E <- numeric(n)
  for (t in unique(event[status == 1 & weight > 0])) {
    D <- event == t & status == 1 & weight > 0
    R <- event >= t & start < t
    d <- sum(D)
    wbar <- mean(weight[D])
    for (r in seq_len(d) - 1) {
      s <- if (efron) r / d else 0
      S <- sum(e * ((R & weight > 0) - s * D))
      E <- E + exp(eta) * (R - s * D) * wbar / S
    }
  }
  E
}

# the risk-set variances: for each failure of positive weight, the variance
# of z over its risk set with weights w exp(eta), averaged over the Efron
# terms as for reference_schoenfeld. Zero for the other rows.
reference_variance <- function(Z, eta, event, status, start, weight, efron) {
  start <- .start_or_inf(start, length(event))
  e <- weight * exp(eta - max(eta))
  V <- matrix(0, nrow(Z), ncol(Z))
  for (t in unique(event[status == 1 & weight > 0])) {
    D <- event == t & status == 1 & weight > 0
    R <- event >= t & start < t & weight > 0
    d <- sum(D)
    v <- sapply(seq_len(d) - 1, function(r) {
      p <- e * (R - (if (efron) r / d else 0) * D)
      p <- p / sum(p)
      drop(p %*% Z^2) - drop(p %*% Z)^2
    })
    v <- if (is.matrix(v)) rowMeans(v) else mean(v)
    V[D, ] <- matrix(v, sum(D), ncol(Z), byrow = TRUE)
  }
  V
}

# Harrell's C-index from its definition: for each failure i at t, each j at
# risk at t that is not a failure at t, in the same stratum, with weight
# w_i w_j, concordant if pred_i > pred_j, half if tied
brute_force_c_index <- function(pred, time, status, start, strata, w) {
  num <- den <- 0
  for (i in which(status == 1)) {
    t <- time[i]
    risk <- start < t & t <= time & !(status == 1 & time == t) & strata == strata[i]
    ww <- w[i] * w[risk]
    num <- num + sum(ww * (1 + sign(pred[i] - pred[risk])) / 2)
    den <- den + sum(ww)
  }
  if (den > 0) num / den else NaN
}

# the setting of tests/test_residuals_hazard.py: cox_data with weights all
# one, unequal, or unequal with 15 zeros, and a linear predictor far from
# mean zero
residual_setting <- function(start, weights, n = 120) {
  d <- cox_data(n = n, start = start)
  w <- if (weights == "ones") rep(1, n) else stats::runif(n, 0.5, 2)
  if (weights == "zeros") w[sample(n, 15)] <- 0
  d$w <- w
  d$eta <- drop(d$X %*% c(0.3, -0.2, 0.1)) + 2
  d
}
