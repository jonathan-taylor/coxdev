# The R interface to the C++ CoxDeviance and StratifiedCoxDeviance classes.
# Mirrors python/coxdev/base.py: CoxDeviance$compute() is CoxDeviance.__call__.

#' Cox model deviance, gradient and information
#'
#' @description
#' An R6 class computing the deviance of the Cox proportional hazards model,
#' its gradient and the diagonal of its Hessian with respect to the linear
#' predictor, the information as a matrix-free linear operator, and draws of
#' the score. It handles Efron's and Breslow's tie-breaking, start times
#' (left truncation), strata and sample weights. It is the R version of the
#' `CoxDeviance` class of the Python package, and runs the same C++ code.
#'
#' The deviance is `2 * (loglik_sat - loglik)`, with `loglik` the (weighted)
#' log partial likelihood and `loglik_sat` its value for the saturated model.
#' The `gradient` and `diag_hessian` are of the deviance with respect to the
#' linear predictor. The information is half the Hessian of the deviance, the
#' negative Hessian of `loglik`: for a design `X`, `crossprod(X, I %*% X)` is
#' the information of the coefficients.
#'
#' Results are cached: a second call with the same linear predictor and
#' weights does no work.
#'
#' @examples
#' set.seed(1)
#' n <- 100
#' X <- matrix(rnorm(n * 3), n, 3)
#' eta <- drop(X %*% c(0.5, -0.5, 0.25))
#' event <- ceiling(rexp(n, exp(eta)) * 4) / 4   # with ties
#' status <- rbinom(n, 1, 0.7)
#' cox <- CoxDeviance$new(event = event, status = status, tie_breaking = "efron")
#' # or, the same, with the response as a Surv object
#' if (requireNamespace("survival", quietly = TRUE)) {
#'   cox <- CoxDeviance$new(survival::Surv(event, status), tie_breaking = "efron")
#' }
#'
#' # deviance, gradient and diagonal of the Hessian
#' res <- cox$compute(eta)
#' res$deviance
#' head(res$gradient)
#'
#' # the information, as an operator
#' I <- cox$information(eta)
#' info_beta <- crossprod(X, I %*% X)
#' sqrt(diag(solve(info_beta)))   # standard errors
#'
#' # draws of the score of the coefficients
#' draws <- crossprod(X, cox$sample_score(eta, n_draws = 2000, seed = 1))
#' diag(cov(t(draws))) / diag(info_beta)   # near 1
#' @export
CoxDeviance <- R6::R6Class("CoxDeviance",
  cloneable = FALSE,
  public = list(
    #' @field tie_breaking The tie-breaking method, `"efron"` or `"breslow"`.
    tie_breaking = NULL,

    #' @description Set up the computations for a survival response, given
    #'   as vectors or as a [survival::Surv()] object.
    #' @param event Event (failure or censoring) times; or the response as a
    #'   `Surv` object, `Surv(time, status)` or `Surv(start, stop, status)`,
    #'   in place of `event`, `status` and `start`. Strata set on it by
    #'   `glmnet::stratifySurv()` are used.
    #' @param status Event indicators: 1 for a failure, 0 for censoring.
    #' @param start Start (entry) times, for left-truncated data, or `NULL`.
    #' @param strata Stratum labels (integers, a factor or character), or
    #'   `NULL` for an unstratified model.
    #' @param sample_weight Sample weights, or `NULL` for all one. They are
    #'   needed here only to know which are 0; each computation takes its own.
    #' @param tie_breaking `"efron"` (the default) or `"breslow"`.
    initialize = function(event,
                          status,
                          start = NULL,
                          strata = NULL,
                          sample_weight = NULL,
                          tie_breaking = c("efron", "breslow")) {
      self$tie_breaking <- match.arg(tie_breaking)
      if (inherits(event, "Surv")) {
        if (!missing(status) || !is.null(start)) {
          stop("give the response either as a Surv object or as event, status and start, not both")
        }
        y <- .from_surv(event)
        event <- y$event
        status <- y$status
        start <- y$start
        if (!is.null(y$strata)) {
          if (!is.null(strata)) stop("strata are given both by stratifySurv and by strata")
          strata <- y$strata
        }
      }
      event <- as.numeric(event)
      n <- length(event)
      if (n == 0L) stop("event must have at least one entry")
      if (anyNA(event)) stop("event must not have missing values")
      status <- .check_length(status, n, "status")
      if (anyNA(status) || !all(status %in% c(0, 1))) {
        stop("status must be binary")
      }
      if (is.null(start)) {
        start <- rep(-Inf, n)
      } else {
        start <- as.numeric(.check_length(start, n, "start"))
        if (anyNA(start)) stop("start must not have missing values")
      }
      if (!is.null(strata)) {
        strata <- .as_strata(.check_length(strata, n, "strata"))
        private$strata_labels <- attr(strata, "labels")
        attr(strata, "labels") <- NULL
      }
      private$n_obs <- n
      private$data <- list(start = start,
                           event = event,
                           status = as.integer(status),
                           strata = if (is.null(strata)) integer(0) else strata)
      private$rebuild(.as_weight(sample_weight, n))
    },

    #' @description Compute the deviance and its derivatives (the `__call__`
    #'   of the Python class).
    #' @param linear_predictor The linear predictor, one entry per observation.
    #' @param sample_weight Sample weights, or `NULL` for all one.
    #' @return A `CoxDevianceResult`: a list with the `linear_predictor`, the
    #'   `sample_weight`, `loglik_sat` (the saturated log-likelihood), the
    #'   `deviance`, its `gradient` with respect to the linear predictor and
    #'   the diagonal `diag_hessian` of its Hessian.
    compute = function(linear_predictor, sample_weight = NULL) {
      private$prepare(linear_predictor, sample_weight)
    },

    #' @description The information at a linear predictor: half the Hessian of
    #'   the deviance, as a matrix-free linear operator.
    #' @param linear_predictor,sample_weight As for `compute()`.
    #' @return A [CoxInformation] operator.
    information = function(linear_predictor, sample_weight = NULL) {
      result <- private$prepare(linear_predictor, sample_weight)
      matmat <- function(arg) {
        private$prepare(result$linear_predictor, result$sample_weight)
        .cox_information_matmat(private$ptr, arg)
      }
      CoxInformation$new(result, matmat)
    },

    #' @description Draws of the score at a linear predictor: from the Cox
    #'   model with that linear predictor, or (`"wild"`) from its score
    #'   residuals.
    #'
    #'   Each draw `r` has one entry per observation, on the scale of the
    #'   score `-gradient / 2`, so that `crossprod(X, r)` is a draw of the
    #'   score of the coefficients of a design `X`. The model-based methods
    #'   draw the failures at each event time from the model and return their
    #'   score: the sum over the failures of `x_j` minus its expectation, with
    #'   risk weights `sample_weight * exp(linear_predictor)`. They differ
    #'   only at tied event times, and need equal positive weights.
    #'
    #'   * `"multinomial"`: the `d` failures drawn with replacement, with
    #'     probabilities proportional to the risk weights: the sampling model
    #'     of Breslow's likelihood. Its covariance is the Breslow information.
    #'   * `"bernoulli"`: each subject at risk fails independently with
    #'     probability `min(1, d p)`, a Cox-model analogue of the weird
    #'     bootstrap, centered with the number of failures drawn. Its mean is
    #'     zero only while `d p < 1`.
    #'   * `"fisher"`: the set of `d` failures drawn from Fisher's noncentral
    #'     hypergeometric distribution, the sampling model of the exact
    #'     discrete partial likelihood, centered at its exact mean. Its
    #'     covariance is the information of the exact likelihood.
    #'   * `"wild"`: the score residuals (of the likelihood set by
    #'     `tie_breaking`) times independent standard normals. Its covariance
    #'     is the sandwich, which does not need the model to be right.
    #'
    #'   See the package README for references.
    #' @param linear_predictor,sample_weight As for `compute()`.
    #' @param method One of `"multinomial"`, `"bernoulli"`, `"fisher"` and
    #'   `"wild"`.
    #' @param n_draws The number of draws, or `NULL` for one draw as a vector.
    #' @param seed The seed of the random numbers: a nonnegative whole number
    #'   (up to `2^53`), or a string of decimal digits for any seed below
    #'   `2^64`. The same seed gives the same draws on every platform, and from
    #'   Python. If `NULL`, a seed is drawn from R's random number generator,
    #'   so that `set.seed()` makes the draws reproducible.
    #' @return A vector of length `n` if `n_draws` is `NULL`, else an
    #'   `n` by `n_draws` matrix.
    sample_score = function(linear_predictor,
                            sample_weight = NULL,
                            method = c("multinomial", "bernoulli", "fisher", "wild"),
                            n_draws = NULL,
                            seed = NULL) {
      method <- match.arg(method)
      k <- if (is.null(n_draws)) 1L else .as_count(n_draws)
      seed <- .as_seed(seed)
      private$prepare(linear_predictor, sample_weight)
      draws <- .cox_sample_score(private$ptr, method, k, seed)
      if (is.null(n_draws)) drop(draws) else draws
    },

    #' @description The wild draw of the score for given multipliers: `r`
    #'   with `crossprod(X, r)` equal to `sum_i multipliers[i] * U_i`, where
    #'   `U_i` is the score residual of observation `i` (its contribution to
    #'   the score, centered within each risk set, with Efron's weights if
    #'   `tie_breaking = "efron"`). With `multipliers` all one, `r` is the
    #'   score `-gradient / 2`.
    #' @param linear_predictor,sample_weight As for `compute()`.
    #' @param multipliers One multiplier per observation.
    #' @return A vector of length `n`.
    wild_score = function(linear_predictor, multipliers, sample_weight = NULL) {
      multipliers <- as.numeric(.check_length(multipliers, private$n_obs, "multipliers"))
      private$prepare(linear_predictor, sample_weight)
      .cox_wild_score(private$ptr, multipliers)
    },

    #' @description The score residuals of a design at a linear predictor.
    #'
    #'   Row `i` is `U_i`, the contribution of observation `i` to the score
    #'   of the coefficients, `crossprod(X, -gradient / 2)`: the sum over the
    #'   failure times of `(w_i 1{i fails} / d - wbar p_i) (x_i - xbar)`,
    #'   centered within each risk set, with Efron's weights for ties if
    #'   `tie_breaking = "efron"` (Lin and Wei, 1989). The rows sum to the
    #'   score, `drop(G %*% U)` is `crossprod(X, wild_score(eta, G))`, and
    #'   `crossprod(U)` is the middle of the robust (sandwich) variance. They
    #'   are `sample_weight` times `residuals(fit, type = "score")` of a
    #'   `survival::coxph` fit, which are per unit weight.
    #' @param linear_predictor,sample_weight As for `compute()`.
    #' @param X The design: a matrix with one row per observation, or a
    #'   vector.
    #' @return A matrix like `X` (a vector for a vector `X`).
    score_residuals = function(linear_predictor, X, sample_weight = NULL) {
      vector_X <- is.null(dim(X))
      X <- if (vector_X) matrix(as.numeric(X), ncol = 1L) else .as_double_matrix(X)
      if (nrow(X) != private$n_obs) {
        stop(sprintf("X must have %d rows (one per observation), not %d",
                     private$n_obs, nrow(X)))
      }
      private$prepare(linear_predictor, sample_weight)
      U <- .cox_score_residuals(private$ptr, X)
      if (vector_X) return(drop(U))
      dimnames(U) <- dimnames(X)
      U
    },

    #' @description The estimate of the baseline hazard at a linear
    #'   predictor: Breslow's, or with `tie_breaking = "efron"` Efron's,
    #'   which spreads the tied failures at a time over Efron's shrinking risk
    #'   sets. It is the `survival::basehaz(fit, centered = FALSE)` of a
    #'   `coxph` fit, at the failure times.
    #' @param linear_predictor,sample_weight As for `compute()`.
    #' @return A data frame with one row per distinct failure time (of
    #'   positive weight), in increasing time within each stratum: the
    #'   `time`, the `hazard` (the increment of the cumulative hazard there)
    #'   and the `cumulative_hazard`, for the linear predictor as given, and
    #'   with strata the `stratum`. The cumulative hazard of observation `i`
    #'   at time `t` is the `cumulative_hazard` at the last failure time
    #'   `<= t` (in its stratum) times `exp(linear_predictor[i])`.
    baseline_hazard = function(linear_predictor, sample_weight = NULL) {
      private$prepare(linear_predictor, sample_weight)
      h <- .cox_baseline_hazard(private$ptr)
      out <- data.frame(time = h$time, hazard = h$hazard,
                        cumulative_hazard = h$cumulative_hazard)
      if (self$stratified) {
        labels <- private$strata_labels
        out$stratum <- if (is.null(labels)) h$stratum else factor(labels[h$stratum], levels = labels)
      }
      out
    },

    #' @description Print a summary.
    #' @param ... Ignored.
    print = function(...) {
      n_strata <- length(unique(private$data$strata))
      cat("<CoxDeviance> ", private$n_obs, " observations, ",
          sum(private$data$status), " failures, ", self$tie_breaking, " ties",
          if (any(is.finite(private$data$start))) ", with start times",
          if (n_strata > 0) paste0(", ", n_strata, " strata"),
          "\n", sep = "")
      invisible(self)
    }
  ),

  active = list(
    #' @field n The number of observations.
    n = function() private$n_obs,
    #' @field stratified Whether the model has strata.
    stratified = function() length(private$data$strata) > 0
  ),

  private = list(
    n_obs = NULL,
    data = NULL,
    strata_labels = NULL,   # the levels of strata given as labels, else NULL
    weight = NULL,     # the weights the C++ object was made with
    ptr = NULL,
    result = NULL,     # the last result: the state of the C++ object

    rebuild = function(weight) {
      d <- private$data
      private$ptr <- .cox_new(d$start, d$event, d$status, d$strata, weight,
                              self$tie_breaking == "efron")
      private$weight <- weight
      private$result <- NULL
    },

    # bring the C++ object to (linear_predictor, sample_weight), unless it is
    # there already, and return the result there
    prepare = function(linear_predictor, sample_weight) {
      n <- private$n_obs
      linear_predictor <- as.numeric(.check_length(linear_predictor, n, "linear_predictor"))
      if (!all(is.finite(linear_predictor))) stop("linear_predictor must be finite")
      sample_weight <- .as_weight(sample_weight, n)
      # a new C++ object when the weights change (zero weights change the
      # preprocessing), or when it was lost (e.g. by saveRDS / readRDS)
      if (!identical(sample_weight, private$weight) || !.cox_valid(private$ptr)) {
        private$rebuild(sample_weight)
      }
      r <- private$result
      if (is.null(r) ||
          !identical(r$linear_predictor, linear_predictor) ||
          !identical(r$sample_weight, sample_weight)) {
        out <- .cox_compute(private$ptr, linear_predictor, sample_weight)
        private$result <- structure(
          list(linear_predictor = linear_predictor,
               sample_weight = sample_weight,
               loglik_sat = out$loglik_sat,
               deviance = out$deviance,
               gradient = out$gradient,
               diag_hessian = out$diag_hessian),
          class = "CoxDevianceResult")
      }
      private$result
    }
  )
)

#' @export
print.CoxDevianceResult <- function(x, ...) {
  cat("<CoxDevianceResult> deviance ", format(x$deviance, ...),
      ", saturated log-likelihood ", format(x$loglik_sat, ...),
      ", ", length(x$linear_predictor), " observations\n", sep = "")
  invisible(x)
}

# event, status and start (NULL for right-censored data) of a Surv object,
# and the strata of glmnet::stratifySurv
.from_surv <- function(y) {
  type <- attr(y, "type")
  m <- unclass(y)
  out <- switch(type,
    right = list(event = m[, "time"], status = m[, "status"], start = NULL),
    counting = list(event = m[, "stop"], status = m[, "status"], start = m[, "start"]),
    stop(sprintf("a Surv object must be right-censored or (start, stop], not %s", type)))
  out$strata <- attr(y, "strata")
  out
}

.check_length <- function(x, n, name) {
  if (length(x) != n) {
    stop(sprintf("%s must have length %d (one entry per observation), not %d",
                 name, n, length(x)))
  }
  x
}

.as_weight <- function(sample_weight, n) {
  if (is.null(sample_weight)) return(rep(1, n))
  w <- as.numeric(.check_length(sample_weight, n, "sample_weight"))
  if (!all(is.finite(w)) || any(w < 0)) {
    stop("sample_weight must be finite and nonnegative")
  }
  w
}

# integer codes of the strata; integer labels are kept, so that the strata
# (and the draws of sample_score) are those of the Python package. Other
# labels are coded by factor(), and their levels kept as the "labels"
# attribute.
.as_strata <- function(strata) {
  if (anyNA(strata)) stop("strata must not have missing values")
  if (is.numeric(strata) && !is.factor(strata) &&
      all(strata == round(strata)) && all(abs(strata) < .Machine$integer.max)) {
    return(as.integer(strata))
  }
  f <- factor(strata)
  structure(as.integer(f), labels = levels(f))
}

.as_count <- function(n_draws) {
  if (length(n_draws) != 1L || !is.numeric(n_draws) || is.na(n_draws) ||
      n_draws < 0 || n_draws != round(n_draws) || n_draws > .Machine$integer.max) {
    stop("n_draws must be a nonnegative whole number")
  }
  as.integer(n_draws)
}

# the seed as a decimal string, for the C++ code's 64-bit seed
.as_seed <- function(seed) {
  if (is.null(seed)) {
    # 53 random bits from R's generator: exact in a double
    seed <- floor(stats::runif(1) * 2^32) * 2^21 + floor(stats::runif(1) * 2^21)
  }
  if (length(seed) != 1L || is.na(seed)) stop("seed must be a single value")
  if (is.character(seed)) {
    if (!grepl("^[0-9]+$", seed)) {
      stop("seed must be a nonnegative integer below 2^64")
    }
    return(seed)
  }
  if (!is.numeric(seed) || seed < 0 || seed != round(seed) || seed > 2^53) {
    stop("a numeric seed must be a whole number between 0 and 2^53; ",
         "give larger seeds as strings")
  }
  sprintf("%.0f", seed)
}
