#' Make cox deviance object
#' @param event the event vector of times
#' @param start the start vector, if start/stop. Use `NA` for just
#'   right censored data
#' @param status the status vector indicating event or censoring
#' @param tie_breaking default 'efron'
#' @param weight the sample weight the vector of sample weights,
#'   default all ones
#' @return a list of two functions named `coxdev` and `information`
#'   each of which takes a linear predictor as argument, along with
#'   weights
#' @examples
#' set.seed(10101)
#' nobs <- 100; nvars <- 10
#' nzc <- nvars %/% 3
#' x <- matrix(rnorm(nobs * nvars), nobs, nvars)
#' beta <- rnorm(nzc)
#' fx <- x[, seq(nzc)] %*% beta / 3
#' hx <- exp(fx)
#' ty <- rexp(nobs,hx)
#' tcens <- rbinom(n = nobs, prob = 0.3, size = 1)
#' cox_deviance <- make_cox_deviance(event = ty,
#'                                   status = tcens,
#'                                   weight = rep(1.0, length(ty)),
#'                                   tie_breaking = 'efron')
#' result  <- cox_deviance$coxdev(linear_predictor = fx)
#' str(result)
#' tx  <- t(x)
#' h <- cox_deviance$information(fx)
#' I <- tx %*% h(x)  ## I should be symmetric
#' cov  <- solve(I)
#' @export
make_cox_deviance <- function(event,
                              start = NA,
                              status,
                              tie_breaking = c('efron', 'breslow'),
                              weight = rep(1.0, length(event))) {
  # Unified implementation: use stratified code with single stratum
  # This reduces code duplication and ensures consistent behavior
  tie_breaking <- match.arg(tie_breaking)
  n <- length(event)

  # Use empty strata vector to signal single-stratum (unstratified) case
  # This avoids O(n) storage and O(2n) iteration in C++ preprocessing
  strata <- integer(0)

  # Call stratified implementation
  result <- make_stratified_cox_deviance(
    event = event,
    start = start,
    status = status,
    strata = strata,
    tie_breaking = tie_breaking
  )

  # Return coxdev, information, and internal pointer for IRLS state creation
  # (stratified version also returns n_strata and n_total)
  list(coxdev = result$coxdev,
       information = result$information,
       .strat_data_ptr = result$.strat_data_ptr)
}

#' Make stratified cox deviance object (C++ implementation)
#'
#' Creates a stratified Cox deviance calculator using pure C++ implementation
#' for all strata processing. This is faster than the R-loop version when
#' there are many strata.
#'
#' @param event the event vector of times
#' @param start the start vector, if start/stop. Use `NA` for just
#'   right censored data
#' @param status the status vector indicating event or censoring
#' @param strata the strata vector (integer or factor)
#' @param tie_breaking default 'efron'
#' @return a list of two functions named `coxdev` and `information`
#'   each of which takes a linear predictor as argument, along with
#'   weights
#' @examples
#' set.seed(10101)
#' nobs <- 100
#' x <- matrix(rnorm(nobs * 5), nobs, 5)
#' beta <- c(1, -0.5, 0.3, 0, 0)
#' fx <- x %*% beta
#' hx <- exp(fx)
#' ty <- rexp(nobs, hx)
#' tcens <- rbinom(n = nobs, prob = 0.3, size = 1)
#' strata <- sample(1:3, nobs, replace = TRUE)
#' cox_strat <- make_stratified_cox_deviance(
#'   event = ty,
#'   status = tcens,
#'   strata = strata,
#'   tie_breaking = 'efron'
#' )
#' result <- cox_strat$coxdev(linear_predictor = fx)
#' str(result)
#' @export
make_stratified_cox_deviance <- function(event,
                                          start = NA,
                                          status,
                                          strata,
                                          tie_breaking = c('efron', 'breslow')) {

  tie_breaking <- match.arg(tie_breaking)
  efron <- tie_breaking == 'efron'

  event <- as.numeric(event)
  nevent <- length(event)
  status <- as.integer(status)
  strata <- as.integer(strata)

  if (length(start) != length(status)) {
    start <- rep(-Inf, nevent)
  } else {
    start <- as.numeric(start)
  }

  # Preprocess and create the XPtr
  strat_data_ptr <- .preprocess_stratified(start, event, status, strata, efron)

  coxdev <- function(linear_predictor, sample_weight = NULL) {
    if (is.null(sample_weight)) {
      sample_weight <- rep(1.0, length(linear_predictor))
    } else {
      sample_weight <- as.numeric(sample_weight)
    }

    result <- .cox_dev_stratified(strat_data_ptr,
                                   as.numeric(linear_predictor),
                                   sample_weight)

    list(
      linear_predictor = linear_predictor,
      sample_weight = sample_weight,
      loglik_sat = result$loglik_sat,
      deviance = result$deviance,
      gradient = result$gradient,
      diag_hessian = result$diag_hessian
    )
  }

  information <- function(eta, sample_weight = NULL) {
    # Compute deviance to update buffers
    coxdev_result <- coxdev(eta, sample_weight)

    matvec <- function(arg) {
      # Handle both vector and matrix
      arg <- as.matrix(-arg)  # Negate for information matrix convention
      apply(arg, 2, function(v) {
        .hessian_matvec_stratified(strat_data_ptr,
                                    as.numeric(v),
                                    as.numeric(eta),
                                    coxdev_result$sample_weight)
      })
    }
    matvec
  }

  list(
    coxdev = coxdev,
    information = information,
    n_strata = .get_n_strata(strat_data_ptr),
    n_total = .get_n_total(strat_data_ptr),
    .strat_data_ptr = strat_data_ptr  # Expose for IRLS state creation
  )
}

#' Create IRLS state for Cox model coordinate descent
#'
#' Creates a stateful object that caches expensive quantities (exp(eta), risk sums,
#' working weights/response) computed once per outer IRLS iteration, enabling
#' efficient coordinate descent.
#'
#' @param cox_obj A Cox deviance object created by \code{make_cox_deviance} or
#'   \code{make_stratified_cox_deviance}
#' @return A list with methods for IRLS/coordinate descent:
#'   \itemize{
#'     \item \code{recompute_outer(eta, weights)}: Recompute all cached quantities (call once per outer IRLS)
#'     \item \code{working_weights()}: Get cached working weights
#'     \item \code{working_response()}: Get cached working response
#'     \item \code{residuals()}: Get cached residuals r = w * (z - eta)
#'     \item \code{current_deviance()}: Get deviance from last recompute_outer
#'     \item \code{weighted_inner_product(x_j)}: Returns c(gradient_j, hessian_jj)
#'     \item \code{update_residuals(delta, x_j)}: Update r -= delta * w * x_j
#'     \item \code{reset_residuals(eta)}: Reset residuals for new CD pass
#'   }
#' @examples
#' # Simple coordinate descent example
#' set.seed(42)
#' n <- 100; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' beta_true <- c(1, -0.5, 0.3, 0, 0)
#' eta_true <- X %*% beta_true
#' time <- rexp(n, exp(eta_true))
#' status <- rbinom(n, 1, 0.7)
#'
#' # Create Cox deviance and IRLS state
#' cox <- make_cox_deviance(event = time, status = status)
#' irls <- make_cox_irls_state(cox)
#'
#' # Initialize
#' beta <- rep(0, p)
#' eta <- X %*% beta
#' weights <- rep(1, n)
#'
#' # Outer IRLS iteration
#' irls$recompute_outer(eta, weights)
#' cat("Initial deviance:", irls$current_deviance(), "\n")
#'
#' # Inner coordinate descent pass
#' for (j in 1:p) {
#'   x_j <- X[, j]
#'   gh <- irls$weighted_inner_product(x_j)
#'   delta <- gh["gradient"] / gh["hessian"]
#'   beta[j] <- beta[j] + delta
#'   irls$update_residuals(delta, x_j)
#' }
#' eta <- X %*% beta
#' irls$recompute_outer(eta, weights)
#' cat("After 1 CD pass:", irls$current_deviance(), "\n")
#' @export
make_cox_irls_state <- function(cox_obj) {
  if (is.null(cox_obj$.strat_data_ptr)) {
    stop("cox_obj must be created by make_cox_deviance or make_stratified_cox_deviance")
  }

  # Create the IRLS state
  irls_state_ptr <- .create_irls_state(cox_obj$.strat_data_ptr)

  list(
    recompute_outer = function(eta, weights = NULL) {
      if (is.null(weights)) {
        weights <- rep(1.0, length(eta))
      }
      .irls_recompute_outer(irls_state_ptr, as.numeric(eta), as.numeric(weights))
    },

    working_weights = function() {
      .irls_working_weights(irls_state_ptr)
    },

    working_response = function() {
      .irls_working_response(irls_state_ptr)
    },

    residuals = function() {
      .irls_residuals(irls_state_ptr)
    },

    current_deviance = function() {
      .irls_current_deviance(irls_state_ptr)
    },

    weighted_inner_product = function(x_j) {
      .irls_weighted_inner_product(irls_state_ptr, as.numeric(x_j))
    },

    update_residuals = function(delta, x_j) {
      .irls_update_residuals(irls_state_ptr, as.numeric(delta), as.numeric(x_j))
    },

    reset_residuals = function(eta) {
      .irls_reset_residuals(irls_state_ptr, as.numeric(eta))
    }
  )
}
