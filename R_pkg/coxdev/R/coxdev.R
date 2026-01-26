#' Make cox deviance object
#' @param event the event vector of times
#' @param start the start vector, if start/stop. Use `NA` for just
#'   right censored data
#' @param status the status vector indicating event or censoring
#' @param tie_breaking default 'efron'
#' @param sample_weight the vector of sample weights, default all ones
#' @return a list of two functions named `coxdev` and `information`
#'   each of which takes a linear predictor as argument
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
#'                                   sample_weight = rep(1.0, length(ty)),
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
                              sample_weight = rep(1.0, length(event))) {
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
    tie_breaking = tie_breaking,
    sample_weight = sample_weight
  )

  # Return coxdev, information, sample_weight accessor, and internal pointer
  list(coxdev = result$coxdev,
       information = result$information,
       sample_weight = result$sample_weight,
       .wrapper_ptr = result$.wrapper_ptr)
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
#' @param sample_weight the vector of sample weights, default all ones
#' @return a list of two functions named `coxdev` and `information`
#'   each of which takes a linear predictor as argument
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
                                          tie_breaking = c('efron', 'breslow'),
                                          sample_weight = rep(1.0, length(event))) {

  tie_breaking <- match.arg(tie_breaking)
  efron <- tie_breaking == 'efron'

  event <- as.numeric(event)
  nevent <- length(event)
  status <- as.integer(status)
  strata <- as.integer(strata)
  sample_weight <- as.numeric(sample_weight)

  if (length(start) != length(status)) {
    start <- rep(-Inf, nevent)
  } else {
    start <- as.numeric(start)
  }

  # Preprocess and create the XPtr (now includes sample_weight)
  wrapper_ptr <- .preprocess_stratified(start, event, status, strata, sample_weight, efron)

  coxdev <- function(linear_predictor) {
    result <- .cox_dev_stratified(wrapper_ptr, as.numeric(linear_predictor))

    list(
      linear_predictor = linear_predictor,
      sample_weight = .get_sample_weight(wrapper_ptr),
      loglik_sat = result$loglik_sat,
      deviance = result$deviance,
      gradient = result$gradient,
      diag_hessian = result$diag_hessian
    )
  }

  information <- function(eta) {
    # Compute deviance to update buffers
    # Warning: Cached buffers are invalidated if coxdev() is called with a
    # different linear_predictor before using this matvec function.
    coxdev_result <- coxdev(eta)

    matvec <- function(arg) {
      # Handle both vector and matrix
      arg <- as.matrix(-arg)  # Negate for information matrix convention
      apply(arg, 2, function(v) {
        .hessian_matvec_stratified(wrapper_ptr, as.numeric(v))
      })
    }
    matvec
  }

  list(
    coxdev = coxdev,
    information = information,
    sample_weight = function() .get_sample_weight(wrapper_ptr),
    n_strata = .get_n_strata(wrapper_ptr),
    n_total = .get_n_total(wrapper_ptr),
    .wrapper_ptr = wrapper_ptr  # Expose for IRLS state creation
  )
}

#' Create IRLS state for Cox model coordinate descent
#'
#' Creates a stateful object that caches expensive quantities (exp(eta), risk sums,
#' working weights/response) computed once per outer IRLS iteration, enabling
#' efficient coordinate descent. Sample weights are stored in the parent cox_obj
#' and used automatically.
#'
#' @param cox_obj A Cox deviance object created by \code{make_cox_deviance} or
#'   \code{make_stratified_cox_deviance}
#' @return A list with methods for IRLS/coordinate descent:
#'   \itemize{
#'     \item \code{recompute_outer(eta)}: Recompute all cached quantities (call once per outer IRLS)
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
#' # Create Cox deviance and IRLS state (weights stored at initialization)
#' cox <- make_cox_deviance(event = time, status = status)
#' irls <- make_cox_irls_state(cox)
#'
#' # Initialize
#' beta <- rep(0, p)
#' eta <- X %*% beta
#'
#' # Outer IRLS iteration (uses stored weights automatically)
#' irls$recompute_outer(eta)
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
#' irls$recompute_outer(eta)
#' cat("After 1 CD pass:", irls$current_deviance(), "\n")
#' @export
make_cox_irls_state <- function(cox_obj) {
  if (is.null(cox_obj$.wrapper_ptr)) {
    stop("cox_obj must be created by make_cox_deviance or make_stratified_cox_deviance")
  }

  # Create the IRLS state
  irls_state_ptr <- .create_irls_state(cox_obj$.wrapper_ptr)

  list(
    recompute_outer = function(eta) {
      .irls_recompute_outer(irls_state_ptr, as.numeric(eta))
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
