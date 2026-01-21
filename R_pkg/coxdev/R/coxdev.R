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

  # Create single-stratum vector (all observations in stratum 1)
  strata <- rep(1L, n)

  # Call stratified implementation
  result <- make_stratified_cox_deviance(
    event = event,
    start = start,
    status = status,
    strata = strata,
    tie_breaking = tie_breaking
  )

  # Return only coxdev and information for backward compatibility
  # (stratified version also returns n_strata and n_total)
  list(coxdev = result$coxdev, information = result$information)
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
    n_total = .get_n_total(strat_data_ptr)
  )
}
