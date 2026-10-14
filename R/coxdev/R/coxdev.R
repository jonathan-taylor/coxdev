#' Make a Cox deviance object (superseded)
#'
#' @description Superseded by [CoxDeviance], which this wraps; kept for
#'   code written for earlier versions of the package.
#' @param event The event times.
#' @param start The start times, or `NA` for right-censored data only.
#' @param status The event indicators: 1 for a failure, 0 for censoring.
#' @param tie_breaking `"efron"` (the default) or `"breslow"`.
#' @param weight The sample weights, default all one.
#' @return A list of two functions: `coxdev(linear_predictor,
#'   sample_weight = NULL)`, giving the result of `CoxDeviance$compute()`,
#'   and `information(linear_predictor, sample_weight = NULL)`, giving a
#'   function that multiplies a vector or matrix by the information,
#'   returning a matrix.
#' @examples
#' set.seed(10101)
#' nobs <- 100; nvars <- 10
#' x <- matrix(rnorm(nobs * nvars), nobs, nvars)
#' fx <- drop(x[, 1:3] %*% rnorm(3)) / 3
#' cox_deviance <- make_cox_deviance(event = rexp(nobs, exp(fx)),
#'                                   status = rbinom(nobs, 1, 0.7))
#' result <- cox_deviance$coxdev(linear_predictor = fx)
#' h <- cox_deviance$information(fx)
#' I <- crossprod(x, h(x))
#' @export
make_cox_deviance <- function(event,
                              start = NA,
                              status,
                              tie_breaking = c("efron", "breslow"),
                              weight = rep(1.0, length(event))) {
  if (length(start) != length(status)) start <- NULL
  cox <- CoxDeviance$new(event = event, status = status, start = start,
                         sample_weight = weight,
                         tie_breaking = match.arg(tie_breaking))
  list(coxdev = function(linear_predictor, sample_weight = NULL) {
         unclass(cox$compute(linear_predictor, sample_weight))
       },
       information = function(linear_predictor, sample_weight = NULL) {
         I <- cox$information(linear_predictor, sample_weight)
         function(arg) I$multiply(as.matrix(arg))
       })
}
