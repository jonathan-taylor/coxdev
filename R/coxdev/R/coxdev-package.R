#' coxdev: Cox model deviance, gradient, information and score sampling
#'
#' @description Fast computation of the deviance of the Cox proportional
#' hazards model, its gradient and the diagonal of its Hessian with respect
#' to the linear predictor, its information as a matrix-free linear operator,
#' and draws of the score, with Efron's or Breslow's ties, start times,
#' strata and sample weights. Start with [CoxDeviance]; the same C++ code
#' runs the Python package `coxdev`.
#'
#' @keywords internal
#' @useDynLib coxdev, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @importFrom R6 R6Class
"_PACKAGE"

