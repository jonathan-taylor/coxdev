# The information as a matrix-free linear operator, the CoxInformation
# LinearOperator of the Python package. The methods follow the LinearOp of
# the lassoinf package: multiply, multiply_transpose, rows, cols, diagonal and
# to_dense.

#' The information of the Cox model as a linear operator
#'
#' @description
#' The information at a linear predictor (half the Hessian of the deviance
#' with respect to the linear predictor, the negative Hessian of the log
#' partial likelihood), as an `n` by `n` symmetric linear operator that is
#' never formed: a product with a vector costs `O(n)`, after the sort done
#' once by [CoxDeviance]. Made by `CoxDeviance$information()`.
#'
#' Besides its methods, the operator works with `%*%` (on either side, with a
#' vector or a matrix; R >= 4.4.0), `dim()`, `nrow()`, `ncol()`, `t()`
#' and `as.matrix()`; its diagonal is `$diagonal()`.
#'
#' The operator keeps its linear predictor and weights: it gives the
#' information there even after the [CoxDeviance] has computed at others.
#'
#' @return Made by `CoxDeviance$information()`, a `CoxInformation` object, an
#'   R6 linear operator: `multiply()` returns the product with a vector (a
#'   vector) or with a matrix (a matrix), `diagonal()` the diagonal (a
#'   vector), and `to_dense()` the `n` by `n` matrix.
#' @examples
#' set.seed(1)
#' n <- 50
#' X <- matrix(rnorm(n * 2), n, 2)
#' eta <- drop(X %*% c(0.5, -0.5))
#' cox <- CoxDeviance$new(event = rexp(n, exp(eta)), status = rbinom(n, 1, 0.7))
#' I <- cox$information(eta)
#' I
#' crossprod(X, I %*% X)     # the information of the coefficients
#' t(X) %*% I %*% X          # the same
#' all.equal(I$diagonal(), diag(as.matrix(I)))
#' @export
CoxInformation <- R6::R6Class("CoxInformation",
  cloneable = FALSE,
  public = list(
    #' @field result The [CoxDeviance] result at which this is the information.
    result = NULL,

    #' @description Made by `CoxDeviance$information()`, not directly.
    #' @param result A `CoxDevianceResult`.
    #' @param matmat A function multiplying each column of a matrix by the
    #'   information at `result`.
    initialize = function(result, matmat) {
      self$result <- result
      private$matmat <- matmat
    },

    #' @description The product of the information with `v`.
    #' @param v A vector of length `n`, or a matrix with `n` rows.
    #' @return A vector of length `n` if `v` is a vector, else a matrix.
    multiply = function(v) {
      n <- self$rows()
      if (is.matrix(v) || inherits(v, "Matrix")) {
        if (nrow(v) != n) stop(sprintf("the matrix must have %d rows, not %d", n, nrow(v)))
        out <- private$matmat(.as_double_matrix(v))
        colnames(out) <- colnames(v)
        out
      } else {
        if (length(v) != n) stop(sprintf("the vector must have length %d, not %d", n, length(v)))
        drop(private$matmat(matrix(as.numeric(v), ncol = 1L)))
      }
    },

    #' @description The product of the transpose with `v`: as the information
    #'   is symmetric, the same as `multiply(v)`.
    #' @param v A vector of length `n`, or a matrix with `n` rows.
    multiply_transpose = function(v) self$multiply(v),

    #' @description The number of rows, `n`.
    rows = function() length(self$result$linear_predictor),

    #' @description The number of columns, `n`.
    cols = function() length(self$result$linear_predictor),

    #' @description The diagonal of the information, `diag_hessian / 2`.
    diagonal = function() self$result$diag_hessian / 2,

    #' @description The `n` by `n` matrix, by `n` products: for small
    #'   problems and testing.
    to_dense = function() {
      n <- self$rows()
      self$multiply(diag(n))
    },

    #' @description Print a summary.
    #' @param ... Ignored.
    print = function(...) {
      n <- self$rows()
      cat("<CoxInformation> ", n, " x ", n, " linear operator\n", sep = "")
      invisible(self)
    }
  ),

  private = list(
    matmat = NULL
  )
)

#' The derivative of the information of the Cox model in a direction
#'
#' @description
#' The derivative of the information at a linear predictor in a direction of
#' the linear predictor, `d/dt I(linear_predictor + t * direction)` at
#' `t = 0`, as an `n` by `n` symmetric linear operator that is never formed:
#' a product with a vector costs `O(n)`. Made by
#' `CoxDeviance$information_derivative()`.
#'
#' It works as a [CoxInformation] does, with `%*%`, `dim()`, `t()` and
#' `as.matrix()`, except that it has no `$diagonal()`.
#'
#' @return Made by `CoxDeviance$information_derivative()`, a
#'   `CoxInformationDerivative` object, an R6 linear operator: `multiply()`
#'   returns the product with a vector (a vector) or with a matrix (a matrix),
#'   and `to_dense()` the `n` by `n` matrix.
#' @examples
#' set.seed(1)
#' n <- 50
#' X <- matrix(rnorm(n * 2), n, 2)
#' eta <- drop(X %*% c(0.5, -0.5))
#' cox <- CoxDeviance$new(event = rexp(n, exp(eta)), status = rbinom(n, 1, 0.7))
#' # the derivative of the information of the coefficients as they move
#' # along b
#' b <- c(1, 0)
#' dI <- cox$information_derivative(eta, drop(X %*% b))
#' crossprod(X, dI %*% X)
#' @export
CoxInformationDerivative <- R6::R6Class("CoxInformationDerivative",
  inherit = CoxInformation,
  cloneable = FALSE,
  public = list(
    #' @field direction The direction of the derivative.
    direction = NULL,

    #' @description Made by `CoxDeviance$information_derivative()`, not
    #'   directly.
    #' @param result A `CoxDevianceResult`.
    #' @param direction The direction of the derivative.
    #' @param matmat A function multiplying each column of a matrix by the
    #'   derivative of the information at `result` in the direction
    #'   `direction`.
    initialize = function(result, direction, matmat) {
      super$initialize(result, matmat)
      self$direction <- direction
    },

    #' @description Not available in `O(n)`: use `diag(as.matrix(dI))`.
    diagonal = function() {
      stop("the diagonal of the derivative of the information is not available; ",
           "use diag(as.matrix(dI)) for small problems")
    },

    #' @description Print a summary.
    #' @param ... Ignored.
    print = function(...) {
      n <- self$rows()
      cat("<CoxInformationDerivative> ", n, " x ", n, " linear operator\n", sep = "")
      invisible(self)
    }
  )
)

.as_double_matrix <- function(v) {
  v <- as.matrix(v)
  storage.mode(v) <- "double"
  v
}

#' @export
dim.CoxInformation <- function(x) c(x$rows(), x$cols())

#' @export
t.CoxInformation <- function(x) x

#' @export
as.matrix.CoxInformation <- function(x, ...) x$to_dense()

# %*% dispatches S3 methods through the matrixOps group generic from R 4.4.0
#' @rawNamespace if (getRversion() >= "4.4.0") S3method(matrixOps, CoxInformation, .matrixOps_CoxInformation)
#' @noRd
.matrixOps_CoxInformation <- function(x, y) {
  if (.Generic != "%*%") {
    stop(sprintf("%s is not defined for a CoxInformation", .Generic))
  }
  if (inherits(x, "CoxInformation") && inherits(y, "CoxInformation")) {
    stop("the product of two CoxInformation operators is not supported; ",
         "use as.matrix() on one of them")
  }
  if (inherits(x, "CoxInformation")) {
    out <- x$multiply(if (is.matrix(y) || inherits(y, "Matrix")) y else as.numeric(y))
    if (is.matrix(out)) out else matrix(out, ncol = 1L)
  } else {
    # y is the operator: x %*% I = t(I %*% t(x)), as I is symmetric
    if (is.matrix(x) || inherits(x, "Matrix")) {
      t(y$multiply(t(x)))
    } else {
      matrix(y$multiply(as.numeric(x)), nrow = 1L)
    }
  }
}
