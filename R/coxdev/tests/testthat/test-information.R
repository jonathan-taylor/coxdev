# The information operator: the Hessian of the deviance, symmetric, with the
# diagonal diag_hessian / 2, and its interface.

make_cox <- function(n = 60, start = TRUE, strata = FALSE, weight = "ones", tie_breaking = "efron") {
  event <- ceiling(stats::rexp(n) * 3) / 3      # a coarse grid: many ties
  status <- stats::rbinom(n, 1, 0.7)
  w <- if (weight == "ones") rep(1, n) else stats::runif(n, 0.5, 2)
  if (weight == "zeros") w[1:5] <- 0
  entry <- if (start) floor(stats::runif(n, 0, 0.5) * event * 6) / 6
  list(cox = CoxDeviance$new(event = event, status = status, start = entry,
                             strata = if (strata) seq_len(n) %% 2,
                             sample_weight = w, tie_breaking = tie_breaking),
       eta = stats::rnorm(n) * 0.3, w = w, event = event, status = status, start = entry)
}

test_that("diag_hessian is twice the diagonal of the information", {
  set.seed(4)
  for (tie_breaking in c("efron", "breslow")) {
    for (start in c(FALSE, TRUE)) {
      for (weight in c("ones", "unequal", "zeros")) {
        for (strata in c(FALSE, TRUE)) {
          m <- make_cox(start = start, strata = strata, weight = weight,
                        tie_breaking = tie_breaking)
          I <- m$cox$information(m$eta, m$w)
          D <- as.matrix(I)
          expect_equal(D, t(D), tolerance = 1e-12)
          expect_equal(m$cox$compute(m$eta, m$w)$diag_hessian, 2 * diag(D), tolerance = 1e-10)
          expect_equal(I$diagonal(), diag(D), tolerance = 1e-10)
        }
      }
    }
  }
})

test_that("the information is half the derivative of the gradient", {
  set.seed(5)
  for (strata in c(FALSE, TRUE)) {
    m <- make_cox(n = 30, strata = strata, weight = "unequal")
    D <- m$cox$information(m$eta, m$w)$to_dense()
    h <- 1e-6
    fd <- sapply(seq_along(m$eta), function(i) {
      e <- m$eta
      e[i] <- e[i] + h
      g_plus <- m$cox$compute(e, m$w)$gradient
      e[i] <- e[i] - 2 * h
      (g_plus - m$cox$compute(e, m$w)$gradient) / (2 * h)
    })
    expect_equal(fd / 2, D, tolerance = 1e-6)
  }
})

test_that("the gradient is the derivative of the deviance", {
  set.seed(6)
  m <- make_cox(n = 30, weight = "unequal")
  g <- m$cox$compute(m$eta, m$w)$gradient
  h <- 1e-6
  fd <- sapply(seq_along(m$eta), function(i) {
    e <- m$eta
    e[i] <- e[i] + h
    d_plus <- m$cox$compute(e, m$w)$deviance
    e[i] <- e[i] - 2 * h
    (d_plus - m$cox$compute(e, m$w)$deviance) / (2 * h)
  })
  expect_equal(fd, g, tolerance = 1e-6)
})

test_that("products with vectors and matrices, on either side", {
  set.seed(7)
  m <- make_cox(n = 40)
  I <- m$cox$information(m$eta)
  D <- I$to_dense()
  X <- matrix(stats::rnorm(40 * 3), 40, 3, dimnames = list(NULL, c("a", "b", "c")))
  v <- X[, 1]

  expect_equal(dim(I), c(40L, 40L))
  expect_equal(nrow(I), 40L)
  expect_equal(I$rows(), 40L)
  expect_equal(I$cols(), 40L)
  expect_identical(t(I), I)
  expect_equal(I$multiply(v), drop(D %*% v))
  expect_equal(I$multiply_transpose(v), drop(D %*% v))
  expect_equal(I$multiply(X), D %*% X)
  expect_equal(colnames(I$multiply(X)), c("a", "b", "c"))
  expect_error(I$multiply(1:3), "length 40")
  expect_error(I$multiply(matrix(0, 3, 2)), "40 rows")

  skip_if(getRversion() < "4.4.0", "%*% dispatches on S3 classes from R 4.4.0")
  expect_equal(I %*% v, D %*% v)
  expect_equal(I %*% X, D %*% X)
  expect_equal(t(X) %*% I, t(X) %*% D)
  expect_equal(v %*% I, v %*% D)
  expect_equal(t(X) %*% I %*% X, t(X) %*% D %*% X)
  expect_error(I %*% I, "not supported")
  expect_error(crossprod(I, X))
})

test_that("Matrix inputs work", {
  skip_if_not_installed("Matrix")
  set.seed(8)
  m <- make_cox(n = 40)
  I <- m$cox$information(m$eta)
  X <- Matrix::Matrix(matrix(stats::rnorm(40 * 2), 40, 2))
  expect_equal(I$multiply(X), I$to_dense() %*% as.matrix(X))
})

test_that("an operator keeps its linear predictor after other computations", {
  set.seed(9)
  m <- make_cox(n = 40)
  I1 <- m$cox$information(m$eta)
  D1 <- I1$to_dense()
  eta2 <- m$eta + 1 + stats::rnorm(40)
  w2 <- stats::runif(40)
  I2 <- m$cox$information(eta2, w2)
  m$cox$compute(eta2 * 2)
  # each gives the information at its own linear predictor and weights
  expect_equal(I1$to_dense(), D1)
  fresh <- CoxDeviance$new(event = m$event, status = m$status, start = m$start)
  expect_equal(I2$to_dense(), fresh$information(eta2, w2)$to_dense())
  expect_false(isTRUE(all.equal(I1$to_dense(), I2$to_dense())))
})
