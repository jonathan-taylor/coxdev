# The information operator: the Hessian of the deviance, symmetric, with the
# diagonal diag_hessian / 2, and its interface.

# make_cox() is in helper-simulate.R

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

test_that("results and operators are not changed by later calls", {
  # tests/test_information_operator.py
  for (strata in c(FALSE, TRUE)) {
    for (tie_breaking in c("efron", "breslow")) {
      set.seed(10)
      d <- cox_data(n = 120, start = TRUE)
      n <- length(d$event)
      w <- stats::runif(n, 0.5, 2)
      cox <- CoxDeviance$new(event = d$event, status = d$status, start = d$start, sample_weight = w,
                             strata = if (strata) seq_len(n) %% 3, tie_breaking = tie_breaking)
      eta1 <- drop(d$X %*% c(0.3, -0.2, 0.1))
      eta2 <- drop(d$X %*% c(-0.5, 0.4, 0.2))
      V <- matrix(stats::rnorm(n * 4), n, 4)

      r1 <- cox$compute(eta1, w)
      I1 <- cox$information(eta1, w)
      P1 <- sapply(1:4, function(j) I1$multiply(V[, j]))

      # another linear predictor in between changes neither the result nor I1
      r2 <- cox$compute(eta2, w)
      expect_identical(r1$linear_predictor, eta1)
      expect_identical(r1$sample_weight, w)
      expect_false(r1$deviance == r2$deviance)
      expect_equal(I1$multiply(V[, 1]), P1[, 1], tolerance = 1e-12)

      # a matrix in one call is the columns one at a time
      expect_equal(I1$multiply(V), P1, tolerance = 1e-12)
      I2 <- cox$information(eta2, w)
      expect_equal(I2$multiply(V), sapply(1:4, function(j) I2$multiply(V[, j])), tolerance = 1e-12)

      # symmetric
      expect_equal(I1$multiply_transpose(V), P1, tolerance = 1e-12)
      expect_equal(sum(V[, 4] * I1$multiply(V[, 3])), sum(V[, 3] * I1$multiply(V[, 4])),
                   tolerance = 1e-10)
      # the diagonal of the Hessian of the deviance is twice that of I
      E <- diag(n)[, 1:10]
      expect_equal(diag(I1$multiply(E)[1:10, ]), r1$diag_hessian[1:10] / 2, tolerance = 1e-10)
    }
  }
})
