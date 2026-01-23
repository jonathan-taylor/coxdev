#!/usr/bin/env Rscript
#' Example: Cox regression via IRLS + Coordinate Descent
#'
#' This example demonstrates using the CoxIRLSState classes for efficient
#' coordinate descent optimization. The pattern matches how glmnetpp
#' integrates with coxdev for penalized Cox regression.

library(coxdev)

# -----------------------------------------------------------------------------
# Generate survival data
# -----------------------------------------------------------------------------
set.seed(42)
n <- 200
p <- 10

# Design matrix
X <- matrix(rnorm(n * p), n, p)
colnames(X) <- paste0("V", 1:p)

# True coefficients (sparse)
beta_true <- c(1.0, -0.5, 0.3, 0, 0, 0, 0, 0, 0, 0)

# Generate survival times
eta_true <- as.vector(X %*% beta_true)
time <- rexp(n, exp(eta_true))
status <- rbinom(n, 1, 0.7)

cat("=== Cox IRLS + Coordinate Descent Example ===\n\n")
cat("Data: n =", n, ", p =", p, "\n")
cat("True beta:", beta_true, "\n\n")

# -----------------------------------------------------------------------------
# Create Cox deviance and IRLS state objects
# -----------------------------------------------------------------------------
cox <- make_cox_deviance(event = time, status = status, tie_breaking = "efron")
irls <- make_cox_irls_state(cox)

# -----------------------------------------------------------------------------
# Simple coordinate descent without penalization
# (Newton-Raphson via cyclic coordinate descent)
# -----------------------------------------------------------------------------
beta <- rep(0, p)
eta <- as.vector(X %*% beta)
weights <- rep(1, n)

max_outer <- 10
max_inner <- 5
tol <- 1e-6

cat("--- Unpenalized Cox via IRLS + CD ---\n")

for (outer in 1:max_outer) {
  # Outer IRLS: recompute expensive quantities
  dev <- irls$recompute_outer(eta, weights)

  if (outer == 1) {
    cat(sprintf("Outer %2d: deviance = %.6f (null)\n", outer, dev))
  }

  beta_old <- beta

  # Inner coordinate descent passes
  for (inner in 1:max_inner) {
    for (j in 1:p) {
      x_j <- X[, j]
      gh <- irls$weighted_inner_product(x_j)

      if (gh["hessian"] > 1e-10) {
        delta <- gh["gradient"] / gh["hessian"]
        beta[j] <- beta[j] + delta
        irls$update_residuals(delta, x_j)
      }
    }
  }

  # Update eta for next outer iteration
  eta <- as.vector(X %*% beta)

  # Recompute deviance to check convergence
  dev <- irls$recompute_outer(eta, weights)

  # Check convergence
  max_change <- max(abs(beta - beta_old))
  cat(sprintf("Outer %2d: deviance = %.6f, max|delta_beta| = %.2e\n",
              outer, dev, max_change))

  if (max_change < tol) {
    cat("Converged!\n")
    break
  }
}

cat("\nEstimated beta:\n")
print(round(beta, 4))

# -----------------------------------------------------------------------------
# Compare with survival::coxph
# -----------------------------------------------------------------------------
if (requireNamespace("survival", quietly = TRUE)) {
  cat("\n--- Comparison with survival::coxph ---\n")

  df <- data.frame(time = time, status = status, X)
  fit <- survival::coxph(survival::Surv(time, status) ~ ., data = df)

  cat("\nsurvival::coxph coefficients:\n")
  print(round(coef(fit), 4))

  cat("\nMax difference: ", round(max(abs(beta - coef(fit))), 6), "\n")
}

# -----------------------------------------------------------------------------
# Example with L1 penalization (soft thresholding)
# -----------------------------------------------------------------------------
cat("\n--- L1-Penalized Cox (Coordinate Descent) ---\n")

soft_threshold <- function(z, lambda) {
  sign(z) * pmax(abs(z) - lambda, 0)
}

lambda <- 0.1
beta_lasso <- rep(0, p)
eta <- as.vector(X %*% beta_lasso)

for (outer in 1:20) {
  dev <- irls$recompute_outer(eta, weights)
  beta_old <- beta_lasso

  for (inner in 1:3) {
    for (j in 1:p) {
      x_j <- X[, j]
      gh <- irls$weighted_inner_product(x_j)

      if (gh["hessian"] > 1e-10) {
        # Soft thresholding for L1 penalty
        z <- beta_lasso[j] + gh["gradient"] / gh["hessian"]
        beta_new <- soft_threshold(z, lambda / gh["hessian"])
        delta <- beta_new - beta_lasso[j]

        if (abs(delta) > 1e-10) {
          beta_lasso[j] <- beta_new
          irls$update_residuals(delta, x_j)
        }
      }
    }
  }

  eta <- as.vector(X %*% beta_lasso)

  if (max(abs(beta_lasso - beta_old)) < tol) {
    cat(sprintf("Converged at outer iteration %d\n", outer))
    break
  }
}

cat("\nL1-penalized beta (lambda =", lambda, "):\n")
print(round(beta_lasso, 4))
cat("Non-zero coefficients:", sum(abs(beta_lasso) > 1e-6), "\n")

# -----------------------------------------------------------------------------
# Compare with glmnet
# -----------------------------------------------------------------------------
if (requireNamespace("glmnet", quietly = TRUE)) {
  cat("\n--- Comparison with glmnet ---\n")

  fit_glmnet <- glmnet::glmnet(X, survival::Surv(time, status),
                                family = "cox", lambda = lambda,
                                standardize = FALSE)

  cat("\nglmnet coefficients:\n")
  print(round(as.vector(coef(fit_glmnet)), 4))
}

cat("\n=== Example Complete ===\n")
