#!/usr/bin/env python
"""
Example: Cox regression via IRLS + Coordinate Descent

This example demonstrates using the CoxIRLSState class for efficient
coordinate descent optimization. The pattern matches how glmnetpp
integrates with coxdev for penalized Cox regression.
"""

import numpy as np
from coxdev import CoxDeviance, CoxIRLSState

# -----------------------------------------------------------------------------
# Generate survival data
# -----------------------------------------------------------------------------
np.random.seed(42)
n = 200
p = 10

# Design matrix
X = np.random.randn(n, p)

# True coefficients (sparse)
beta_true = np.array([1.0, -0.5, 0.3, 0, 0, 0, 0, 0, 0, 0])

# Generate survival times
eta_true = X @ beta_true
time = np.random.exponential(1 / np.exp(eta_true))
status = np.random.binomial(1, 0.7, n)

print("=== Cox IRLS + Coordinate Descent Example ===\n")
print(f"Data: n = {n}, p = {p}")
print(f"True beta: {beta_true}\n")

# -----------------------------------------------------------------------------
# Create Cox deviance and IRLS state objects
# -----------------------------------------------------------------------------
cox = CoxDeviance(event=time, status=status, tie_breaking="efron")
irls = CoxIRLSState(cox)

# -----------------------------------------------------------------------------
# Simple coordinate descent without penalization
# (Newton-Raphson via cyclic coordinate descent)
# -----------------------------------------------------------------------------
beta = np.zeros(p)
eta = X @ beta
weights = np.ones(n)

max_outer = 10
max_inner = 5
tol = 1e-6

print("--- Unpenalized Cox via IRLS + CD ---")

for outer in range(1, max_outer + 1):
    # Outer IRLS: recompute expensive quantities
    dev = irls.recompute_outer(eta, weights)

    if outer == 1:
        print(f"Outer {outer:2d}: deviance = {dev:.6f} (null)")

    beta_old = beta.copy()

    # Inner coordinate descent passes
    for inner in range(max_inner):
        for j in range(p):
            x_j = X[:, j]
            grad_j, hess_jj = irls.weighted_inner_product(x_j)

            if hess_jj > 1e-10:
                delta = grad_j / hess_jj
                beta[j] += delta
                irls.update_residuals(delta, x_j)

    # Update eta for next outer iteration
    eta = X @ beta

    # Recompute deviance to check convergence
    dev = irls.recompute_outer(eta, weights)

    # Check convergence
    max_change = np.max(np.abs(beta - beta_old))
    print(f"Outer {outer:2d}: deviance = {dev:.6f}, max|delta_beta| = {max_change:.2e}")

    if max_change < tol:
        print("Converged!")
        break

print("\nEstimated beta:")
print(np.round(beta, 4))

# -----------------------------------------------------------------------------
# Compare with lifelines (if available)
# -----------------------------------------------------------------------------
try:
    from lifelines import CoxPHFitter
    import pandas as pd

    print("\n--- Comparison with lifelines ---")

    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(p)])
    df["time"] = time
    df["status"] = status

    cph = CoxPHFitter()
    cph.fit(df, duration_col="time", event_col="status")

    print("\nlifelines coefficients:")
    print(np.round(cph.params_.values, 4))

    print(f"\nMax difference: {np.max(np.abs(beta - cph.params_.values)):.6f}")
except ImportError:
    print("\n(lifelines not installed, skipping comparison)")

# -----------------------------------------------------------------------------
# Example with L1 penalization (soft thresholding)
# -----------------------------------------------------------------------------
print("\n--- L1-Penalized Cox (Coordinate Descent) ---")

def soft_threshold(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0)

lam = 0.1
beta_lasso = np.zeros(p)
eta = X @ beta_lasso

for outer in range(1, 21):
    dev = irls.recompute_outer(eta, weights)
    beta_old = beta_lasso.copy()

    for inner in range(3):
        for j in range(p):
            x_j = X[:, j]
            grad_j, hess_jj = irls.weighted_inner_product(x_j)

            if hess_jj > 1e-10:
                # Soft thresholding for L1 penalty
                z = beta_lasso[j] + grad_j / hess_jj
                beta_new = soft_threshold(z, lam / hess_jj)
                delta = beta_new - beta_lasso[j]

                if abs(delta) > 1e-10:
                    beta_lasso[j] = beta_new
                    irls.update_residuals(delta, x_j)

    eta = X @ beta_lasso

    if np.max(np.abs(beta_lasso - beta_old)) < tol:
        print(f"Converged at outer iteration {outer}")
        break

print(f"\nL1-penalized beta (lambda = {lam}):")
print(np.round(beta_lasso, 4))
print(f"Non-zero coefficients: {np.sum(np.abs(beta_lasso) > 1e-6)}")

print("\n=== Example Complete ===")
