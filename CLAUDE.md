# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

Cox proportional hazards model deviance, gradients, and Hessian computations for survival analysis. Header-only C++ library (`include/coxdev.hpp`) with separate bindings for Python (pybind11) and R (Rcpp), handling stratified and unstratified models.

## Build & Test Commands

```bash
# Install (Homebrew Eigen)
EIGEN_LIBRARY_PATH=/opt/homebrew/include/eigen3 uv pip install -e .

# Alternative: Use Eigen submodule
git submodule update --init --recursive && uv pip install -e .

# Tests
uv run pytest tests/                           # All tests
uv run pytest tests/test_compareR.py -v        # R comparison (requires rpy2, glmnet, survival)
uv run pytest tests/test_compare_adelie.py -v  # Adelie comparison (requires rpy2, R adelie)
uv run pytest tests/test_compareR.py::test_simple  # Single test
```

## File Structure (Adapter Pattern)

| Location | Purpose |
|----------|---------|
| `include/coxdev.hpp` | **Canonical header-only C++ library** (single source of truth) |
| `src/coxdev_py.cpp` | Python pybind11 bindings |
| `R_pkg/coxdev/src/coxdev_r.cpp` | R Rcpp bindings |
| `R_pkg/coxdev/inst/include/coxdev.hpp` | Symlink to `../../../../include/coxdev.hpp` |
| `coxdev/base.py` | Python `CoxDeviance`, `CoxIRLSState` wrappers |
| `coxdev/stratified.py` | Python `StratifiedCoxDeviance` wrapper |
| `coxdev/stratified_cpp.py` | Python `StratifiedCoxDevianceCpp` direct C++ wrapper |
| `R_pkg/coxdev/R/coxdev.R` | R wrappers: `make_cox_deviance()`, `make_stratified_cox_deviance()` |
| `doc/main.tex` | Score/Hessian derivations (Breslow & Efron) |
| `doc/saturated_likelihood_calc.tex` | Saturated log-likelihood derivation |
| `doc/architecture.md` | Architecture diagrams and glmnetpp integration |

## Key APIs

### Python

```python
# Constructor accepts sample_weight (init-time, not call-time)
cox = CoxDeviance(event, status, start=None, sample_weight=None, tie_breaking='efron')
result = cox(eta)  # Returns CoxDevianceResult with deviance, gradient, diag_hessian
info = cox.information(eta)  # LinearOperator for Hessian-vector products

# Stratified version
strat_cox = StratifiedCoxDeviance(event, status, strata, start=None, sample_weight=None)
result = strat_cox(eta)
```

### Stateful IRLS/Coordinate Descent Interface

```python
from coxdev import CoxDeviance, CoxIRLSState

cox = CoxDeviance(event, status, sample_weight=weights)
irls = CoxIRLSState(cox)

# Outer IRLS loop
irls.recompute_outer(eta)        # O(15n), once per outer iteration
w = irls.working_weights()       # O(1), accessor
z = irls.working_response()      # O(1), accessor

# Inner coordinate descent loop
for j in range(p):
    grad_j, hess_jj = irls.weighted_inner_product(X[:, j])  # O(n)
    delta = grad_j / hess_jj
    beta[j] += delta
    irls.update_residuals(delta, X[:, j])  # O(n), incremental
```

### R

```r
# Constructor accepts sample_weight (init-time, not call-time)
cox <- make_cox_deviance(event, status, start = NA, sample_weight = NULL, tie_breaking = 'efron')
result <- cox$coxdev(eta)  # Returns list with deviance, gradient, diag_hessian
info <- cox$information(eta)  # Returns matvec function for Hessian-vector products

# Stratified version
strat_cox <- make_stratified_cox_deviance(event, status, strata, start = NA, sample_weight = NULL)
result <- strat_cox$coxdev(eta)

# IRLS/Coordinate Descent Interface
irls <- make_cox_irls_state(cox)

# Outer IRLS loop
irls$recompute_outer(eta)        # O(15n), once per outer iteration
w <- irls$working_weights()      # O(1), accessor
z <- irls$working_response()     # O(1), accessor

# Inner coordinate descent loop
for (j in 1:p) {
  x_j <- X[, j]
  gh <- irls$weighted_inner_product(x_j)  # O(n)
  delta <- gh["gradient"] / gh["hessian"]
  beta[j] <- beta[j] + delta
  irls$update_residuals(delta, x_j)  # O(n), incremental
}
```

### C++ (header-only)

```cpp
#include "coxdev.hpp"
// Structs: CoxPreprocessed, CoxWorkspace, StratifiedCoxData, CoxIRLSState*
// Use namespace coxdev:: (or glmnetpp::coxdev:: if GLMNET_INTERFACE defined)
```

## Architecture Layers

```
Layer 1: CoxPreprocessed  - event_order, risk sets, tie groups (computed once per dataset)
Layer 2: CoxIRLSState     - exp(eta), risk_sums, working weights/response (once per outer IRLS)
Layer 3: CD Primitives    - coordinate_gradient, update_residual (per inner iteration)
```

## Current Status

- **Python IRLS interface**: Complete (`CoxIRLSState` in `coxdev/base.py`)
- **R IRLS interface**: Complete (`make_cox_irls_state()` in `R_pkg/coxdev/R/coxdev.R`)
- **glmnetpp integration**: Not yet started (Phase 5)

## Design Principles

### 1. Independence Constraint (CRITICAL)

**coxdev must NOT depend on glmnetpp.** Dependency is one-way: glmnet depends on coxdev.

- coxdev may only depend on: Eigen, standard C++, other coxdev classes
- coxdev must NOT depend on: glmnetpp classes, elastic net logic, glmnet-specific code
- Deployment: copy `include/coxdev.hpp` to glmnet unmodified, define `GLMNET_INTERFACE` before including

### 2. Single Source of Truth

`include/coxdev.hpp` is the only C++ implementation. Bindings are thin wrappers:
- Python: `src/coxdev_py.cpp` uses pybind11 array mapping
- R: `R_pkg/coxdev/src/coxdev_r.cpp` uses `Eigen::Map` for zero-copy

### 3. Buffer Access Annotations

Document RO (read-only), RW (read-write), RW-PERSIST (preserved across calls), SCRATCH (temporary):
```cpp
Eigen::VectorXd grad_buffer;      // RW: output
Eigen::VectorXd exp_w_buffer;     // RW-PERSIST: reused in hessian_matvec
Eigen::VectorXd forward_scratch;  // SCRATCH: temporary
```

### 4. Unified Code Paths

Stratified code with n strata naturally handles n=1 (unstratified). No separate implementations.

### 5. Thread Safety via Isolation

Each `CoxDeviance` instance (Python class) or closure (R) has its own `StratifiedCoxData` with dedicated workspace buffers, enabling safe parallel execution.

### 6. Empty Strata Optimization

Pass empty strata vector for unstratified models:
- Python: `np.array([], dtype=np.int32)`
- R: `integer(0)`

## Algorithm Notes

**DO NOT simplify the algorithm.** Cox deviance involves complex cumsum operations, Efron correction terms (C_01, C_02, C_11, C_21, C_22), and careful gradient/Hessian calculations. See `include/coxdev.hpp`.

Key functions:
- `preprocess_single_stratum()`, `compute_sat_loglik_stratum()`
- `cox_dev_single_stratum()` (COMPLEX), `hessian_matvec_single_stratum()`

Tie-breaking: **Efron** (default, more accurate) or **Breslow** (glmnet compatible).

## Related Projects

- glmnet repo: `/Users/naras/GitHub/glmnet`
- glmnet working copy: `/Users/naras/research/glmnet/pkg_src`
