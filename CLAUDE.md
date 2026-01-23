# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

Cox proportional hazards model deviance, gradients, and Hessian computations for survival analysis. Single unified C++ codebase (`coxdev.cpp`) serves both Python (pybind11) and R (Rcpp), handling stratified and unstratified models.

## Build & Test Commands

```bash
# Install (Homebrew Eigen)
EIGEN_LIBRARY_PATH=/opt/homebrew/include/eigen3 uv pip install -e .

# Alternative: Use Eigen submodule
git submodule update --init --recursive && uv pip install -e .

# Tests
uv run pytest tests/                           # All tests
uv run pytest tests/test_compareR.py -v        # R comparison (requires rpy2, glmnet, survival)
uv run pytest tests/test_compareR.py::test_simple  # Single test
```

## File Structure

| Location | Purpose |
|----------|---------|
| `R_pkg/coxdev/inst/include/coxdev.h` | Struct definitions: `CoxPreprocessed`, `CoxWorkspace`, `StratifiedCoxData`, `CoxIRLSState*` |
| `R_pkg/coxdev/src/coxdev.cpp` | **Single C++ source** for Python and R |
| `coxdev/base.py` | Python `CoxDeviance` wrapper |
| `coxdev/stratified.py` | Python `StratifiedCoxDeviance` wrapper |
| `R_pkg/coxdev/R/coxdev.R` | R wrappers: `make_cox_deviance()`, `make_stratified_cox_deviance()` |
| `doc/main.tex` | Score/Hessian derivations (Breslow & Efron) |
| `doc/saturated_likelihood_calc.tex` | Saturated log-likelihood derivation |
| `doc/architecture.md` | Architecture diagrams and glmnetpp integration |

## Key APIs

### Stateless (backward compatible)
- `cox_dev(eta, weights)` → deviance, gradient, diagonal Hessian
- `hessian_matvec(v)` → Hessian-vector product

### Stateful (for coordinate descent integration)
```cpp
CoxIRLSStateStratified state;
state.initialize(strat_data);

// Outer IRLS loop:
state.recompute_outer(eta, weights);  // O(15n), once per outer
auto w = state.working_weights();      // O(1)
auto z = state.working_response();     // O(1)

// Inner CD loop:
auto [grad_j, hess_jj] = state.weighted_inner_product(x_j);  // O(n)
state.update_residuals(delta, x_j);  // O(n), incremental
```

## Architecture Layers

```
Layer 1: CoxPreprocessed  - event_order, risk sets, tie groups (computed once per dataset)
Layer 2: CoxIRLSState     - exp(η), risk_sums, working weights/response (once per outer IRLS)
Layer 3: CD Primitives    - coordinate_gradient, update_residual (per inner iteration)
```

## Upcoming Tasks

- **Phase 3**: R interface via `Rcpp::XPtr<CoxIRLSStateStratified>`
- **Phase 4**: Python pybind11 bindings for `CoxIRLSState` classes
- **Phase 5**: Integrate into glmnetpp's `ElnetPointInternalCox*`

## Design Principles

### 1. Independence Constraint (CRITICAL)

**coxdev must NOT depend on glmnetpp.** Dependency is one-way: glmnet depends on coxdev.

- coxdev may only depend on: Eigen, standard C++, other coxdev classes
- coxdev must NOT depend on: glmnetpp classes, elastic net logic, glmnet-specific code
- Deployment: develop in coxdev → copy `coxdev.h`/`coxdev.cpp` to glmnet unmodified

### 2. Buffer Access Annotations

Document RO (read-only), RW (read-write), RW-PERSIST (preserved across calls), SCRATCH (temporary):
```cpp
Eigen::VectorXd grad_buffer;      // RW: output
Eigen::VectorXd exp_w_buffer;     // RW-PERSIST: reused in hessian_matvec
Eigen::VectorXd forward_scratch;  // SCRATCH: temporary
```

### 3. Unified Code Paths

Stratified code with n strata naturally handles n=1 (unstratified). No separate implementations.

### 4. Thread Safety via Isolation

Each `CoxDeviance` instance (Python class) or closure (R) has its own `StratifiedCoxData` with dedicated workspace buffers, enabling safe parallel execution.

### 5. Empty Strata Optimization

Pass empty strata vector for unstratified models:
- Python: `np.array([], dtype=np.int32)`
- R: `integer(0)`

## Algorithm Notes

**DO NOT simplify the algorithm.** Cox deviance involves complex cumsum operations, Efron correction terms (C_01, C_02, C_11, C_21, C_22), and careful gradient/Hessian calculations. See `coxdev.cpp`.

Key functions:
- `preprocess_single_stratum()`, `compute_sat_loglik_stratum()`
- `cox_dev_single_stratum()` (COMPLEX), `hessian_matvec_single_stratum()`

Tie-breaking: **Efron** (default, more accurate) or **Breslow** (glmnet compatible).

## Related Projects

- glmnet repo: `/Users/naras/GitHub/glmnet`
- glmnet working copy: `/Users/naras/research/glmnet/pkg_src`
