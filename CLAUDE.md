# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Focus

Stratified Cox C++ implementation is complete. Both Python and R use a single unified C++ codebase (`coxdev.cpp`) that handles stratified and unstratified models (unstratified = single stratum).

## Completed Features

### Saturated Likelihood (Efron) - COMPLETE

The `compute_sat_loglik_stratum()` function in `coxdev.cpp` implements both Breslow and Efron formulas for saturated log-likelihood.

**Formula:**
```
LL_sat = -Σ_C W_C * [log(W_C) + (1/K_C+) * (log(K_C+!) - K_C+ * log(K_C+))]
```

Where:
- `C` = cluster of tied failure times
- `W_C` = total weight of individuals in cluster C: `Σ_{j∈C} w_j`
- `K_C+` = count of individuals in cluster C with **positive weights** (`w_j > 0`)
- For **Breslow** (`efron=false`): the second term vanishes, giving `-W_C * log(W_C)`
- For **Efron** (`efron=true`): includes the factorial penalty term using `lgamma(K_C+ + 1)`

**Zero-Weight Handling:** `K_C+` only counts positive-weight individuals. If `K_C+ = 0` for a cluster, it's skipped entirely.

**Documentation:** See `doc/saturated_likelihood_calc.tex` and `doc/main.tex` for complete derivations.

## Upcoming Tasks

(None currently)

### Related Projects

- **glmnet working copy**: `/Users/naras/research/glmnet/pkg_src`
- **glmnet repo**: `/Users/naras/GitHub/glmnet`

## Build & Development Commands

```bash
# Install in development mode (using Homebrew Eigen)
EIGEN_LIBRARY_PATH=/opt/homebrew/include/eigen3 uv pip install -e .

# Alternative: Initialize Eigen submodule instead of Homebrew
git submodule update --init --recursive
uv pip install -e .

# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_compareR.py

# Run a specific test
uv run pytest tests/test_compareR.py::test_simple

# Run tests with R comparison (requires rpy2, glmnet, survival R packages)
uv run pytest tests/test_compareR.py -v
```

## Architecture

This library computes Cox proportional hazards model deviance, gradients, and Hessian information matrices for survival analysis.

### Core Structure

- **`coxdev/`** - Python package
  - `base.py` - `CoxDeviance` class (thin wrapper around C++ stratified with n_strata=1) and `CoxInformation` linear operator
  - `stratified.py` - `StratifiedCoxDeviance` (thin wrapper around C++ `StratifiedCoxDevianceCpp`)
  - `coxc.cpython-*.so` - Compiled C++ extension (pybind11)

- **`R_pkg/coxdev/`** - R package with shared C++ implementation
  - `inst/include/coxdev.h` - Struct definitions (CoxPreprocessed, CoxWorkspace, StratifiedCoxData)
  - `src/coxdev.cpp` - **Single C++ source** for both Python and R (stratified implementation)
  - `R/coxdev.R` - `make_cox_deviance()` and `make_stratified_cox_deviance()` R wrappers

- **`doc/`** - LaTeX documentation
  - `main.tex` - Score and Hessian derivations for weighted Cox with ties
  - `saturated_likelihood_calc.tex` - Saturated log-likelihood derivation (Efron method)

### Key Design Patterns

1. **Unified C++ Core**: A single file `coxdev.cpp` implements all Cox deviance computations for both Python (via pybind11) and R (via Rcpp). Unstratified models use n_strata=1.

2. **Pre-allocated Buffers**: `CoxWorkspace` structs allocate all working memory during preprocessing to avoid repeated allocations during optimization loops.

3. **Thin Python Wrappers**: Both `CoxDeviance` and `StratifiedCoxDeviance` are thin wrappers around the C++ `StratifiedCoxDevianceCpp` class.

4. **LinearOperator Pattern**: Information matrices are returned as `scipy.sparse.linalg.LinearOperator` objects for efficient matrix-vector products without forming the full dense matrix.

## Code Design Principles

### 1. Buffer Documentation: Mark RO vs RW Access

In workspace structs, clearly document which buffers are read-only (RO) vs read-write (RW) during computation:

```cpp
struct CoxWorkspace {
    // RW: Output buffers - primary outputs scattered to caller
    Eigen::VectorXd grad_buffer;           // RW: gradient output (native order)
    Eigen::VectorXd diag_hessian_buffer;   // RW: diagonal Hessian output

    // RW-PERSIST: Intermediate values preserved across deviance/hessian_matvec
    Eigen::VectorXd exp_w_buffer;          // RW-PERSIST: weight * exp(eta)
    Eigen::VectorXd diag_part_buffer;      // RW-PERSIST: exp_eta_w * T_1_term

    // SCRATCH: Temporary storage, contents not preserved between calls
    Eigen::VectorXd forward_scratch;       // SCRATCH: temporary for cumsum steps
};
```

Note: `CoxPreprocessed` fields (scaling, first, last, etc.) are all **RO** after preprocessing.

This prevents accidental reuse of buffers that would cause correctness issues (as happened when attempting to reuse `eta_local_buffer` for hessian_matvec).

### 2. Thread Safety: Closure Pattern for Parallelization

Use closure patterns to encapsulate mutable state and guard against race conditions when parallelization is enabled (e.g., cross-validation, strata handling):

**R Pattern** (closures with private environments):
```r
make_cox_deviance <- function(event, status, ...) {
    # Private state in closure environment - each closure has isolated workspace
    strat_data_ptr <- .preprocess_stratified(...)

    coxdev <- function(eta, weights) {
        # Each call uses its own workspace via strat_data_ptr
        .cox_dev_stratified(strat_data_ptr, eta, weights)
    }

    list(coxdev = coxdev, ...)
}
```

**Python Pattern** (class instances with private state):
```python
class CoxDeviance:
    def __init__(self, event, status, ...):
        # Each instance has isolated C++ object with its own workspace
        self._cpp = StratifiedCoxDevianceCpp(event, status, ...)

    def __call__(self, eta, weights):
        # Each instance has isolated state
        return self._cpp(eta, weights)
```

The key principle is **isolation**: each CoxDeviance instance (Python) or closure (R) has its own C++ `StratifiedCoxData` object with dedicated workspace buffers. This allows safe parallel execution (e.g., cross-validation folds) without race conditions.

### 3. Avoid Code Bloat: Unified Code Paths

Write code that handles general cases without special-casing. For example, stratified code with `n` strata should naturally handle `n=1` (unstratified):

**DO**: Single implementation that works for all cases
```cpp
// Stratified implementation handles n_strata >= 1
double cox_dev_stratified(data, eta, weights) {
    double total = 0.0;
    for (int s = 0; s < data.n_strata; ++s) {  // Works for n_strata=1
        total += cox_dev_single_stratum(...);
    }
    return total;
}
```

**DON'T**: Duplicate code paths
```cpp
// Avoid: separate implementations for stratified vs unstratified
double cox_dev(data, eta, weights) {
    if (data.n_strata == 1) {
        return cox_dev_unstratified(...);  // Code duplication!
    } else {
        return cox_dev_stratified(...);
    }
}
```

Benefits:
- Less code to maintain
- Single code path to test and debug
- Consistent behavior across all configurations

### Tie-Breaking Methods

- **Efron** (default): More accurate for data with tied event times
- **Breslow**: Simpler approximation, compatible with glmnet

### Testing

Tests in `tests/test_compareR.py` validate against R's `survival::coxph` and `glmnet` packages. The `simulate.py` module generates test data with various tie patterns. Tests require `rpy2` and R packages `glmnet` and `survival`.

## Stratified Cox C++ Implementation

### Current Status

Complete and unified. A single C++ file (`coxdev.cpp`) serves both Python and R, handling both stratified and unstratified models (unstratified = single stratum). Performance optimizations using pre-allocated workspace buffers provide 31-235x speedup in R and 22x mean speedup in Python compared to loop-based approaches.

### Files Structure

| File | Purpose |
|------|---------|
| `R_pkg/coxdev/inst/include/coxdev.h` | Struct definitions (CoxPreprocessed, CoxWorkspace, StratifiedCoxData) |
| `R_pkg/coxdev/src/coxdev.cpp` | **Single C++ source** - all Cox computations for both Python and R |
| `coxdev/base.py` | Python `CoxDeviance` (uses stratified C++ with n_strata=1) |
| `coxdev/stratified.py` | Python `StratifiedCoxDeviance` wrapper |
| `R_pkg/coxdev/R/coxdev.R` | R wrappers: `make_cox_deviance()`, `make_stratified_cox_deviance()` |
| `tests/test_stratified_cpp.py` | Python tests comparing C++ vs reference |
| `R_pkg/coxdev/tests/testthat/test-stratified.R` | R tests comparing against survival::coxph |

### Key Structs (in coxdev.h)

```cpp
// All CoxPreprocessed fields are RO (read-only) after preprocessing
template <class ValueType = double, class IndexType = int>
struct CoxPreprocessed {
    Eigen::VectorXi event_order, start_order, status, first, last;  // RO
    Eigen::VectorXd scaling, original_scaling, event, start;        // RO
    Eigen::VectorXi event_map, start_map;                           // RO
    bool have_start_times, efron;                                   // RO
    int n;                                                          // RO
};

// CoxWorkspace buffers: RW (output), RW-PERSIST (reused across calls), SCRATCH (temporary)
template <class ValueType = double>
struct CoxWorkspace {
    // RW: Output buffers scattered to caller
    Eigen::VectorXd grad_buffer, diag_hessian_buffer, matvec_result_buffer;

    // RW-PERSIST: Computed in cox_dev, reused in hessian_matvec
    Eigen::VectorXd diag_part_buffer, exp_w_buffer, w_avg_buffer;
    Eigen::VectorXd eta_local_buffer, weight_local_buffer;
    std::vector<Eigen::VectorXd> risk_sum_buffers;  // [0]=RW-PERSIST, [1]=SCRATCH

    // RW: Zero-weight handling (computed per-call when zero weights exist)
    Eigen::VectorXd effective_cluster_sizes, zero_weight_mask;
    bool use_zero_weight_handling;

    // SCRATCH: Temporary buffers, contents not preserved
    Eigen::VectorXd T_1_term, T_2_term, forward_scratch_buffer, hess_matvec_buffer;
    std::vector<Eigen::VectorXd> forward_cumsum_buffers;  // 5 buffers, length n+1
    std::vector<Eigen::VectorXd> reverse_cumsum_buffers;  // 4 buffers, length n+1
    std::vector<Eigen::VectorXd> event_reorder_buffers;   // 3 buffers, length n
};

template <class ValueType = double, class IndexType = int>
struct StratifiedCoxData {
    int n_strata, n_total;
    std::vector<int> strata_labels;
    std::vector<std::vector<int>> stratum_indices;  // global indices per stratum
    std::vector<CoxPreprocessed<ValueType, IndexType>> preproc;
    std::vector<CoxWorkspace<ValueType>> workspace;
    std::vector<double> loglik_sat;
    std::vector<bool> efron_stratum;
};
```

### Interface Differences (Python vs R)

| Aspect | Python | R |
|--------|--------|---|
| Data management | `StratifiedCoxData` stored as class member | `Rcpp::XPtr<StratifiedCoxData>` (external pointer) |
| Member access | `strat_data.n_strata` (dot notation) | `xptr->n_strata` (arrow notation) |
| Binding function | `bind_stratified(py::module_& m)` | Rcpp attributes `// [[Rcpp::export(.func_name)]]` |

### Maintenance Guidelines (IMPORTANT)

**DO NOT simplify the algorithm.** The Cox deviance computation involves complex cumsum operations via `forward_prework()`, multiple C_01, C_02, C_11, C_21, C_22 terms for Efron correction, and careful gradient/Hessian calculation. See `coxdev.cpp` for the algorithm.

**Key single-stratum helper functions:**
- `preprocess_single_stratum()` - preprocessing for one stratum
- `compute_sat_loglik_stratum()` - saturated log-likelihood
- `sum_over_risk_set_internal()` - risk sum computation
- `sum_over_events_internal()` - forward pass sums
- `cox_dev_single_stratum()` - deviance for one stratum (COMPLEX - do not simplify)
- `hessian_matvec_single_stratum()` - Hessian matvec for one stratum

### Performance Benchmarks (from working implementation)

C++ stratified vs R loop (evaluation only, preprocessing excluded):
- 1k obs, 50 strata: **33x speedup**
- 10k obs, 200 strata: **24x speedup**
- 50k obs, 500 strata: **31x speedup**

## Resources

1. `doc/main.tex` - Score and Hessian derivations for weighted Cox proportional hazards with ties (Breslow and Efron methods)
2. `doc/saturated_likelihood_calc.tex` - Complete derivation of saturated log-likelihood for Efron method, including zero-weight handling
