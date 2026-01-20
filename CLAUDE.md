# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Focus

Stratified Cox C++ implementation is complete. Both Python and R use a single unified C++ codebase (`coxdev_strata.cpp`) that handles stratified and unstratified models (unstratified = single stratum).

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
  - `inst/include/coxdev_strata.h` - Struct definitions (CoxPreprocessed, CoxWorkspace, StratifiedCoxData)
  - `src/coxdev_strata.cpp` - **Single C++ source** for both Python and R (stratified implementation)
  - `R/coxdev.R` - `make_cox_deviance()` and `make_stratified_cox_deviance()` R wrappers

- **`doc/cox_calculations.tex`** - LaTeX document showing complete calculations used in current implementation

### Key Design Patterns

1. **Unified C++ Core**: A single file `coxdev_strata.cpp` implements all Cox deviance computations for both Python (via pybind11) and R (via Rcpp). Unstratified models use n_strata=1.

2. **Pre-allocated Buffers**: `CoxWorkspace` structs allocate all working memory during preprocessing to avoid repeated allocations during optimization loops.

3. **Thin Python Wrappers**: Both `CoxDeviance` and `StratifiedCoxDeviance` are thin wrappers around the C++ `StratifiedCoxDevianceCpp` class.

4. **LinearOperator Pattern**: Information matrices are returned as `scipy.sparse.linalg.LinearOperator` objects for efficient matrix-vector products without forming the full dense matrix.

## Code Design Principles

### 1. Buffer Documentation: Mark RO vs RW Access

In workspace structs, clearly document which buffers are read-only (RO) vs read-write (RW) during computation:

```cpp
struct CoxWorkspace {
    // RO after preprocessing - do not modify during computation
    Eigen::VectorXd scaling;           // RO: tie-breaking scaling factors

    // RW during computation - modified by algorithms
    Eigen::VectorXd grad_buffer;       // RW: gradient output
    Eigen::VectorXd exp_w_buffer;      // RW: intermediate exp(eta)*weight

    // Scratch buffers - temporary storage, contents not preserved
    Eigen::VectorXd forward_scratch;   // SCRATCH: temporary for forward pass
};
```

This prevents accidental reuse of buffers that would cause correctness issues (as happened when attempting to reuse `eta_local_buffer` for hessian_matvec).

### 2. Thread Safety: Closure Pattern for Parallelization

Use closure patterns to encapsulate mutable state and guard against race conditions when parallelization is enabled (e.g., cross-validation, strata handling):

**R Pattern** (closures with private environments):
```r
make_cox_deviance <- function(event, status, ...) {
    # Private state in closure environment
    last_eta <- NULL
    preprocessed_data <- preprocess(event, status, ...)

    coxdev <- function(eta, weights) {
        # Each call uses its own workspace via preprocessed_data
        result <- .Call(internal_coxdev, preprocessed_data, eta, weights)
        last_eta <<- eta  # Safe: only this closure modifies last_eta
        result
    }

    list(coxdev = coxdev, ...)
}
```

**Python Pattern** (class instances with private state):
```python
class CoxDeviance:
    def __init__(self, event, status, ...):
        self._preprocessed = preprocess(event, status, ...)
        self._last_eta = None  # Instance-private state

    def __call__(self, eta, weights):
        # Each instance has isolated state
        result = internal_coxdev(self._preprocessed, eta, weights)
        self._last_eta = eta
        return result
```

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

Complete and unified. A single C++ file (`coxdev_strata.cpp`) serves both Python and R. The previous `coxdev.cpp` was deleted after consolidating all functionality into the stratified implementation. Performance optimizations using pre-allocated workspace buffers provide 31-235x speedup in R and 22x mean speedup in Python compared to loop-based approaches.

### Files Structure

| File | Purpose |
|------|---------|
| `R_pkg/coxdev/inst/include/coxdev_strata.h` | Struct definitions (CoxPreprocessed, CoxWorkspace, StratifiedCoxData) |
| `R_pkg/coxdev/src/coxdev_strata.cpp` | **Single C++ source** - all Cox computations for both Python and R |
| `coxdev/base.py` | Python `CoxDeviance` (uses stratified C++ with n_strata=1) |
| `coxdev/stratified.py` | Python `StratifiedCoxDeviance` wrapper |
| `R_pkg/coxdev/R/coxdev.R` | R wrappers: `make_cox_deviance()`, `make_stratified_cox_deviance()` |
| `tests/test_stratified_cpp.py` | Python tests comparing C++ vs reference |
| `R_pkg/coxdev/tests/testthat/test-stratified.R` | R tests comparing against survival::coxph |

### Key Structs (in coxdev_strata.h)

```cpp
template <class ValueType = double, class IndexType = int>
struct CoxPreprocessed {
    Eigen::VectorXi event_order, start_order, status, first, last;
    Eigen::VectorXd scaling, original_scaling, event, start;
    Eigen::VectorXi event_map, start_map;
    bool have_start_times, efron;
    int n;
};

template <class ValueType = double>
struct CoxWorkspace {
    Eigen::VectorXd grad_buffer, diag_hessian_buffer, diag_part_buffer;
    Eigen::VectorXd exp_w_buffer, T_1_term, T_2_term, w_avg_buffer;
    Eigen::VectorXd forward_scratch_buffer, hess_matvec_buffer;
    Eigen::VectorXd effective_cluster_sizes, zero_weight_mask;
    std::vector<Eigen::VectorXd> risk_sum_buffers;        // 2 buffers, length n
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

**DO NOT simplify the algorithm.** The Cox deviance computation involves complex cumsum operations via `forward_prework()`, multiple C_01, C_02, C_11, C_21, C_22 terms for Efron correction, and careful gradient/Hessian calculation. See `coxdev_strata.cpp` for the algorithm.

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

1. The file `doc/cox_calculations.tex` shows the complete calculations that inform the current implementation.
