# Changelog

## 0.4.1

- **The C++ library is in `namespace coxdev`.** `coxdev::CoxDeviance`,
  `coxdev::StratifiedCoxDeviance`, `coxdev::BaselineHazard`,
  `coxdev::Concordance`, `coxdev::concordance()` and the helper functions;
  `coxdev_detail` is now `coxdev::detail`, and the header's macros are
  `COXDEV_ERROR`, `COXDEV_MAP_Xd` and `COXDEV_MAP_Xi` (`EIGEN_REF` is gone).
  `coxdev.h` no longer declares `namespace py = pybind11` either. Code that
  includes `coxdev.h` (new in 0.4.0) qualifies the names, or says
  `using namespace coxdev;`. The Python, R, C and Julia interfaces are
  unchanged.
- The R package is licensed under the GPL-2 (it was GPL (>= 3)). The
  Python package and the C++ library stay BSD-3-Clause.

## 0.4.0 (2026-10-23)

### New

- **Header-only C++.** `coxdev.h` includes its implementation,
  `coxdev_impl.h`, with every definition inline, so another package compiles
  coxdev by including `coxdev.h` in any of its source files. The Python
  package installs both headers, and `coxdev.get_include()` returns their
  directory; the R package has them in `inst/include`.
  `cpp/src/coxdev.cpp` is kept, compiling nothing, for builds that list it
  (#41).
- **The derivative of the information in a direction.**
  `CoxDeviance.information_derivative(linear_predictor, direction)` (and
  `CoxDeviance$information_derivative()` in R) is
  `d/dt I(linear_predictor + t * direction)` as a symmetric matrix-free
  operator, `O(n)` per product, with Efron's and Breslow's ties, start
  times, zero weights and strata: for the derivatives of log determinants,
  as in REML (#62).
- **`coxdev.jax`.** The deviance written in JAX (`pip install coxdev[jax]`),
  so its derivatives of any order come from automatic differentiation; it
  matches the C++ code with every kind of ties, start times, zero weights
  and strata (#62).
- **A C API** (`cpp/include/coxdev_c_api.h`, `cpp/capi/coxdev_c.cpp`): an
  `extern "C"` interface with an opaque handle over `CoxDeviance` /
  `StratifiedCoxDeviance`, for foreign-function interfaces (Julia, Rust, the
  JVM, ctypes). It covers construction (start times, strata, weights, Efron or
  Breslow ties), the deviance with its gradient and diagonal Hessian,
  information-vector products, the baseline hazard, the C-index, score
  residuals, wild scores and `sample_score` (the same draws as Python and R
  for a seed, to rounding). Errors are returned as codes, with `coxdev_last_error()`.
  `cpp/capi/build_local.sh` builds `libcoxdev_c` with CMake; `test_c_api.c`
  checks it from C, and `tests/test_c_api.py` checks every function against
  the Python package (skipped unless the library is built).
- **The C API reuses its last evaluation** while `eta` and the weights are
  unchanged (compared by value), as the Python package does: information
  products after `coxdev_deviance` no longer recompute the deviance (4x
  faster), and `eta` and the weights are compared in place rather than copied.
- **The information derivative in the C API** (C API version 2):
  `coxdev_information_derivative_matvec`, and in CoxDev.jl
  `information_derivative(cd, eta, direction)`, a symmetric `LinearMap`.
- **CoxDev.jl** (`julia/CoxDev.jl`): a Julia package over the C API, with
  the Python package's API (`CoxDeviance`, `deviance`, `information` as a
  `LinearMap`, `baseline_hazard`, `concordance`, `score_residuals`,
  `wild_score`, `sample_score`) and semantics (`weight` defaults to ones;
  strata by label). Tested against fixtures from the Python package
  (`tools/make_julia_fixtures.py`): the same values to rounding and the same
  draws for the same seed. A draft BinaryBuilder recipe for a `coxdev_jll`
  is in `julia/CoxDev.jl/deps/`.

## 0.3.1 (2026-10-21)

### Fixes

- **Windows wheels import.** 0.3.0's Windows wheels were built with a MinGW
  gcc that meson found on the build machine's PATH. Their extension needs
  MinGW's runtime DLLs (`libstdc++-6.dll`, `libgcc_s_seh-1.dll`,
  `libwinpthread-1.dll`), which the wheels don't ship, so importing coxdev
  failed with "DLL load failed while importing coxc". The wheels are now
  built with MSVC, with its C++ runtime linked statically (so they need no
  Visual C++ redistributable), and each is tested to import and compute a
  deviance.
  The tests also run on Windows (#59).

## 0.3.0 (2026-10-15)

The C++ library computes everything on n-vectors in the linear predictor, with the data sorted once. The Python and R packages share it and have the same interface.

### New

All of these are in C++, Python and R, and agree with R's `survival` (and `glmnet`). Weights, start times, strata and both Breslow's and Efron's ties are supported throughout.

- **Score residuals** (`score_residuals`), one covariate at a time; with `wild_score`, for robust (sandwich) variances and wild bootstraps (#35, #40, #42).
- **The baseline hazard** (`baseline_hazard`): Breslow's estimate, or Efron's for Efron's ties (#35, #40).
- **Schoenfeld residuals** (`schoenfeld_residuals`) and **risk-set variances** (`risk_set_variance`), the `V_z` of Grambsch and Therneau's scaled residuals (#48).
- **Expected events** and **survival at exit** (`expected_events`, `survival_at_exit`). `status − expected_events` is the martingale residual (#48).
- **Harrell's C-index** (`c_index`), O(n log n), from glmstar (#53).
- **R package** in `R/coxdev`, with the interface of the Python package (#33).

### Changes and fixes

- **Python information operator:** products with a matrix now run in one C++ call (#50). Three fixes:
  - an operator now keeps its own linear predictor after other calls;
  - results are no longer overwritten by later calls;
  - `I.T` and `I.H` now work.
- **Python strata** can be any labels, as in R (#51).
- **Caching** compares arrays instead of hashing them, and joblib is no longer a dependency (#43).
- **Preprocessing** breaks ties in the sort by index (#34).

### Documentation

https://jonathan-taylor.github.io/coxdev/ is a Jupyter Book with (#45, #46, #47, #49):
- worked examples in Python and R;
- a Computations chapter with the derivations;
- examples on penalized fits, checking proportional hazards and residuals;
- a Python API reference.

**Full Changelog**: https://github.com/jonathan-taylor/coxdev/compare/v0.2.0...v0.3.0

## Earlier versions

See the [GitHub releases](https://github.com/jonathan-taylor/coxdev/releases).
