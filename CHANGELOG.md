# Changelog

## 0.3.1

### New

- **Header-only C++.** `coxdev.h` includes its implementation,
  `coxdev_impl.h`, with every definition inline, so another package compiles
  coxdev by including `coxdev.h` in any of its source files. The Python
  package installs both headers, and `coxdev.get_include()` returns their
  directory; the R package has them in `inst/include`.
  `cpp/src/coxdev.cpp` is kept, compiling nothing, for builds that list it
  (#41).

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
