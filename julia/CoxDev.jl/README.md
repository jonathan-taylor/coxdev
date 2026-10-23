# CoxDev.jl

Julia access to [coxdev](https://github.com/jonathan-taylor/coxdev): the deviance, gradient, diagonal Hessian and information of the Cox proportional hazards model. It covers:
- Efron's or Breslow's ties;
- `(start, event]` data;
- strata and weights;
- the baseline hazard, Harrell's C-index, score residuals and score draws (wild bootstrap and model-based).

It calls coxdev's C++ library through its C API (`cpp/include/coxdev_c_api.h`). That is the same code as the Python and R packages, so it gives the same numbers, and the same draws for the same seed (to rounding: across platforms, math libraries can differ in the last bit).

**Status:** an unregistered prototype on coxdev's `julia` branch.

## Use

```julia
using CoxDev
cd  = CoxDeviance(event, status; start=nothing, weight=nothing, ties=:efron, strata=nothing)
r   = deviance(cd, eta; weight=w)       # (deviance, gradient, diag_hessian, loglik_sat)
I   = information(cd, eta; weight=w)    # LinearMap: I * v, X' * (I * X)
bh  = baseline_hazard(cd, eta; weight=w)
c   = concordance(cd, eta; weight=w)        # within strata; also concordance(pred, event, status; start, weight)
U   = score_residuals(cd, eta, X; weight=w)
r   = wild_score(cd, eta, multipliers; weight=w)
S   = sample_score(cd, eta; weight=w, method="multinomial", n_draws=100, seed=1)
```

**API.** The names and meanings mirror coxdev's Python `CoxDeviance`:
- **`weight` defaults to ones** in every evaluation, whatever the weights given at construction. Construction weights only fix the data.
- **Strata** may be any labels `sort` can order. `baseline_hazard` reports each stratum by its label.
- **Caching.** The last `deviance` result is reused while `eta` and `weight` are unchanged; don't modify its arrays. The C library also reuses its last evaluation, so information products at that `eta` cost one product each.
- **Threads.** A `CoxDeviance` holds a lock, so tasks can share one, but calls are serialized. Use one object per task for parallel work.

## Development

The package needs the shared library `libcoxdev_c`.

```sh
cpp/capi/build_local.sh                 # builds cpp/capi/build/libcoxdev_c.{dylib,so} (CMake, C++17)
cd julia/CoxDev.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

**Finding the library:** `libpath()` uses `ENV["COXDEV_C_LIB"]` if set, else `cpp/capi/build` of the enclosing coxdev checkout.

**Test fixtures:** the files in `test/fixtures/` come from the Python package:

```sh
python tools/make_julia_fixtures.py
```

It covers 36 cases: Efron/Breslow × right-censored/`(start, event]` × no strata, integer or string labels × unit, random or zero weights.

**Benchmarks:** `bench/` compares the per-call times of `deviance` and the information product with Python coxdev.

## Releasing (not done)

1. **`coxdev_jll`.** Submit `deps/build_tarballs.jl` to [Yggdrasil](https://github.com/JuliaPackaging/Yggdrasil) as `C/coxdev/build_tarballs.jl`. It builds `libcoxdev_c` for all platforms against `Eigen_jll`, and must be rebuilt for each coxdev release. Then, in `src/CoxDev.jl`:
   ```julia
   using coxdev_jll: libcoxdev_c
   libpath() = get(ENV, "COXDEV_C_LIB", libcoxdev_c)
   ```
   Add `coxdev_jll` to `[deps]` and `[compat]`.
2. **Registration.**
   - Register `CoxDev.jl` in the General registry with [Registrator](https://github.com/JuliaRegistries/Registrator.jl). The package lives in the `julia/CoxDev.jl` subdirectory of the coxdev repository, which Registrator supports with `subdir`.
   - Version it independently of coxdev, with `[compat] coxdev_jll` pinning the C library.
3. **CI.** Add [julia-actions](https://github.com/julia-actions) workflows:
   - `setup-julia`;
   - `cache`;
   - `julia-buildpkg` (after building `libcoxdev_c` with `cpp/capi/build_local.sh`, setting `COXDEV_C_LIB`);
   - `julia-runtest`.

   Run them on Linux, macOS and Windows, with Julia 1.9 and the latest release.
