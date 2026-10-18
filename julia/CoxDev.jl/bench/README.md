# Benchmarks

```sh
python make_data.py .                         # data_{1e4,1e5,1e6}.bin, shared by both
python bench_python.py .                      # Python coxdev: deviance, information matvec
julia --project=<env with CoxDev, JSON> bench_julia.jl . label
julia --project=<env with CoxDev> bench_cache.jl
```
The library is `ENV["COXDEV_C_LIB"]`; run once per build to compare builds.
