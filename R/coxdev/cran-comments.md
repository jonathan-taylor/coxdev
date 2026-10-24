## Submission

This is a first submission of coxdev.

coxdev computes the deviance of the Cox model, its gradient and Hessian
products, in O(n) after one sort. Its C++ library is header-only and installed
in `inst/include`, so that other packages (glmnet, next) can use it through
`LinkingTo` rather than vendoring it. The same library is used by the Python
package coxdev on PyPI.

## Test environments

* local macOS 26.6.2 (aarch64), R 4.5.3
* GitHub Actions (`.github/workflows/R-CMD-check.yaml`): macOS and Windows with
  R release; Ubuntu with R-devel, R release and R oldrel-1

## R CMD check results

0 errors | 0 warnings | 1 note

* New submission.
