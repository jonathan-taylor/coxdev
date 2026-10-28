# coxdev (development version)

* `CoxDeviance$survfit(linear_predictor, sample_weight, new_linear_predictor,
  new_strata, center)`: survival curves at a fixed linear predictor, as
  `survival::survfit.coxph` (`se.fit = FALSE`) of `coxph(y ~
  offset(linear_predictor))`: the `time`, `n_risk`, `n_event`, `n_censor`,
  and the `cumulative_hazard` and `survival` of each curve.
* `CoxDeviance$martingale_residuals()` and `$deviance_residuals()`:
  `residuals.coxph`'s `"martingale"` and `"deviance"`.
* The result of `compute()` has `loglik`, the log partial likelihood
  (`coxph`'s `loglik`).

# coxdev 0.4.1

* The C++ library is in `namespace coxdev` (`coxdev::CoxDeviance` etc.), so
  a package using it with `LinkingTo: coxdev` shares no global names with it.
* The R package is licensed under the GPL-2 (it was GPL (>= 3)).

# coxdev 0.4.0

* `CoxDeviance$information_derivative(linear_predictor, direction,
  sample_weight)`: the derivative of the information in a direction of the
  linear predictor, `d/dt I(eta + t direction)`, as a symmetric matrix-free
  operator (`CoxInformationDerivative`, with `%*%`, `t()`, `dim()` and
  `as.matrix()`), `O(n)` per product: for the derivatives of log
  determinants, as in REML.
* The C++ library is header-only: `inst/include` has `coxdev.h` and its
  implementation `coxdev_impl.h`, all inline, so another package can compile
  against it with `LinkingTo: coxdev` and `#include <coxdev.h>`.
* glmnet is no longer in Suggests: the comparisons with it (and the
  vignettes' use of it) are removed, as glmnet will depend on coxdev
  (`LinkingTo: coxdev`). The comparisons with survival remain.
* More tests, mirroring the Python package's: the residuals, the baseline
  hazard, the expected events and the risk-set variances against direct
  computations, with zero weights; the C-index against the count of its
  definition and `glmnet::Cindex`; strata (a model per stratum, the block
  diagonal information, labels, strata of unequal sizes against `coxph`);
  zero weights as dropped observations, for every kind of ties; the
  caching of results; the problematic case of `tests/test_bad.py`; the
  moments of the `"fisher"` draws; edge cases (no failures, one failure,
  everyone failing at once, entries at failure times, strata without
  failures, extreme linear predictors) and finite differences of the
  derivatives under every option; and the values of the Python package,
  draws of `sample_score()` included, for a fixed data set.

# coxdev 0.3.0

The first release of the R package. It runs the C++ library of the Python
package coxdev 0.3.0 and has the same interface: `CoxDeviance$new(...)` in
place of `CoxDeviance(...)`.

* `CoxDeviance$new()` takes the response as vectors or as a `Surv` object,
  with start times, strata (any labels), sample weights, and Breslow's or
  Efron's ties.
* `compute()`: the deviance, its gradient and the diagonal of its Hessian
  in the linear predictor; `information()`: the information as a
  matrix-free operator.
* `sample_score()` and `wild_score()`: draws of the score (multinomial,
  Bernoulli, Fisher's noncentral hypergeometric, wild).
* `score_residuals()`, `schoenfeld_residuals()` and `risk_set_variance()`,
  a column at a time for a dense or sparse design.
* `baseline_hazard()`, `expected_events()` and `survival_at_exit()`.
* `c_index()`: Harrell's concordance index, in O(n log n).

All agree with the survival package, and the vignettes match the Python
notebooks.
