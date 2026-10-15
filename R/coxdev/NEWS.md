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
