# coxdev

`coxdev` evaluates the objective of the Cox proportional hazards model, the
deviance, together with its gradient and Hessian with respect to the linear
predictor, as a stand-alone class. The computations are in C++ (with Eigen),
shared by a Python package and an R package with the same interface.

## Why a stand-alone class

Most generalized linear models are easy to fit with a generic solver: the
log-likelihood is a sum over observations, so its gradient and Hessian with
respect to the linear predictor $\eta$ are elementwise. The Cox partial
likelihood is not. Each failure contributes the log of a sum over its risk
set, so every observation is coupled to every other observation at risk at
the same time. Computed naively, the gradient costs $O(n^2)$, and getting
ties, left truncation (start times) and weights right is fiddly.

`coxdev` packages these computations once:

- **The data are sorted once.** A `CoxDeviance` object is built from the
  response: event (stop) times, status, optional start times, weights and
  strata, and Efron's or Breslow's handling of ties. All the sorting and
  bookkeeping happens at construction. Each evaluation at a linear predictor
  is then $O(n)$, a few cumulative sums over the sorted data.
- **It works in $\eta$, not in coefficients.** Every method takes and returns
  vectors of length $n$: the deviance, its gradient, the diagonal of its
  Hessian, and products of the full Hessian (the information) with a vector,
  as a matrix-free linear operator. The design matrix, whether dense, sparse
  or never formed, is left to the caller. A solver gets the gradient in the
  coefficients as $X^T \nabla_\eta$ and Hessian products as $X^T H X v$.
- **It is solver-agnostic.** The same object serves Newton's method, IRLS with
  the diagonal of the Hessian as working weights (as glmnet does),
  conjugate gradients with Hessian products, or inference at a fitted model.
  [glmstar](https://github.com/jonathan-taylor/glmstar), the Python glmnet,
  and R's `glmnet` both compute their Cox models with `coxdev`.

The [Computations](notes_no_ties.md) chapter derives the score and Hessian,
with ties, start times and zero weights, and the cumulative sums that make
each evaluation $O(n)$.

## Additional tools

Beyond the objective and its derivatives, a `CoxDeviance` object gives some
quantities that come cheaply from the same sorted data:

- **`sample_score`**: draws of the score from the fitted model. The failures
  at each event time are redrawn (multinomial, Bernoulli, or Fisher's
  noncentral hypergeometric), or the score residuals are resampled (a wild
  bootstrap). Draws are reproducible from a seed on every platform. See
  [Sampling the score](sampling.md).
- **`score_residuals`**: each observation's contribution to the score of a
  design $X$, centered within each risk set. Their crossproduct is the middle
  of the robust (sandwich) variance. For wild-bootstrap draws $U^T G$ there
  is no need to form $U$: `wild_score` gives them in one $O(n)$ call.
- **`baseline_hazard`**: the estimate of the baseline hazard at each distinct
  failure time, within each stratum, for the linear predictor as given. With
  Breslow's ties it is Breslow's estimate. With Efron's ties, the $d$ tied
  failures at a time are spread over Efron's shrinking risk sets, as in the
  likelihood: the increment is
  $\sum_{r=0}^{d-1} \bar{w} / \left(S - \tfrac{r}{d} D\right)$, where $S$ is
  the sum of $w e^\eta$ over the risk set, $D$ its sum over the tied
  failures, and $\bar{w}$ their mean weight. This is what R's
  `survival::basehaz` gives for a `coxph` fit with `ties = "efron"`, and the
  tests check it against that.
- **`schoenfeld_residuals`**: for each failure, its covariates less their
  mean over its risk set (Schoenfeld, 1982), with Efron's ties as in R's
  `survival`. Plotted against time, they check the proportional hazards
  assumption (Grambsch and Therneau, 1994). Like `score_residuals`, they are
  computed one covariate, an $n$-vector, at a time.
- **`risk_set_variance`**: for each failure, the variance of a covariate
  over its risk set, the $V_z$ by which Grambsch and Therneau (1994) scale
  the Schoenfeld residuals.
- **`c_index`**: Harrell's concordance index of a linear predictor
  (Harrell et al., 1982), in $O(n \log n)$: R's `survival::concordance`
  and `glmnet::Cindex`.
- **`expected_events`** and **`survival_at_exit`**: the expected number of
  events of each observation over its time at risk, $e^{\eta_i}$ times the
  increase of the baseline cumulative hazard, and the expected survival
  probability at exit, $\exp(-\text{expected})$. `status - expected_events`
  is the martingale residual (Barlow and Prentice, 1988; Therneau, Grambsch
  and Fleming, 1990). These are `predict(fit, type = "expected")` and
  `predict(fit, type = "survival")` of a `coxph` fit.

The README lists the references.

The [Examples](examples/penalized_fits.md) fit penalized models with
general-purpose solvers, check proportional hazards with the Schoenfeld
residuals, and check a model with martingale and Cox–Snell residuals. The
[API reference](https://jonathan-taylor.github.io/coxdev/api/) documents
every method.

[Using coxdev](using_coxdev.md) shows these in a worked example, and
[Using coxdev from R](r_using_coxdev.md) does the same with the R package,
which has the same interface: `CoxDeviance$new(...)` in place of
`CoxDeviance(...)`.

## Installation

The Eigen headers are a git submodule:

```bash
git clone --recurse-submodules https://github.com/jonathan-taylor/coxdev.git
cd coxdev
pip install .
```

For an existing clone, run `git submodule update --init --recursive` first.
To use an Eigen installed elsewhere:

```bash
env EIGEN_LIBRARY_PATH=/path/to/eigen pip install .
```

The notebooks here also use pandas, matplotlib and
[glmstar](https://pypi.org/project/glmstar/):
`pip install pandas matplotlib "glmstar>=0.2.0"`.

The R package is in `R/coxdev`. From a clone:

```bash
R CMD INSTALL R/coxdev
```

Its pages here also use survival and, if installed, glmnet (>= 5.0). They
run on the R kernel for Jupyter, from the IRkernel package.
