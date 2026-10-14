# coxdev

A high-performance C++ for computing Cox proportional hazards model deviance, gradients, and Hessian information matrices. Built with Eigen for ease of use in R and Python, this library provides efficient survival analysis computations with support for different tie-breaking methods.

## Features

- **High Performance**: C++ implementation with Eigen linear algebra library
- **Comprehensive Support**: Handles both Efron and Breslow tie-breaking methods
- **Left-Truncated Data**: Support for left-truncated survival data
- **Efficient Computations**: Optimized algorithms for deviance, gradient, and Hessian calculations
- **Memory Efficient**: Uses linear operators for large-scale computations
- **Sampling the Score**: Draws of the score from the fitted model (multinomial, Bernoulli or Fisher noncentral hypergeometric draws of the failures at each event time) or from its score residuals (a wild bootstrap), reproducible from a seed on every platform
- **Cross-Platform**: Works on Linux, macOS, and Windows

## Installation

### Prerequisites

This package requires the Eigen C++ library headers. The Eigen library is included as a git submodule.

1. **Initialize Eigen submodule**: The Eigen library is included as a git submodule. Make sure it's initialized:
   ```bash
   git submodule update --init --recursive
   ```


### Standard Installation

```bash
pip install .
```

### With Custom Eigen Path

If you have Eigen installed elsewhere, you can specify its location:
```bash
env EIGEN_LIBRARY_PATH=/path/to/eigen pip install .
```

### Development Installation

```bash
pip install .[dev]
pip install -e . --no-build-isolation
```

## Quick Start

```python
import numpy as np
from coxdev import CoxDeviance

# Generate sample survival data
n_samples = 1000
event_times = np.random.exponential(1.0, n_samples)
status = np.random.binomial(1, 0.7, n_samples)  # 70% events, 30% censored
linear_predictor = np.random.normal(0, 1, n_samples)

# Create CoxDeviance object
coxdev = CoxDeviance(event=event_times, status=status, tie_breaking='efron')

# Compute deviance and related quantities
result = coxdev(linear_predictor)

print(f"Deviance: {result.deviance:.4f}")
print(f"Saturated log-likelihood: {result.loglik_sat:.4f}")
print(f"Gradient norm: {np.linalg.norm(result.gradient):.4f}")
```

## Advanced Usage

### Left-Truncated Data

```python
# With start times (left-truncated data)
start_times = np.random.exponential(0.5, n_samples)
coxdev = CoxDeviance(
    event=event_times, 
    status=status, 
    start=start_times,
    tie_breaking='efron'
)
```

### Computing Information Matrix

```python
# Get information matrix as a linear operator
info_matrix = coxdev.information(linear_predictor)

# Matrix-vector multiplication
v = np.random.normal(0, 1, n_samples)
result_vector = info_matrix @ v

# For small problems, you can compute the full matrix
X = np.random.normal(0, 1, (n_samples, 10))
beta = np.random.normal(0, 1, 10)
eta = X @ beta

# Information matrix for coefficients: X^T @ I @ X
I = info_matrix @ X
information_matrix = X.T @ I
```

### Sampling the Score

`sample_score` draws the score at a linear predictor. Each draw has one
entry per observation, on the scale of the score `-gradient / 2`, so for a
design `X` the vector `X.T @ draw` is a draw of the score of the
coefficients. Ties, start times and strata are handled as for the
deviance.

```python
X = np.random.normal(0, 1, (n_samples, 3))
eta = X @ np.array([0.5, -0.5, 0.25])
coxdev = CoxDeviance(event=event_times, status=status, tie_breaking='efron')

# 2000 draws of the score of the coefficients
draws = X.T @ coxdev.sample_score(eta, method='multinomial', n_draws=2000, seed=1)

# their covariance is about the (Breslow) information: the ratios are near 1
I = coxdev.information(eta)
information_matrix = X.T @ (I @ X)
print(np.diag(np.cov(draws)) / np.diag(information_matrix))
```

The methods are:

| `method` | Draw | Score of the draw | Covariance |
|---|---|---|---|
| `'multinomial'` | the `d` failures at each event time drawn with replacement from the risk set, with probabilities proportional to `sample_weight * exp(eta)` | under Breslow's (1974) likelihood, of which this is the sampling model | the Breslow information |
| `'bernoulli'` | each subject at risk fails independently with probability `min(1, d * p)`, a Cox-model analogue of the weird bootstrap (Andersen et al., 1993; Burr, 1994) | Breslow's, with the number of failures drawn | less than the information when ties are heavy |
| `'fisher'` | the set of `d` failures at each event time drawn from Fisher's noncentral hypergeometric distribution (Harkness, 1965; Fog, 2008): independent Bernoullis given `d` failures, P(A) proportional to the product of `exp(eta_j)` over the set A | under the exact discrete partial likelihood (Cox, 1972; Kalbfleisch and Prentice, 2002), of which this is the sampling model, centered at its exact mean | the information of the exact likelihood |
| `'wild'` | the score residuals of the observations times independent standard normals (Lin, Wei and Ying, 1993; Kline and Santos, 2012) | of the likelihood set by `tie_breaking` (Breslow, 1974; Efron, 1977) | the sandwich, the sum of the outer products of the score residuals (Lin and Wei, 1989) |

Each model-based draw is the score of failures drawn from the model: at
each event time, the sum over the failures of `x_j` minus their expected
sum. For `'multinomial'` and `'fisher'` the draw is from the model whose
likelihood gives the score, so every draw has mean exactly zero. For
`'bernoulli'` this holds only while `d * p < 1` for every subject at risk:
the cap `min(1, d * p)` biases it when ties are heavy. The
methods differ only at tied event times: with one failure at each event
time all three draw the same failure distribution. The model-based methods
(all but `'wild'`) need equal positive weights and do not depend on
`tie_breaking`. `'wild'` does not rely on the model: its covariance is the
sandwich, which is right even when the proportional hazards model is not.

`'fisher'` samples and centers with the elementary symmetric polynomials
of the risk weights, computed on the log scale by the recursion of Gail,
Lubin and Rubinstein (1981), the one used for the exact partial likelihood
by R's survival package and SAS PROC PHREG. Each event time with `d` tied
failures among `m` at risk costs `O(m d)` once, and `O(m)` per draw.

`wild_score` gives the wild draw for given multipliers: with multipliers
`G`, `X.T @ coxdev.wild_score(eta, G)` is the sum over observations of
`G[i]` times the score residual of observation `i`. With `G` all one it is
the score.

#### References

- Andersen, P. K., Borgan, Ø., Gill, R. D. and Keiding, N. (1993).
  *Statistical Models Based on Counting Processes*. Springer, New York.
- Breslow, N. E. (1974). Covariance analysis of censored survival data.
  *Biometrics*, 30(1), 89–99.
- Burr, D. (1994). A comparison of certain bootstrap confidence intervals
  in the Cox model. *Journal of the American Statistical Association*,
  89(428), 1290–1302.
- Cox, D. R. (1972). Regression models and life-tables (with discussion).
  *Journal of the Royal Statistical Society, Series B*, 34(2), 187–220.
- Efron, B. (1977). The efficiency of Cox's likelihood function for
  censored data. *Journal of the American Statistical Association*,
  72(359), 557–565.
- Fog, A. (2008). Sampling methods for Wallenius' and Fisher's noncentral
  hypergeometric distributions. *Communications in Statistics: Simulation
  and Computation*, 37(2), 241–257.
- Gail, M. H., Lubin, J. H. and Rubinstein, L. V. (1981). Likelihood
  calculations for matched case-control studies and survival studies with
  tied death times. *Biometrika*, 68(3), 703–707.
- Harkness, W. L. (1965). Properties of the extended hypergeometric
  distribution. *The Annals of Mathematical Statistics*, 36(3), 938–945.
- Kalbfleisch, J. D. and Prentice, R. L. (2002). *The Statistical Analysis
  of Failure Time Data*, 2nd edition. Wiley, Hoboken, NJ.
- Kline, P. and Santos, A. (2012). A score based approach to wild
  bootstrap inference. *Journal of Econometric Methods*, 1(1), 23–41.
- Lin, D. Y. and Wei, L. J. (1989). The robust inference for the Cox
  proportional hazards model. *Journal of the American Statistical
  Association*, 84(408), 1074–1078.
- Lin, D. Y., Wei, L. J. and Ying, Z. (1993). Checking the Cox model with
  cumulative sums of martingale-based residuals. *Biometrika*, 80(3),
  557–572.

The random numbers come from the C++ code, from the bits of
`std::mt19937_64`, so the same `seed` gives the same draws on every
platform. With `seed=None`, a fresh seed is drawn.

### Different Tie-Breaking Methods

```python
# Efron's method (default)
coxdev_efron = CoxDeviance(event=event_times, status=status, tie_breaking='efron')

# Breslow's method
coxdev_breslow = CoxDeviance(event=event_times, status=status, tie_breaking='breslow')
```

## API Reference

### CoxDeviance

The main class for computing Cox model quantities.

#### Parameters

- **event**: Event times (failure times) for each observation
- **status**: Event indicators (1 for event occurred, 0 for censored)
- **start**: Start times for left-truncated data (optional)
- **tie_breaking**: Method for handling tied event times ('efron' or 'breslow')

#### Methods

- **`__call__(linear_predictor, sample_weight=None)`**: Compute deviance and related quantities
- **`information(linear_predictor, sample_weight=None)`**: Get information matrix as linear operator
- **`sample_score(linear_predictor, sample_weight=None, method='multinomial', n_draws=None, seed=None)`**: Draws of the score, one entry per observation: a vector if `n_draws` is None, else an `(n, n_draws)` array. See [Sampling the Score](#sampling-the-score)
- **`wild_score(linear_predictor, multipliers, sample_weight=None)`**: The wild draw of the score for given multipliers

### CoxDevianceResult

Result object containing computation results.

#### Attributes

- **linear_predictor**: The linear predictor values used
- **sample_weight**: Sample weights used
- **loglik_sat**: Saturated log-likelihood value
- **deviance**: Computed deviance value
- **gradient**: Gradient of deviance with respect to linear predictor
- **diag_hessian**: Diagonal of Hessian matrix

## Performance

The library is optimized for performance:

- **C++ Implementation**: Core computations in C++ with Eigen
- **Memory Efficient**: Reuses buffers and uses linear operators
- **Vectorized Operations**: Leverages Eigen's optimized linear algebra
- **Minimal Python Overhead**: Heavy computations done in C++

## Building from Source

### Prerequisites

- C++ compiler with C++17 support
- Eigen library headers
- For Python library: Python >= 3.9 + pybind11


## C++ API

The Python package (and the R package in `R/coxdev`) wrap the C++ classes
`CoxDeviance` and `StratifiedCoxDeviance` in `cpp/include/coxdev.h`. After
`compute_deviance(eta, weights)`, besides the gradient, the diagonal of the
Hessian, products with the information and draws of the score, they give:

- `score_residuals(X)`: for a design `X`, the `n x p` matrix of score
  residuals `U_i`, each observation's contribution to the score
  `X^T (-gradient / 2)`, centered within each risk set (with Efron's
  weights for ties if `efron`). The rows sum to the score, `sum_i G_i U_i`
  is `X^T wild_score(G)`, and `U^T U` is the middle of the robust
  (sandwich) variance (Lin and Wei, 1989).
- `baseline_hazard()`: the estimate of the baseline hazard, Breslow's
  (1972) or, with Efron's ties, Efron's (1977), at each distinct failure
  time (within each stratum): its increments and cumulative sum, for the
  linear predictor as given, so that the cumulative hazard of observation
  `i` at time `t` is `cumulative_hazard(t) * exp(eta_i)`.

`cpp/tests` has tests of these against direct computations:

```bash
c++ -std=c++17 -O2 -Ieigen -Icpp/include cpp/src/coxdev.cpp cpp/tests/test_residuals_hazard.cpp -o test_residuals_hazard
./test_residuals_hazard
```

## Testing

Most of the tests compare `coxdev` with R's `survival` and `glmnet`
packages, through [rpy2](https://rpy2.github.io/). If rpy2, R or those
packages are missing, those tests are **skipped** and the run still
passes, so check the summary for skipped tests.

### Running the full test suite in a fresh environment

1. Install R (4.x), and in it `survival` and `glmnet` 5.0 or later (the
   tests use `glmnet::coxnet.deviance` and `glmnet:::coxgrad` with
   `cox.ties`):

   ```bash
   Rscript -e 'install.packages(c("survival", "glmnet"), repos = "https://cloud.r-project.org")'
   Rscript -e 'packageVersion("glmnet")'   # 5.0 or later
   ```

2. Clone the repository with its Eigen submodule, and make a new virtual
   environment:

   ```bash
   git clone --recurse-submodules https://github.com/jonathan-taylor/coxdev.git
   cd coxdev
   python -m venv .venv
   source .venv/bin/activate          # on Windows: .venv\Scripts\activate
   python -m pip install --upgrade pip
   ```

   (In an existing clone without the submodule, run
   `git submodule update --init --recursive`.)

3. Build and install `coxdev` (this needs a C++17 compiler), with the test
   dependencies and rpy2:

   ```bash
   pip install ".[test]" rpy2
   ```

4. Check that rpy2 finds R and the two packages:

   ```bash
   python -c "from rpy2.robjects.packages import importr; importr('survival'); importr('glmnet'); print('R is ready')"
   ```

   - If rpy2 first prints `Error importing in API mode ... Trying to import
     in ABI mode`, its compiled interface was built for another version of
     R. ABI mode works; `export RPY2_CFFI_MODE=ABI` skips the attempt and
     the message.
   - If rpy2 cannot find R, set `R_HOME`: `export R_HOME=$(R RHOME)`.

5. From the top of the repository, run the tests, reporting any skipped:

   ```bash
   python -m pytest tests -rs
   ```

   It takes a couple of minutes, and should end with every test passed and
   none skipped.

For a quick check, `--test-size N` runs a random sample of `N` of the
(many) parametrized cases in each of the large comparison files:

```bash
python -m pytest tests -rs --test-size 50
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{coxdev2024,
  title={coxdev: High-performance Cox proportional hazards deviance computation},
  author={Taylor, Jonathan and Hastie, Trevor and Narasimhan, Balasubramanian},
  year={2024},
  url={https://github.com/jonathan-taylor/coxdev}
}
```

## Acknowledgments

- Built with [Eigen](http://eigen.tuxfamily.org/) for efficient linear algebra
- Uses [pybind11](https://pybind11.readthedocs.io/) for Python bindings
- Inspired by the R `glmnet` package for survival analysis
