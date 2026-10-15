---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Checking proportional hazards

The Cox model assumes each covariate's effect is constant in time. The
Schoenfeld residuals check this (Schoenfeld, 1982; Grambsch and Therneau,
1994). For a failure at time $t_k$, the residual $r_k$ is its covariates
less their mean over the risk set at $t_k$. If the coefficient is really
$\beta(t)$, the *scaled* residual
$$
r^*_k = \hat\beta + V_k^{-1} r_k
$$
has mean about $\beta(t_k)$, where $V_k$ is the covariance of the
covariates over the risk set at $t_k$. Plotting $r^*_k$ against time shows
how the effect changes, and regressing it on a function of time tests
whether it does.

`schoenfeld_residuals` gives the $r_k$, and `risk_set_variance` the
variances in $V_k$, both one covariate at a time.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from coxdev import CoxDeviance
```

## Data with an effect that fades

The effect of `x1` is 1 until time 1 and 0 after; those of `x2` and `x3`
are constant. We simulate from the piecewise-constant hazard by inverting
its cumulative hazard.

```{code-cell} ipython3
rng = np.random.default_rng(1)
n = 600
X = rng.standard_normal((n, 3))
names = ['x1', 'x2', 'x3']
h_early = np.exp(X @ [1., -0.5, 0.5])     # the hazard before time 1
h_late = np.exp(X @ [0., -0.5, 0.5])      # and after
E = rng.exponential(size=n)               # the cumulative hazard at failure
T = np.where(E < h_early, E / h_early, 1 + (E - h_early) / h_late)
C = rng.uniform(0, 4, n)
event = np.minimum(T, C)
status = (T <= C).astype(int)

cox = CoxDeviance(event=event, status=status)
```

We fit the (misspecified) proportional hazards model by Newton's method.

```{code-cell} ipython3
def fit_cox(cox, X):
    b = np.zeros(X.shape[1])
    for _ in range(30):
        eta = X @ b
        score = -X.T @ cox(eta).gradient / 2
        step = np.linalg.solve(X.T @ (cox.information(eta) @ X), score)
        b = b + step
        if np.abs(step).max() < 1e-10:
            break
    return b

b = fit_cox(cox, X)
eta = X @ b
info = X.T @ (cox.information(eta) @ X)
dict(zip(names, b.round(3)))
```

The estimate for `x1` averages its early and late effects.

## Scaled Schoenfeld residuals

`schoenfeld_residuals` has a row for every observation, zero for those
that do not fail. `risk_set_variance` gives the diagonal of each $V_k$; the
covariances follow by polarization,
$\operatorname{Cov}(y, z) = (V(y + z) - V(y - z)) / 4$.

```{code-cell} ipython3
fail = status == 1
t_fail = event[fail]
d = fail.sum()
r = cox.schoenfeld_residuals(eta, X)[fail]

p = X.shape[1]
V = np.empty((d, p, p))
for j in range(p):
    for k in range(j, p):
        if j == k:
            v = cox.risk_set_variance(eta, X[:, j])
        else:
            v = (cox.risk_set_variance(eta, X[:, j] + X[:, k]) -
                 cox.risk_set_variance(eta, X[:, j] - X[:, k])) / 4
        V[:, j, k] = V[:, k, j] = v[fail]

scaled = b + np.linalg.solve(V, r[..., None])[..., 0]
```

The $V_k$ add up to the information, the sum over the failures of the
risk-set covariances:

```{code-cell} ipython3
np.abs(V.sum(0) - info).max()
```

Against time, with the means of the scaled residuals in bins of 40
failures, and the true coefficients dashed:

```{code-cell} ipython3
order = np.argsort(t_fail)
bins = np.array_split(order, d // 40)
truth = {'x1': lambda t: np.where(t < 1, 1., 0.), 'x2': lambda t: -0.5 + 0 * t,
         'x3': lambda t: 0.5 + 0 * t}
grid = np.linspace(0, t_fail.max(), 400)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for j, ax in enumerate(axes):
    ax.scatter(t_fail, scaled[:, j], s=4, alpha=0.3, color='gray')
    ax.plot([t_fail[i].mean() for i in bins], [scaled[i, j].mean() for i in bins],
            'o-', color='C0', label='binned means')
    ax.plot(grid, truth[names[j]](grid), '--', color='C3', label='true coefficient')
    ax.axhline(b[j], color='black', lw=0.8, label='estimate')
    ax.set_title(names[j])
    ax.set_xlabel('time')
    ax.set_ylim(-4, 4)
axes[0].set_ylabel(r'scaled Schoenfeld residual, $\beta(t)$')
axes[0].legend(fontsize=8);
```

The binned means of `x1` follow its coefficient down from 1 to 0 at time
1, while those of `x2` and `x3` stay at their estimates.

## The test of Grambsch and Therneau

Grambsch and Therneau (1994) test for $\beta_j(t) = \beta_j + \theta_j
g(t)$ with a score test of $\theta = 0$. Replacing each $V_k$ by their
average $I / d$ gives a simple form, with $u = \sum_k (g_k - \bar g) r_k$:
$$
T_j = \frac{d \, \big[ (I^{-1} u)_j \big]^2}{(I^{-1})_{jj} \sum_k (g_k - \bar g)^2},
\qquad
T = \frac{d \, u^T I^{-1} u}{\sum_k (g_k - \bar g)^2},
$$
each covariate's statistic on 1 degree of freedom and the global one on
$p$. Here $g$ is the rank of the failure time.

```{code-cell} ipython3
g = np.argsort(np.argsort(t_fail)) + 1.
gc = g - g.mean()
u = gc @ r
info_inv = np.linalg.inv(info)
T_j = d * (info_inv @ u)**2 / (np.diag(info_inv) * (gc @ gc))
T = d * u @ info_inv @ u / (gc @ gc)
for name, stat in zip(names + ['global'], np.append(T_j, T)):
    df = p if name == 'global' else 1
    print(f'{name:7s} chi2 = {stat:6.2f}, p = {chi2.sf(stat, df):.2g}')
```

The test finds the fading effect of `x1`, and nothing for `x2` and `x3`.

## References

- Grambsch, P. M. and Therneau, T. M. (1994). Proportional hazards tests
  and diagnostics based on weighted residuals. *Biometrika*, 81(3),
  515–526.
- Schoenfeld, D. (1982). Partial residuals for the proportional hazards
  regression model. *Biometrika*, 69(1), 239–241.
- Therneau, T. M. and Grambsch, P. M. (2000). *Modeling Survival Data:
  Extending the Cox Model*, Chapter 6. Springer, New York.
