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

# Martingale and Cox–Snell residuals

`expected_events` gives each observation's expected number of events over
its time at risk: $E_i = e^{\eta_i}$ times the increase of the baseline
cumulative hazard over $(s_i, t_i]$. Two classical residuals come from it:

- the **martingale residual** $M_i = \delta_i - E_i$ (Barlow and Prentice,
  1988; Therneau, Grambsch and Fleming, 1990), which shows how a covariate
  should enter the model;
- the **Cox–Snell residual** $E_i$ itself, which, if the model is right,
  behaves like a censored sample from the unit exponential: their
  cumulative hazard is the identity.

`survival_at_exit` is $e^{-E_i}$, the fitted probability that observation
$i$ survives to $t_i$ (given survival to $s_i$).

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from coxdev import CoxDeviance
```

The log hazard is linear in `x1` but has a hinge in `x2`: no effect below
0, a slope of 1.2 above.

```{code-cell} ipython3
rng = np.random.default_rng(2)
n = 800
x1 = rng.standard_normal(n)
x2 = rng.uniform(-2, 2, n)
eta_true = 0.7 * x1 + 1.2 * np.maximum(x2, 0)
T = rng.exponential(np.exp(-eta_true))
C = rng.exponential(2, n)
event = np.minimum(T, C)
status = (T <= C).astype(int)

cox = CoxDeviance(event=event, status=status)

def fit_cox(cox, X):
    b = np.zeros(X.shape[1])
    for _ in range(30):
        eta = X @ b
        step = np.linalg.solve(X.T @ (cox.information(eta) @ X),
                               -X.T @ cox(eta).gradient / 2)
        b = b + step
        if np.abs(step).max() < 1e-10:
            break
    return b
```

## The functional form of a covariate

Fit the model without `x2` and plot its martingale residuals against `x2`
(Therneau, Grambsch and Fleming, 1990). Their local mean is roughly
proportional to the effect of `x2` on the log hazard, up to a constant:

```{code-cell} ipython3
X1 = x1[:, None]
b1 = fit_cox(cox, X1)
martingale = status - cox.expected_events(X1 @ b1)

edges = np.quantile(x2, np.linspace(0, 1, 11))
which = np.clip(np.digitize(x2, edges) - 1, 0, 9)
centers = [x2[which == k].mean() for k in range(10)]
means = [martingale[which == k].mean() for k in range(10)]

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.scatter(x2, martingale, s=4, alpha=0.3, color='gray')
ax.plot(centers, means, 'o-', label='means in deciles of x2')
ax.set_xlabel('x2')
ax.set_ylabel('martingale residual, without x2')
ax.legend();
```

The residuals are roughly flat below 0 and rise above it: the hinge, which a
linear term in `x2` would miss. Martingale residuals sum to zero at any
linear predictor, since the expected events add up to the number of
failures:

```{code-cell} ipython3
martingale.sum()
```

## Goodness of fit, from the Cox–Snell residuals

If the model is right, the pairs $(E_i, \delta_i)$ are like censored unit
exponentials, so the Nelson–Aalen estimate of their cumulative hazard
follows the line of slope 1. The Nelson–Aalen estimate is Breslow's
baseline hazard at $\eta = 0$, so `coxdev` gives it too: a `CoxDeviance`
with the residuals as event times.

```{code-cell} ipython3
models = {'linear in x2': np.column_stack([x1, x2]),
          'hinge in x2': np.column_stack([x1, np.maximum(x2, 0)])}

fig, ax = plt.subplots(figsize=(5, 4))
for name, X in models.items():
    b = fit_cox(cox, X)
    cox_snell = cox.expected_events(X @ b)
    na = CoxDeviance(event=cox_snell, status=status,
                     tie_breaking='breslow').baseline_hazard(np.zeros(n))
    ax.step(na.time, na.cumulative_hazard, where='post', label=name)
ax.axline((0, 0), slope=1, color='black', ls='--', lw=0.8)
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
ax.set_xlabel('Cox–Snell residual')
ax.set_ylabel('Nelson–Aalen cumulative hazard')
ax.legend();
```

Both models follow the line closely. The Cox–Snell plot is a weak check
of how a covariate enters the model: the martingale residuals above found
the misfit that it misses.

## Survival at exit

`survival_at_exit` is the fitted probability of surviving to each
observation's exit time, $e^{-E_i}$:

```{code-cell} ipython3
X = models['hinge in x2']
b = fit_cox(cox, X)
S = cox.survival_at_exit(X @ b)
np.allclose(S, np.exp(-cox.expected_events(X @ b)))
```

The failures with the highest fitted survival are the most surprising
ones:

```{code-cell} ipython3
failed = np.flatnonzero(status == 1)
surprising = failed[np.argsort(S[failed])[::-1][:5]]
pd.DataFrame({'time': event[surprising], 'x1': x1[surprising], 'x2': x2[surprising],
              'survival at exit': S[surprising]}).round(3)
```

## References

- Barlow, W. E. and Prentice, R. L. (1988). Residuals for relative risk
  regression. *Biometrika*, 75(1), 65–74.
- Therneau, T. M. and Grambsch, P. M. (2000). *Modeling Survival Data:
  Extending the Cox Model*, Chapters 4 and 5. Springer, New York.
- Therneau, T. M., Grambsch, P. M. and Fleming, T. R. (1990).
  Martingale-based residuals for survival models. *Biometrika*, 77(1),
  147–160.
