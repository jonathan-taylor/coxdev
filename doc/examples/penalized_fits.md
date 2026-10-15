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

# Penalized fits with any solver

`CoxDeviance` gives the objective, its gradient and products with its
Hessian, all in the linear predictor $\eta$. With a design $X$ these become
the objective of the coefficients $\beta$, its gradient $X^T \nabla_\eta$
and the Hessian products $X^T I X v$, which is all that most solvers need.
Here we fit a ridge-penalized Cox model with a trust-region method from
SciPy, and a lasso by proximal gradient descent, and check the lasso
against glmstar's `CoxNet`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from coxdev import CoxDeviance

rng = np.random.default_rng(0)
n, p = 300, 20
X = rng.standard_normal((n, p))
beta = np.zeros(p)
beta[:4] = [1, -0.8, 0.6, 0.4]
T = rng.exponential(np.exp(-X @ beta))
C = rng.exponential(2, n)
event = np.minimum(T, C)
status = (T <= C).astype(int)

cox = CoxDeviance(event=event, status=status, tie_breaking='breslow')
```

We use glmnet's scaling: the deviance over $2n$, which is the negative log
partial likelihood over $n$, up to a constant. Its Hessian is the
information over $n$, since the information is half the Hessian of the
deviance.

```{code-cell} ipython3
def objective(b):
    return cox(X @ b).deviance / (2 * n)

def gradient(b):
    return X.T @ cox(X @ b).gradient / (2 * n)

def hessian_product(b, v):
    return X.T @ (cox.information(X @ b) @ (X @ v)) / n
```

`cox` keeps the result at the last linear predictor, so `gradient(b)` right
after `objective(b)` costs nothing more.

## Ridge, by a trust-region method

The ridge objective adds $\frac{\lambda}{2} \|\beta\|_2^2$. SciPy's
`trust-krylov` needs only products with the Hessian, so the $n \times n$
information is never formed:

```{code-cell} ipython3
lam_ridge = 0.1
ridge = minimize(lambda b: objective(b) + lam_ridge / 2 * b @ b,
                 np.zeros(p),
                 jac=lambda b: gradient(b) + lam_ridge * b,
                 hessp=lambda b, v: hessian_product(b, v) + lam_ridge * v,
                 method='trust-krylov')
print(ridge.message, f'{ridge.nit} iterations')
pd.Series(ridge.x[:6], index=[f'x{j}' for j in range(6)]).round(3)
```

At the solution the gradient of the penalized objective is zero:

```{code-cell} ipython3
np.abs(gradient(ridge.x) + lam_ridge * ridge.x).max()
```

## Lasso, by proximal gradient descent

The lasso objective adds $\lambda \|\beta\|_1$. Proximal gradient descent
(FISTA, with a backtracking line search) takes a gradient step on the
smooth part and soft-thresholds:

```{code-cell} ipython3
def soft_threshold(v, t):
    return np.sign(v) * np.maximum(np.abs(v) - t, 0)

def fista(lam, step=1.0, tol=1e-9, max_iter=5000):
    b = z = np.zeros(p)
    t = 1.0
    for it in range(max_iter):
        f_z, g_z = objective(z), gradient(z)
        while True:
            b_new = soft_threshold(z - step * g_z, step * lam)
            d = b_new - z
            # the sufficient-decrease test, with room for rounding
            if objective(b_new) <= f_z + g_z @ d + d @ d / (2 * step) + 1e-14:
                break
            step /= 2
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        z = b_new + (t - 1) / t_new * (b_new - b)
        change = np.abs(b_new - b).max()
        b, t = b_new, t_new
        if change < tol:
            break
    return b, it

lam = 0.05
b_lasso, iterations = fista(lam)
print(f'{iterations} iterations, {np.sum(b_lasso != 0)} nonzero coefficients')
```

The KKT conditions: on the active set the gradient is $-\lambda$ times the
sign of the coefficient, and elsewhere it is at most $\lambda$ in absolute
value.

```{code-cell} ipython3
g = gradient(b_lasso)
active = b_lasso != 0
print('active:  ', np.abs(g[active] + lam * np.sign(b_lasso[active])).max())
print('inactive:', np.abs(g[~active]).max(), '<=', lam)
```

glmstar's `CoxNet` minimizes the same objective, by coordinate descent,
with the same C++ computing the partial likelihood:

```{code-cell} ipython3
from glmnet import CoxNet
from glmnet.cox import CoxFamily

family = CoxFamily(event_id='event', status_id='status', tie_breaking='breslow')
y = pd.DataFrame({'event': event, 'status': status})
fit = CoxNet(family=family, lambda_values=[lam], standardize=False).fit(X, y)
b_glmnet = np.asarray(fit.coefs_[0]).ravel()
pd.DataFrame({'FISTA': b_lasso, 'CoxNet': b_glmnet})[active | (b_glmnet != 0)].round(4)
```

The two agree to the tolerance of `CoxNet`'s coordinate descent, and the
objectives to many digits:

```{code-cell} ipython3
print(np.abs(b_lasso - b_glmnet).max())
for b in [b_lasso, b_glmnet]:
    print(objective(b) + lam * np.abs(b).sum())
```
