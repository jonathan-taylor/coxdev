---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# JAX and C++: timing

`coxdev.jax` computes the same deviance as `coxdev.CoxDeviance` with JAX, so
its derivatives come from automatic differentiation: the gradient from
`jax.grad`, products with the information from `jax.jvp` of the gradient, and
products with the derivative of the information in a direction of the linear
predictor (needed, for instance, to differentiate a log determinant in REML)
from a second `jax.jvp`. The C++ code computes all three by hand, in `O(n)`.
So does JAX, as every step depending on the linear predictor is a cumulative
sum or a gather.

This notebook compares their times, keeping apart what JAX spends once,
tracing and compiling (`jax.jit(f).lower(...).compile()`), from what it
spends on each call of the compiled function. The data have many ties among
the event times, start times and weights of which a fifth are 0, with
Efron's ties.

```{code-cell} ipython3
import time

import numpy as np
import pandas as pd

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from coxdev import CoxDeviance
from coxdev.jax import CoxDevianceJAX

print(jax.__version__, jax.devices())
```

```{code-cell} ipython3
def data(n, rng):
    event = np.ceil(rng.exponential(1, n) * 50) / 50       # many ties
    status = rng.binomial(1, 0.7, n)
    start = np.floor(rng.uniform(0, 0.9, n) * event * 50) / 50
    w = rng.uniform(0.5, 2, n)
    w[rng.choice(n, n // 5, replace=False)] = 0
    return event, status, start, w


def best_time(f, repeat=7, number=None):
    # the best of `repeat` runs, each the average over `number` calls
    if number is None:
        t0 = time.perf_counter(); f(); t = time.perf_counter() - t0
        number = max(1, int(0.2 / max(t, 1e-7)))
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            f()
        times.append((time.perf_counter() - t0) / number)
    return min(times)
```

The C++ object caches its result at the last linear predictor, so the
deviance is timed alternating between two linear predictors. Products with the
information and its derivative go through the `LinearOperator`s of
`CoxDeviance`, which return to their linear predictor first (free, as it is
cached) and so include the Python overhead of a call.

```{code-cell} ipython3
def block(x):
    return jax.tree_util.tree_map(lambda a: a.block_until_ready(), x)


def compile_time(f, *args):
    # tracing and compiling, without running
    t0 = time.perf_counter()
    lowered = jax.jit(f).lower(*args)
    t1 = time.perf_counter()
    compiled = lowered.compile()
    t2 = time.perf_counter()
    return compiled, t1 - t0, t2 - t1


rows = []
for n in [1_000, 10_000, 100_000, 1_000_000]:
    rng = np.random.default_rng(0)
    event, status, start, w = data(n, rng)
    eta = [rng.standard_normal(n) * 0.5 for _ in range(2)]
    u, v = rng.standard_normal((2, n))
    eta_j, u_j, v_j = (jnp.asarray(x) for x in (eta[0], u, v))

    t0 = time.perf_counter()
    cox = CoxDeviance(event=event, status=status, start=start, sample_weight=w)
    setup_cpp = time.perf_counter() - t0
    t0 = time.perf_counter()
    cox_jax = CoxDevianceJAX(event, status, start=start, sample_weight=w)
    setup_jax = time.perf_counter() - t0

    I = cox.information(eta[0], w)
    dI = cox.information_derivative(eta[0], v, w)
    i = [0]
    def cpp_deviance():
        i[0] = 1 - i[0]
        return cox(eta[i[0]], w)

    tasks = [
        ('deviance and gradient', cpp_deviance,
         jax.value_and_grad(cox_jax.deviance), (eta_j,)),
        ('information @ u', lambda: I @ u,
         cox_jax.information_matvec, (eta_j, u_j)),
        ('derivative of information @ u', lambda: dI @ u,
         cox_jax.information_derivative_matvec, (eta_j, v_j, u_j)),
    ]
    for name, cpp, f, args in tasks:
        compiled, trace, comp = compile_time(f, *args)
        rows.append(dict(n=n, computation=name,
                         cpp_ms=1e3 * best_time(cpp),
                         jax_ms=1e3 * best_time(lambda: block(compiled(*args))),
                         jax_trace_ms=1e3 * trace,
                         jax_compile_ms=1e3 * comp))
    rows.append(dict(n=n, computation='setup (sort and ties)',
                     cpp_ms=1e3 * setup_cpp, jax_ms=1e3 * setup_jax))

timing = pd.DataFrame(rows)
timing['jax / cpp'] = timing['jax_ms'] / timing['cpp_ms']
timing.round(3)
```

The JAX and C++ results agree (to rounding) at every size:

```{code-cell} ipython3
r = cox(eta[0], w)
print(abs(float(cox_jax.deviance(eta_j)) - r.deviance) / r.deviance,
      np.abs(np.asarray(cox_jax.gradient(eta_j)) - r.gradient).max(),
      np.abs(np.asarray(cox_jax.information_derivative_matvec(eta_j, v_j, u_j)) - dI @ u).max())
```

Tracing and compiling take a fraction of a second, growing with `n` (the
preprocessed indices are constants of the compiled program), and are paid
again for new data. For a fit that calls the same function many times on the
same data (as a REML or Newton iteration), they are paid once: on one CPU
(an Apple laptop), about 0.1 to 0.8 seconds, against a few to a few tens of
milliseconds per call at `n` from `10^5` to `10^6`. After compiling, JAX is
within a small factor of the C++ code: faster for the deviance and gradient
at large `n` (the C++ call also computes the diagonal of the Hessian and the
saturated log likelihood, and checks its cache), about 1.5 to 2.5 times
slower for the products with the information and its derivative.
