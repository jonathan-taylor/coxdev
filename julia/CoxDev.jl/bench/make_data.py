"""Benchmark data shared by Julia and Python: raw float64 files (event, status, eta1, eta2, v, w)."""
import sys, numpy as np
out = sys.argv[1]
for n in [10**4, 10**5, 10**6]:
    rng = np.random.default_rng(n)
    event = np.round(rng.exponential(1, n), 2) + 0.01       # ties from rounding
    status = rng.binomial(1, 0.7, n).astype(float)
    eta1 = 0.3 * rng.standard_normal(n); eta2 = eta1 + 1e-3
    v = rng.standard_normal(n); w = rng.uniform(0.5, 2, n)
    np.column_stack([event, status, eta1, eta2, v, w]).T.astype(np.float64).tofile(f'{out}/data_{n}.bin')
print('ok')
