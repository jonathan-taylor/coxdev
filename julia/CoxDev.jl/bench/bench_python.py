"""Per-call times of Python coxdev: deviance (fresh eta each call) and information matvec."""
import sys, time, json
import numpy as np
from coxdev import CoxDeviance

d = sys.argv[1]


def per_call(f, reps):
    f(); f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


res = {}
for n in [10**4, 10**5, 10**6]:
    a = np.fromfile(f'{d}/data_{n}.bin').reshape(6, n)
    event, status, eta1, eta2, v, w = a
    cox = CoxDeviance(event=event, status=status.astype(int), sample_weight=w, tie_breaking='efron')
    reps = 200 if n <= 10**5 else 30
    k = [0]
    etas = [eta1, eta2]

    def dev():
        # alternate two etas so the cache never hits: a full evaluation each call
        k[0] ^= 1
        cox(etas[k[0]], w)

    t_dev = per_call(dev, reps)
    info = cox.information(eta1, w)
    t_mv = per_call(lambda: info @ v, reps)
    # deviance then information product at the same eta (Python's cache: one evaluation)
    def dev_then_mv():
        k[0] ^= 1
        cox(etas[k[0]], w)
        cox.information(etas[k[0]], w) @ v
    t_both = per_call(dev_then_mv, reps)
    res[n] = dict(deviance=t_dev, matvec=t_mv, deviance_then_matvec=t_both)
print(json.dumps(res))
