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

# The score and Hessian without ties

*Trevor Hastie, Balasubramanian Narasimhan, Jonathan Taylor.* These notes,
in three parts, develop the score and Hessian of Cox's partial likelihood
with weights and ties, for both Breslow's and Efron's handling of ties.
This part treats data without ties; [the next](notes_ties.md) adds ties and
zero weights, and [the last](notes_hessian.md) gives the full Hessian and
the lemmas behind the algorithm.

There are two terms in Cox's partial log-likelihood

$$
\sum_{i=1}^n d_i w_i \left[ \eta_i - \log \left(\sum_{j:j \geq i} w_j e^{\eta_j} \right) \right].
$$

We focus on the second, as the first is simple to deal with, even with
ties.

## The easy case: no ties

We first compute things when there are no ties in the data. Individuals
may have tied start times. However, for any event time $t_i$ where a
failure occurs ($d_i=1$), we assume there are no other individuals with the
same event time, nor any start times equal to $t_i$. This makes ordering
the $t_i$'s unambiguous, and the distinction between a start time equal to
$t_i$ or not irrelevant.

:::{important} The sort order
We assume the data are sorted by event time: $j > i \implies t_j > t_i$. As
there are no ties in this part, we can recover "disk order" by reversing
the sort. More generally, with ties we assume that $j > i \implies t_j \geq
t_i$, and that tied event and start times are sorted in an arbitrary but
fixed order, so that we can map unambiguously between this fixed order and
disk order.
:::

### Right censored

Ignoring the linear term, and the sign, we need to differentiate (by
$\eta_k$)

$$
\sum_{i = 1}^n d_i w_i \log \left(\sum_{j \geq i} w_j e^{\eta_j}\right).
$$

Here $d_i=\text{status}_i$, so it is always binary, even in the presence of
ties. Ties are dealt with in [the next part](notes_ties.md).

$$
\begin{aligned}
\frac{\partial}{\partial \eta_k} \sum_{i = 1}^n d_i w_i \log \left(\sum_{j \geq i} w_j e^{\eta_j}\right)
&= \sum_{i=1}^n d_i w_i \frac{ \sum_{j \geq i} w_j \delta_{jk} e^{\eta_k}}
{\sum_{l \geq i} w_l e^{\eta_l}} \\
&= \sum_{i=1}^n d_i w_i \frac{ 1_{\{k \geq i\}} w_k e^{\eta_k}}
{\sum_{l \geq i} w_l e^{\eta_l}} \\
&= w_k e^{\eta_k} \sum_{i=1}^k \frac{d_i w_i}{\sum_{l \geq i} w_l e^{\eta_l}}.
\end{aligned}
$$

The second derivative:

$$
\begin{aligned}
\frac{\partial^2}{\partial \eta_k^2} \sum_{i = 1}^n d_i w_i \log \left(\sum_{j \geq i} w_j e^{\eta_j}\right)
&= \frac{\partial}{\partial \eta_k} \left[w_k e^{\eta_k} \sum_{i=1}^k \frac{d_i w_i}{\sum_{l \geq i} w_l e^{\eta_l}} \right] \\
&= w_k e^{\eta_k} \left[\sum_{i=1}^k \frac{d_i w_i}{\sum_{l \geq i} w_l e^{\eta_l}}
- \sum_{i=1}^k d_i w_i \frac{ \sum_{l \geq i} w_l \delta_{lk} e^{\eta_l}}{(\sum_{l \geq i} w_l e^{\eta_l})^2} \right] \\
&= w_k e^{\eta_k} \left[\sum_{i=1}^k \frac{d_i w_i}{\sum_{l \geq i} w_l e^{\eta_l}}
- \sum_{i=1}^k \frac{d_i w_i \, w_k e^{\eta_k}}{(\sum_{l \geq i} w_l e^{\eta_l})^2} \right].
\end{aligned}
$$

### Start / stop

Let the start times be $(s_m)_{1 \leq m \leq n}$. We make the basic
observation

$$
\begin{aligned}
\frac{\partial}{\partial \eta_k} \left(\sum_{j:j \geq i} w_j e^{\eta_j} - \sum_{m:s_m \geq t_i} w_m e^{\eta_m} \right)
&= w_k e^{\eta_k} \left(\sum_{j:j \geq i} \delta_{jk} - \sum_{m: s_m \geq t_i} \delta_{mk} \right) \\
&= w_k e^{\eta_k} \left[ 1_{\{k \geq i\}} - 1_{\{s_k \geq t_i\}} \right].
\end{aligned}
$$

We need to differentiate (by $\eta_k$)

$$
\sum_{i = 1}^n d_i w_i \log \left(\sum_{j \geq i} w_j e^{\eta_j} - \sum_{m:s_m \geq t_i} w_m e^{\eta_m} \right).
$$

Proceeding,

$$
\begin{aligned}
\frac{\partial}{\partial \eta_k} \sum_{i = 1}^n d_i w_i \log \left(\sum_{j \geq i} w_j e^{\eta_j}- \sum_{m:s_m \geq t_i} w_m e^{\eta_m} \right)
&= w_k e^{\eta_k} \sum_{i=1}^n d_i w_i \frac{1_{\{k \geq i\}} - 1_{\{s_k \geq t_i\}}}
{\sum_{j \geq i} w_j e^{\eta_j}- \sum_{m:s_m \geq t_i} w_m e^{\eta_m}} \\
&= w_k e^{\eta_k}\biggl[ \sum_{i=1}^k \frac{d_i w_i}
{\sum_{l \geq i} w_l e^{\eta_l}- \sum_{m:s_m \geq t_i} w_m e^{\eta_m}} \\
& \qquad \quad - \sum_{i:t_i \leq s_k} \frac{d_i w_i}
{\sum_{l \geq i} w_l e^{\eta_l}- \sum_{m:s_m \geq t_i} w_m e^{\eta_m}}\biggr].
\end{aligned}
$$

This can be computed from the cumsum of the sequence

$$
\left( \frac{ d_i w_i}
{\sum_{l \geq i} w_l e^{\eta_l}- \sum_{m:s_m \geq t_i} w_m e^{\eta_m} } \right)_{1 \leq i \leq n}.
$$

:::{note}
This depends only on the risk sums at failure times. Hence we need only an
algorithm that correctly computes the risk sums at failure times. We return
to this with ties.
:::

### Difference of cumsums

Let

$$
E_i(\xi) = \sum_{j \geq i: w_j > 0} \xi_j = \sum_{j: t_j \geq t_i, w_j > 0} \xi_j
\qquad \text{and} \qquad
S_i(\xi) = \sum_{j: s_j \geq t_i, w_j > 0} \xi_j.
$$

The quantity $S_i$ can be expressed in terms of the cumsum of the $w_j
e^{\eta_j}$ ordered by start time, and the index of $t_i$ among the start
times $(s_i)_{1 \leq i \leq n}$. The difference

$$
\sum_{j: j \geq i} w_j e^{\eta_j} - \sum_{m:s_m \geq t_i} w_m e^{\eta_m}
$$

is the risk sum

$$
R_i(we^{\eta}) = E_i(we^{\eta}) - S_i(we^{\eta}).
$$

The objective can then be expressed as

$$
\sum_{i=1}^n d_i w_i \log(R_i(we^{\eta})),
$$

with gradient

$$
w_k e^{\eta_k}
\sum_{i=1}^n
\frac{d_i w_i [1_{\{k \geq i\}} - 1_{\{s_k \geq t_i\}}]}{E_i(we^{\eta})-S_i(we^{\eta})} = w_k e^{\eta_k}
\sum_{i=1}^n
\frac{d_i w_i [1_{\{k \geq i\}} - 1_{\{s_k \geq t_i\}}]}{R_i(we^{\eta})}.
$$

This is a difference of cumsums of the sequence

$$
\left(\frac{d_i w_i}{R_i(we^{\eta})}\right)_{1 \leq i \leq n}.
$$

The cumsums are ordered by event (stop) time, so we need the index of the
start time $s_k$ among the event times $(t_i)_{1 \leq i \leq n}$.

### Second derivative

The diagonal of the Hessian is

$$
w_k e^{\eta_k}
\sum_{i=1}^n
\frac{d_i w_i [1_{\{k \geq i\}} - 1_{\{s_k \geq t_i\}}]}{E_i(we^{\eta})-S_i(we^{\eta})}
- w_k^2 e^{2 \eta_k} \sum_{i=1}^n \frac{d_i w_i [1_{\{k \geq i\}} - 1_{\{s_k \geq t_i\}}]}{(E_i(we^{\eta})-S_i(we^{\eta}))^2},
$$

using $1_{\{k \geq i\}}-1_{\{s_k \geq t_i\}} \in \{0,1\}$, so that it equals
its square. This too is a difference of cumsums: of the sequence above and
of the sequence

$$
\left(\frac{d_i w_i}{(E_i(we^{\eta})-S_i(we^{\eta}))^2}\right)_{1 \leq i \leq n},
$$

again cumsummed in event (stop) order.

## Strata

With strata, the partial likelihood is the sum of one per stratum: the risk
set of a failure holds only the observations of its own stratum. The C++
code keeps a model per stratum. A single pass over all the strata is also
possible, with the same risk sums $R_i = E_i - S_i$ and the same cumsums as
above: only the sort changes.

**Each stratum in a window of its own.** Number the strata $h = 1, \dots,
H$ and choose $B$ larger than the range of all the times. Add $hB$ to every
time of stratum $h$, its start times and its event times, and sort as
usual. Stratum $h$'s times then all lie in a window $[hB, hB + B)$ of their
own, and the windows follow one another. Consider a failure $i$ in stratum
$h$, at shifted time $t_i$, and an observation $j$ of another stratum $h'$:

- **An earlier stratum, $h' < h$:** all of $j$'s times are below $hB$, so
  $t_j < t_i$ and $j$ is not in $E_i$. It is not at risk.
- **A later stratum, $h' > h$:** all of $j$'s times are above
  $t_i$. So $t_j \geq t_i$ and $j$ is in $E_i$; but also $s_j \geq t_i$, so
  $j$ is in $S_i$ as well, and the difference $R_i = E_i - S_i$ takes it out
  again. It is not at risk, because it has not yet entered.

So $R_i$ is the risk sum of failure $i$ within its own stratum, and every
other cumsum of these notes (for the gradient, the Hessian and Efron's
ties) adds up in the same way.

**What makes it work:**

- **The start times.** The later strata are taken out by $S_i$, not by
  $E_i$, so the trick needs start / stop data. Without start times, each
  observation has a start of $-\infty$ within its stratum, the start of its
  window: shifted, these are all below the window's failure times, so $S_i$
  is then exactly the sum over the later strata. Right-censored data take
  one more cumsum this way than without strata.
- **No ties across strata.** As $B$ exceeds the range of the times, no time
  of one window equals a time of another. Sets of tied failures (and with
  them Efron's weights, the averaged weights $\bar{w}$ and the saturated
  log-likelihood, all differences of cumsums over a set
  $\{\text{first}(i), \dots, \text{last}(i)\}$) never cross a stratum.
- **No $B$ in practice.** Adding a large $B$ to floating-point times would
  round them, and could merge or split ties. Sorting by (stratum, time)
  instead orders the observations exactly as the shifted times would.
  Equivalently, sort each stratum on its own, concatenate them, and shift
  each stratum's positions (`first`, `last` and the maps between event and
  start orders) by the number of observations before it.

This is how `coxdev.jax` handles strata. Its compiled program then
depends only on $n$, not on the number of strata. With a program per
stratum, compiling for 1000 strata takes minutes, against a fraction of a
second in one pass.

## Probabilistic interpretation

For computing derivatives it is perhaps more helpful to think of
differentiating

$$
\sum_{i=1}^n d_i w_i \log(R_i(we^{\eta})/R_i(w))
\overset{\text{def}}{=} \sum_{i=1}^n d_i w_i \Lambda_i(\eta).
$$

The map $\eta \mapsto \log(R_i(we^{\eta})/R_i(w))$ is the cumulant
generating function of an exponential family built on a base measure on
the risk set. The law of this base random variable $V_i$ changes with time,
as it depends on the risk set at $t_i$: it has density proportional to
$w_j$ on $\{j:t_j \geq t_i, s_j < t_i\}$. Call its law $P_i$, and the
corresponding law under $\eta$ $P_{i,\eta}$.

This interpretation gives the $k$-th coordinate of the gradient as

$$
\sum_{i=1}^n d_i w_i E_{i,\eta}[1_{\{k\}}(V_i)] =
\sum_{i=1}^n d_i w_i \frac{w_k e^{\eta_k} 1_{\{k \in \text{supp}(P_i)\}}}{R_i(we^{\eta})} = \sum_{i=1}^n d_i w_i \pi_{i,\eta}[k].
$$

The second derivative has $(r,c)$ entry

$$
\sum_{i=1}^n d_i w_i\left( \pi_{i,\eta}[r] \delta_{rc} -
\pi_{i,\eta}[r] \pi_{i,\eta}[c] \right).
$$

:::{note}
This also describes the saturated log-likelihood, by considering
multinomials over the successive risk sets.
:::

## A check against coxdev

With distinct event times and start times, we compute the gradient and the
diagonal of the Hessian from the cumsums above, and compare them with
`CoxDeviance`. Its gradient and Hessian are of the deviance, $2(\ell_{\rm
sat} - \ell)$, so they are $-2$ times those of the log-likelihood $\ell$.

```{code-cell} ipython3
import numpy as np
from coxdev import CoxDeviance

rng = np.random.default_rng(0)
n = 50
t = rng.exponential(size=n) + 1             # distinct event times
s = rng.uniform(0, 1, n) * t                 # distinct start times, s < t
d = rng.binomial(1, 0.7, n)
w = rng.uniform(0.5, 2, n)
eta = rng.standard_normal(n) * 0.5

# sort by event time: i < j implies t_i < t_j
o = np.argsort(t)
t, s, d, w, eta = t[o], s[o], d[o], w[o], eta[o]
e = w * np.exp(eta)

# the risk sums R_i = E_i - S_i
E = np.cumsum(e[::-1])[::-1]
S = np.array([e[s >= ti].sum() for ti in t])
R = E - S

# cumsums of d w / R and of d w / R^2, evaluated at k and at the last i with t_i <= s_k
C1 = np.concatenate([[0], np.cumsum(d * w / R)])
C2 = np.concatenate([[0], np.cumsum(d * w / R**2)])
start_index = np.searchsorted(t, s, side='right')   # the number of i with t_i <= s_k
k = np.arange(n)
A1 = C1[k + 1] - C1[start_index]
A2 = C2[k + 1] - C2[start_index]

grad_loglik = d * w - e * A1                 # the linear term, less the log term
diag_hess_loglik = -(e * A1 - e**2 * A2)

cox = CoxDeviance(event=t, start=s, status=d, sample_weight=w)
result = cox(eta, w)
print('gradient:         ', np.abs(result.gradient - (-2 * grad_loglik)).max())
print('diagonal Hessian: ', np.abs(result.diag_hessian - (-2 * diag_hess_loglik)).max())
```
