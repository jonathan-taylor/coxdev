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

# Ties and zero weights

This part extends [the case without ties](notes_no_ties.md) to tied
failure times and zero weights, for Breslow's and Efron's handling of
ties.

:::{important} The sort order, and zero weights
As before, we pick an arbitrary sort by event time: $j \geq i \implies t_j
\geq t_i$, and recover disk order by inverting this sort. Sequences indexed
by $i$ below are indexed by this fixed but arbitrary sort of the event
times. Any sum over cases $j$ is restricted to the cases with $w_j > 0$.
:::

With ties, which of a set of tied failures occurred "first" must be
disambiguated in some way. Given a hypothetical ordering of the failures,
the risk sum of each failure at a time could be computed. This is usually
not done exactly: the Breslow and Efron methods approximate the risk sums.
Both proceed essentially as before, computing a risk sum for each
individual and then the log of the risk sum.

Any reasonable approximation that computes a risk sum for each individual
should be exchangeable with respect to reordering the failures at an
observed failure time.

The linear term has no issue; it is the sum of log risk sums that needs
care. For this, we rewrite the log-likelihood as

$$
\sum_{i=1, w_i > 0}^n d_i \left[w_i \eta_i - \bar{w}_i \log \left( \bar{R}_i(we^{\eta}) \right) \right].
$$

Again $d_i \in \{0,1\}$, and we need only compute the risk sums $\bar{R}_i$
correctly for the individuals who failed, $d_i=1$.

There are several orderings of the individuals by event time. The Efron and
Breslow methods, expressed as sums over individuals, can be viewed as
picking an arbitrary such order and proposing $\bar{w}$ and $\bar{R}$ that
are invariant to this choice.

## Clusters of tied failures

We group individuals into clusters so that for any individual $i$ who
failed ($d_i=1$) at time $t_i$, their cluster is all the others who failed
at that same time. Given such a clustering $C$, set

```{math}
:label: eq-avg-weight
\bar{w}_i = \begin{cases} \dfrac{\sum_{j:C_j=C_i, w_j>0} w_j}{K_i} & w_i > 0, \\
0 & \text{otherwise,}
\end{cases}
```

with

```{math}
:label: eq-effective-cluster
K_i = \# \{j:C_j=C_i, w_j>0\}.
```

We have again summed only over $\{j:w_j>0\}$ to compute the average weight
of the cluster.

:::{note}
By the form of the likelihood, it only matters that $\bar{w}_i$ is computed
correctly for the individuals who fail ($d_i=1$).
:::

We choose a clustering that puts each individual in a cluster of their own
*unless* $d_i=1$. If an individual failed, we find all the others who
failed at that time and put them in the same cluster. So for each $i$ there
is a $\texttt{first}(i)$, the first index in our ordering in the same
cluster as $i$, and a $\texttt{last}(i)$. The size of each cluster is
$\texttt{last}(i) + 1 - \texttt{first}(i)$: this is 1 unless $t_i$ is a
failure ($d_i=1$) with more than one failure at $t_i$.

From $\texttt{first}$ and $\texttt{last}$ we compute a "centering" weight
$\sigma$ for the Efron or Breslow correction. It is easier to describe in
words than in formulae: for a cluster of size $K$ (a set of $K$ failures at
some $t$), the entries of $\sigma$ are $\{0,1/K,2/K,\dots,(K-1)/K\}$, with
$K$ as in [](#eq-effective-cluster). Entries of the cluster with weight 0
have $\sigma_i=0$. For example, a cluster of size 6 with weights
$w=[1/2,0,0,3/4,0,2]$ has $\sigma=[0,0,0,1/3,0,2/3]$: the $\sigma$ we would
get by dropping the entries with weight 0 from the `(start, stop, event)`
arrays to begin with. This is done by a lexical sort that places, among
equal event times, the failures with $w_i>0$ at the start of the segment,
in one cluster.

We can now define our risk sums

```{math}
:label: eq-risk-sum
\begin{aligned}
\Delta_i(\xi) &= \sum_{j:C_j=C_i, w_j>0} \xi_j, \\
\bar{R}_i(\xi) &= \sum_{j: t_j \geq t_{i}, w_j>0} \xi_j -
\sum_{m: s_m \geq t_i, w_m>0} \xi_m
- \delta \cdot \sigma_i \cdot \Delta_i(\xi).
\end{aligned}
```

$\bar{R}$ is linear: $\bar{R}_i(\xi)=\bar{r}_i'\xi$ for some vector
$\bar{r}_i$.

Following our choice of clustering, $R_i(we^{\eta})=\bar{R}_i(we^{\eta})$
unless $d_i=1$. The cumulative sum of $\bar{R}_i(we^{\eta})$ over $i$ does
not depend on our initial ordering of the $t_i$'s. Similarly, the cumsum of
$\zeta_i \bar{R}_i(we^{\eta})$ with $\zeta_i$ constant within clusters (such
as $\bar{w}$) does not depend on it. Finally, for such $\zeta$ the
corresponding $\Delta_i$ can be expressed in terms of the cumsums of $(w_j
\zeta_j)_{1 \leq j \leq n}$.

Breslow's correction uses $\delta=0$; Efron's uses $\delta=1$.

## Probabilistic interpretation

In differentiating with respect to $\eta$ we can again consider

```{math}
:label: eq-loglik-final
\sum_{i=1}^n d_i \bar{w}_i
\log(\bar{R}_i(we^{\eta})/\bar{R}_i(w)).
```

This again has a probabilistic interpretation, as without ties, with a
slightly different base measure. Breslow's method fixes the risk set and
weights to be the same for all $i$ in the current cluster of failures
(recall we have a valid but arbitrary ordering by event time); this is
clearly exchangeable. Efron's method changes the weights as $i$ cycles
through the cluster of tied failures, downweighting all of the cluster by
$1/K$ at each step.

## First derivative

Proceeding as before, we differentiate

$$
\sum_{i=1}^n d_i \bar{w}_i \log \left( \bar{R}_i(we^{\eta}) / \bar{R}_i(w) \right).
$$

As without ties, this is

$$
w_k e^{\eta_k}\sum_{i=1}^n \frac{d_i \bar{w}_i \left[1_{\{t_k \geq t_i\}} - 1_{\{s_k \geq t_i\}} - \delta \cdot \sigma_i \cdot \left( 1_{\{i \leq \texttt{last}(k)\}} - 1_{\{i \leq \texttt{first}(k)-1\}}\right) \right]}{\bar{R}_i(we^{\eta})} = \sum_{i=1}^n d_i \bar{w}_i
\pi_{i,\eta}[k].
$$

## Second derivative

The $k$-th entry of the diagonal is the first derivative above, less

$$
w^2_k e^{2\eta_k}\sum_{i=1}^n \frac{d_i \bar{w}_i \left[1_{\{t_k \geq t_i\}} - 1_{\{s_k \geq t_i\}} - \delta \cdot \sigma_i\cdot\left( 1_{\{i \leq \texttt{last}(k)\}} - 1_{\{i \leq \texttt{first}(k)-1\}}\right)\right]^2}{\bar{R}_i(we^{\eta})^2}.
$$

Expanding the square, this is a difference of cumsums of the sequences

$$
\left(\frac{d_i \bar{w}_i}{\bar{R}_i(we^{\eta})^2}, \frac{d_i \bar{w}_i \sigma_i}{\bar{R}_i(we^{\eta})^2}, \frac{d_i \bar{w}_i \sigma^2_i}{\bar{R}_i(we^{\eta})^2}\right)_{1 \leq i \leq n}.
$$

[The last part](notes_hessian.md) gives the lemmas that turn these sums
into cumsums, and the full Hessian.

## A check against coxdev

On data with many ties, start times and zero weights, we compute $\sigma$,
$\bar{w}$ and $\bar{R}$ directly from their definitions, and from them the
gradient and the diagonal of the Hessian, for Breslow ($\delta=0$) and
Efron ($\delta=1$). `CoxDeviance` gives those of the deviance, $-2$ times
those of the log-likelihood.

```{code-cell} ipython3
import numpy as np
from coxdev import CoxDeviance

def sigma(w):
    """sigma for one cluster of tied failures, in the given order: 0, 1/K, ...
    over the entries with positive weight, 0 for those with weight 0."""
    pos = w > 0
    out = np.zeros(len(w))
    out[pos] = np.arange(pos.sum()) / pos.sum()
    return out

print(sigma(np.array([1/2, 0, 0, 3/4, 0, 2])))
```

```{code-cell} ipython3
rng = np.random.default_rng(1)
n = 80
t = np.ceil(rng.exponential(size=n) * 4) / 4 + 0.25     # a coarse grid: many ties
s = np.floor(rng.uniform(0, 0.5, n) * t * 8) / 8         # start times, s < t
d = rng.binomial(1, 0.7, n)
w = rng.uniform(0.5, 2, n)
w[rng.choice(n, 10, replace=False)] = 0
eta = rng.standard_normal(n) * 0.5
e = w * np.exp(eta)

def derivatives(delta):
    """The gradient and diagonal Hessian of the log-likelihood, from the
    definitions: a loop over the failures with positive weight."""
    log_term_grad = np.zeros(n)
    log_term_hess = np.zeros(n)
    for tau in np.unique(t[(d == 1) & (w > 0)]):
        C = np.flatnonzero((t == tau) & (d == 1) & (w > 0))   # the cluster, in any order
        wbar = w[C].mean()
        for i, sig in zip(C, sigma(w[C])):
            # r_i: the risk set at t_i, less delta * sigma_i times the cluster
            r = ((t >= tau) & (s < tau) & (w > 0)).astype(float)
            r[C] -= delta * sig
            Rbar = r @ e
            log_term_grad += wbar * e * r / Rbar
            log_term_hess += wbar * (e * r / Rbar - (e * r / Rbar)**2)
    return d * w - log_term_grad, -log_term_hess

for ties, delta in [('breslow', 0), ('efron', 1)]:
    grad, hess = derivatives(delta)
    result = CoxDeviance(event=t, start=s, status=d, sample_weight=w, tie_breaking=ties)(eta, w)
    print(f'{ties:8s} gradient: {np.abs(result.gradient + 2 * grad).max():.1e},'
          f' diagonal Hessian: {np.abs(result.diag_hessian + 2 * hess).max():.1e}')
```
