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

# Cumsums and the full Hessian

This part gives the two lemmas that turn the sums of [the previous
part](notes_ties.md) into cumsums, and uses them for the product of the
full Hessian with a vector. The proofs are at the end.

## Two lemmas

:::{prf:lemma} Risk sums at failure times
:label: lem-forward
Given a sequence $(Z_i)_{1 \leq i \leq n}$, let $E(Z)_i$ and $S(Z)_i$ denote
the reversed cumsums of $Z$ (padded with a 0 as the $(n+1)$-st element) in
an arbitrary, but otherwise valid, ordering of the $t_i$'s and the $s_i$'s
respectively. Let $\texttt{eventmap}(i)$ be a sequence such that, whenever
$d_i=1$, $\texttt{eventmap}(i) = \#\{j:s_j < t_i\}$. Then

$$
d_i \cdot \left( \sum_{j: t_j \geq t_i} Z_j - \sum_{m: s_m \geq t_i} Z_m \right) = d_i \cdot \left(E(Z)_{\texttt{first}(i)} - S(Z)_{\texttt{eventmap}(i)}\right).
$$

In particular, for $i$ such that $d_i=1$,

$$
\bar{r}_i'Z = E(Z)_{\texttt{first}(i)} - S(Z)_{\texttt{eventmap}(i)} - \delta \cdot \sigma_i \cdot \left(E(Z)_{\texttt{first}(i)} - E(Z)_{\texttt{last}(i)+1}\right).
$$
:::

If $d_i=1$, then $E(Z)_{\texttt{first}(i)}$ sums all the entries of $Z$ with
$t_j \geq t_i$, and under our assumptions $S(Z)_{\texttt{eventmap}(i)}$ is
the sum over the start times greater than or equal to $t_i$.

:::{note}
- Without start times, $\texttt{eventmap}(i)=n$ (0-based indexing) and
  $S(Z)_{\texttt{eventmap}(i)} \equiv 0$.
- The lemma says that the risk sums are computed correctly *at the failure
  times*. Without the factor $d_i$ the identity is false: we have the
  correct sums only at failure times. But that is all we need.
- To bring the result into disk order, we invert our fixed sort of the
  event times.
:::

:::{prf:lemma} Sums over the failures in a risk set
:label: lem-adjoint
Given a sequence $(Z_i)_{1 \leq i \leq n}$, let $\bar{E}(d \cdot Z)_i$
denote the cumsums of $d \cdot Z$ in a valid, but otherwise arbitrary,
order (with a 0 padded in as the first element). Then

$$
\sum_{i=1}^n d_i Z_i \left(1_{\{t_k \geq t_i\}} - 1_{\{s_k \geq t_i\}}\right) =
\bar{E}(d \cdot Z)_{\texttt{last}(k)+1} - \bar{E}(d \cdot Z)_{\texttt{startmap}(k)},
$$

where $\texttt{startmap}(k) = \# \{j: t_j \leq s_k\}$. Further,
$\texttt{startmap}(k) = \texttt{first}(\texttt{startmap}(k)) \leq
\texttt{first}(k)$.
:::

:::{note}
- Without start times, $\texttt{startmap} \equiv 0$ (0-based indexing) and
  $\bar{E}(d \cdot Z)_{\texttt{startmap}(k)} \equiv 0$.
- The elementwise multiplication by $d$ is crucial here.
:::

## The derivatives revisited

In the light of these lemmas, we can rewrite the first derivative as

$$
w_k e^{\eta_k}\sum_{i=1}^n \frac{d_i \bar{w}_i \left[1_{\{i \leq \texttt{last}(k)\}} - 1_{\{i < \texttt{startmap}(k)\}} - \delta \cdot \sigma_i \cdot \left( 1_{\{i \leq \texttt{last}(k)\}} - 1_{\{i \leq \texttt{first}(k)-1\}}\right) \right]}{\bar{R}_i(we^{\eta})}.
$$

Similarly, the corresponding term of the second derivative is

$$
w_k^2 e^{2 \eta_k}\sum_{i=1}^n \frac{d_i \bar{w}_i \left[1_{\{i \leq \texttt{last}(k)\}} - 1_{\{i < \texttt{startmap}(k)\}} - \delta \cdot \sigma_i \cdot \left( 1_{\{i \leq \texttt{last}(k)\}} - 1_{\{i \leq \texttt{first}(k)-1\}}\right) \right]^2}{\bar{R}_i(we^{\eta})^2}.
$$

The last conclusion of [](#lem-adjoint) resolves the products of all the
indicators above. The only one that needs any thought is

$$
1_{\{i < \texttt{startmap}(k)\}} \, 1_{\{i \leq \texttt{first}(k)-1\}} = 1_{\{i < \texttt{startmap}(k)\}},
$$

which holds because $\texttt{startmap}(k) \leq \texttt{first}(k)$.

## The full Hessian

Suppose we want to multiply the Hessian by a vector $(\xi_i)_{1 \leq i \leq
n}$. We write $(r,c)$ for the $r$-th row and $c$-th column of the Hessian.

### The non-diagonal term

There is a term

$$
-w_r w_c e^{ \eta_r + \eta_c} \sum_{i=1}^n \frac{d_i \bar{w}_i }{\bar{R}_i(we^{\eta})^2} \, \bar{r}_i'e_r \, \bar{r}_i'e_c,
$$

with $e_j$ the elementary basis vectors. We are free to rewrite this as

$$
-w_r w_c e^{ \eta_r + \eta_c} \sum_{i=1}^n \frac{d_i \bar{w}_i }{\bar{R}_i(we^{\eta})^2} \left[1_{\{t_r \geq t_i\}} - 1_{\{s_r \geq t_i\}} - \delta \cdot \sigma_i \cdot \left(1_{\{i \leq \texttt{last}(r)\}} - 1_{\{i \leq \texttt{first}(r)-1\}}\right)\right] \bar{r}_i'e_c,
$$

which lets us get rid of the two sums conveniently. Multiplying on the
right by $\xi$ gives

$$
-w_r e^{\eta_r} \sum_{i=1}^n \frac{d_i \bar{w}_i }{\bar{R}_i(we^{\eta})^2} \left[1_{\{t_r \geq t_i\}} - 1_{\{s_r \geq t_i\}} - \delta \cdot \sigma_i \cdot \left(1_{\{i \leq \texttt{last}(r)\}} - 1_{\{i \leq \texttt{first}(r)-1\}}\right)\right] \left(\sum_{c=1}^n \bar{r}_i'e_c \, w_c e^{\eta_c} \xi_c\right).
$$

The sum over $c$ is simply

$$
\bar{r}_i'(\xi w e^{\eta}) = \bar{R}_i(\xi w e^{\eta}).
$$

[](#lem-forward) tells us how to compute this correctly for the $i$ with
$d_i=1$. In the code this is `sum_over_risk_set`, applied to the argument
$\xi w e^{\eta}$. We then have to compute the sum over $i$

$$
-w_r e^{\eta_r} \sum_{i=1}^n \frac{d_i \bar{w}_i \bar{R}_i(\xi w e^{\eta})}{\bar{R}_i(w e^{\eta})^2} \left[1_{\{t_r \geq t_i\}} - 1_{\{s_r \geq t_i\}} - \delta \cdot \sigma_i \cdot \left(1_{\{i \leq \texttt{last}(r)\}} - 1_{\{i \leq \texttt{first}(r)-1\}}\right)\right].
$$

Up to the scaling by $-w_r e^{\eta_r}$, [](#lem-adjoint) handles this,
applied to the sequences

$$
\frac{\bar{w}_i \bar{R}_i(\xi w e^{\eta})}{\bar{R}_i(w e^{\eta})^2}, \qquad \frac{\bar{w}_i \sigma_i \bar{R}_i(\xi w e^{\eta})}{\bar{R}_i(w e^{\eta})^2}.
$$

In the code this is `sum_over_events`.

### The diagonal term

The diagonal term has $(r,c)$ entry

$$
\delta_{rc} \, w_r e^{\eta_r} \sum_{i=1}^n \frac{d_i \bar{w}_i \left[1_{\{t_r \geq t_i\}} - 1_{\{s_r \geq t_i\}} - \delta \cdot \sigma_i \cdot \left(1_{\{i \leq \texttt{last}(r)\}} - 1_{\{i \leq \texttt{first}(r)-1\}}\right)\right]}{\bar{R}_i(we^{\eta})}.
$$

Summing over $c$ in a multiplication on the right by $\xi$ gives

$$
\xi_r w_r e^{\eta_r} \sum_{i=1}^n \frac{d_i \bar{w}_i \left[1_{\{t_r \geq t_i\}} - 1_{\{s_r \geq t_i\}} - \delta \cdot \sigma_i \cdot \left(1_{\{i \leq \texttt{last}(r)\}} - 1_{\{i \leq \texttt{first}(r)-1\}}\right)\right]}{\bar{R}_i(we^{\eta})}.
$$

We again apply [](#lem-adjoint), to the sequences

$$
\frac{\bar{w}_i}{\bar{R}_i(we^{\eta})}, \qquad \frac{\bar{w}_i \sigma_i}{\bar{R}_i(we^{\eta})}.
$$

### Probabilistic interpretation

The Hessian has $(r,c)$ entry

$$
\sum_{i=1}^n d_i \bar{w}_i \left( \delta_{rc} \cdot \pi_{i,\eta}[r] -
\pi_{i,\eta}[r]\pi_{i,\eta}[c] \right).
$$

Multiplying on the right by $\xi$ gives

$$
\sum_{i=1}^n d_i \bar{w}_i \left( \xi_r \pi_{i,\eta}[r] -
\pi_{i,\eta}[r] E_{i,\eta}[\xi] \right) =
\sum_{i=1}^n d_i \bar{w}_i (\xi_r - E_{i,\eta}[\xi]) \, \pi_{i,\eta}[r].
$$

## Checks against coxdev

`CoxDeviance` keeps the arrays of the lemmas from its preprocessing: the
sort by event time (`event_order`) and by start time (`start_order`),
$\texttt{first}$, $\texttt{last}$, $\texttt{eventmap}$ (`event_map`) and
$\sigma$ (`scaling`). With them, [](#lem-forward) gives the Efron risk sums
at the failures:

```{code-cell} ipython3
import numpy as np
from coxdev import CoxDeviance

rng = np.random.default_rng(1)
n = 80
t = np.ceil(rng.exponential(size=n) * 4) / 4 + 0.25     # a coarse grid: many ties
s = np.floor(rng.uniform(0, 0.5, n) * t * 8) / 8         # start times, s < t
d = rng.binomial(1, 0.7, n)
w = rng.uniform(0.5, 2, n)
w[rng.choice(n, 10, replace=False)] = 0
eta = rng.standard_normal(n) * 0.5
e = w * np.exp(eta)

cox = CoxDeviance(event=t, start=s, status=d, sample_weight=w, tie_breaking='efron')
cox(eta, w)
c = cox._coxc
event_order, start_order = np.asarray(c.event_order), np.asarray(c.start_order)
first, last = np.asarray(c.first), np.asarray(c.last)
event_map, sigma, status = np.asarray(c.event_map), np.asarray(c.scaling), np.asarray(c.status)

# the reversed cumsums E and S, padded with a 0
E = np.concatenate([np.cumsum(e[event_order][::-1])[::-1], [0]])
S = np.concatenate([np.cumsum(e[start_order][::-1])[::-1], [0]])
lemma = E[first] - S[event_map] - sigma * (E[first] - E[last + 1])

# directly: the risk set at t_i, less sigma_i times the cluster of i
t_ev = t[event_order]
direct = np.array([e[(t >= t_ev[i]) & (s < t_ev[i])].sum()
                   - sigma[i] * e[event_order][first[i]:last[i] + 1].sum()
                   for i in range(n)])
failures = (status == 1) & (w[event_order] > 0)
print('at the failures:', np.abs(lemma - direct)[failures].max())
```

And the information, the Hessian of the log-likelihood's log term (minus
that of the log-likelihood), multiplies a vector as in the probabilistic
interpretation:

```{code-cell} ipython3
xi = rng.standard_normal(n)
H_xi = np.zeros(n)
for tau in np.unique(t[(d == 1) & (w > 0)]):
    C = np.flatnonzero((t == tau) & (d == 1) & (w > 0))   # the cluster of failures at tau
    K, wbar = len(C), w[C].mean()
    for k in range(K):                                     # Efron: sigma = k / K
        r = ((t >= tau) & (s < tau) & (w > 0)).astype(float)
        r[C] -= k / K
        pi = e * r / (r @ e)                               # pi_{i, eta}
        H_xi += wbar * (xi - pi @ xi) * pi
print('information:', np.abs(cox.information(eta, w) @ xi - H_xi).max())
```

## Proofs

### Proof of [](#lem-forward)

:::{prf:proof}
Consider the risk sum for an individual $i$ who fails ($d_i = 1$). It
decomposes in two parts by the definition of the risk set at time $t_i$,
the individuals $j$ with $s_j < t_i \leq t_j$:

$$
\sum_{j: s_j < t_i \leq t_j} Z_j = \sum_{j: t_j \geq t_i} Z_j - \sum_{m: s_m \geq t_i, t_m \geq t_i} Z_m.
$$

As the start time is always strictly less than the event time ($s_m <
t_m$), $s_m \geq t_i$ implies $t_m \geq t_i$, and the second sum simplifies:

$$
\sum_{j: s_j < t_i \leq t_j} Z_j = \sum_{j: t_j \geq t_i} Z_j - \sum_{m: s_m \geq t_i} Z_m.
$$

Use 0-based indexing. The sequence $Z$ is sorted in ascending order of
event times $t$, and $\texttt{first}(i)$ is the first occurrence of the
event time $t_i$ in this sorted sequence. So summing $Z_j$ over all $j$
with $t_j \geq t_i$ is summing the sequence from index $\texttt{first}(i)$
to the end ($n-1$), which the reversed cumsum $E(Z)_{\texttt{first}(i)}$
computes.

Similarly, let $S(Z)$ be the reversed cumsum of $Z$ sorted by start times
$s$. The quantity $\texttt{eventmap}(i) = \#\{j : s_j < t_i\}$ is the
number of start times strictly less than $t_i$. In the array sorted by
start time, the indices $0, 1, \dots, \texttt{eventmap}(i)-1$ are those
with $s_m < t_i$, and the indices from $\texttt{eventmap}(i)$ on are those
with $s_m \geq t_i$. So $S(Z)_{\texttt{eventmap}(i)}$ is the sum of $Z_m$
over $s_m \geq t_i$.

Multiplying by $d_i$ (which gives 0 on both sides if $d_i=0$) establishes
the first equality:

$$
d_i \cdot \left( \sum_{j: t_j \geq t_i} Z_j - \sum_{m: s_m \geq t_i} Z_m \right) = d_i \cdot \left( E(Z)_{\texttt{first}(i)} - S(Z)_{\texttt{eventmap}(i)} \right).
$$

For the second part, recall the definition [](#eq-risk-sum) of
$\bar{R}_i$:

$$
\bar{R}_i(Z) = \sum_{j: t_j \geq t_i} Z_j - \sum_{m: s_m \geq t_i} Z_m - \delta \cdot \sigma_i \cdot \Delta_i(Z),
$$

where $\Delta_i(Z) = \sum_{j: C_j=C_i} Z_j$ is the sum over the cluster of
tied event times at $t_i$. The cluster spans the indices
$\texttt{first}(i)$ to $\texttt{last}(i)$, so in terms of the reversed
cumsum, $\Delta_i(Z) = E(Z)_{\texttt{first}(i)} - E(Z)_{\texttt{last}(i)+1}$.
Substituting this into $\bar{R}_i(Z)$ gives

$$
\bar{r}_i'Z = E(Z)_{\texttt{first}(i)} - S(Z)_{\texttt{eventmap}(i)} - \delta \cdot \sigma_i \cdot \left( E(Z)_{\texttt{first}(i)} - E(Z)_{\texttt{last}(i)+1} \right).
$$
:::

### Proof of [](#lem-adjoint)

:::{prf:proof}
The indicator $1_{\{t_k \geq t_i\}} - 1_{\{s_k \geq t_i\}}$ is 1 if and
only if $s_k < t_i \leq t_k$, and 0 otherwise. So the sum accumulates $d_i
Z_i$ over the interval $(s_k, t_k]$.

Use 0-based indexing for the arrays sorted by event time $t$, and find the
bounds of this interval in terms of indices. The upper bound is $t_i \leq
t_k$: in the sorted array, the elements satisfying this occupy the indices
$0$ through $\texttt{last}(k)$, ${\texttt{last}(k) + 1}$ elements in all.
The lower bound is $s_k < t_i$, which excludes the elements with $t_i \leq
s_k$. There are $\texttt{startmap}(k) = \#\{j : t_j \leq s_k\}$ of these,
so the excluded indices are $0$ through $\texttt{startmap}(k) - 1$. So the
interval $s_k < t_i \leq t_k$ is the index range $i \in
\{\texttt{startmap}(k), \dots, \texttt{last}(k)\}$.

Let $\bar{E}(d \cdot Z)$ be the forward cumsum of $d \cdot Z$, padded with
a 0 at index 0: $\bar{E}(d \cdot Z)_m = \sum_{i=0}^{m-1} d_i Z_i$. The sum
over the index range is the sum up to $\texttt{last}(k)$, less the sum up
to $\texttt{startmap}(k)-1$:

$$
\sum_{i=\texttt{startmap}(k)}^{\texttt{last}(k)} d_i Z_i = \bar{E}(d \cdot Z)_{\texttt{last}(k)+1} - \bar{E}(d \cdot Z)_{\texttt{startmap}(k)}.
$$

This establishes the equality. The last claim, $\texttt{startmap}(k) =
\texttt{first}(\texttt{startmap}(k))$, holds because all the individuals
$j$ with the same event time fall either entirely within the threshold
$t_j \leq s_k$ or entirely outside it; so the count is the
$\texttt{first}$ index of a block of tied event times, and in particular
at most $\texttt{first}(k)$, as $s_k < t_k$.
:::
