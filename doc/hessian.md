---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import numpy as np
import pandas as pd

from coxdev import CoxDeviance
                       
```

# A small dataset (with ties)

- Event times (or "stop" times) $t_i$.

- Start times: $s_i$.

```{code-cell} ipython3
data_df = pd.read_csv('dataset.csv', index_col='index')[:20]
data_df
n = data_df.shape[0]
```

## Key sequences

+++

- `event_order`: sorts event times into increasing order (according to this joint sort)
- `start_order`: sorts start times into increasing order (according to this joint sort)
- `first`: for a failure time, which may have ties, this is the first entry of `event_order` in this set of ties -- the entries of `first` that are not failure times are not used, but they are defined consistently in that no ties of non-failure times are material to the computations.
- `last`: analogous to `first` but the last time
- `event_map`: for the $i$-th event time, this is the number of start times stricly less than this event time: $\text{event\_map}[i] = \# \{j: s_j < t_i\}$.
- `start_map`: for the $i$-th start time, this is the number of event times less than or equal to $s_i$:
  $\text{start\_map}[i] = \# \{j: t_j \leq s_i\}$.

+++

## Inverse properties

Note
$$
\begin{aligned}
i < \text{event\_map}(k) & \iff  \text{start\_map}(i) < k \\
k \geq \text{first\_map}(i) & \iff  \text{last\_map}(k) \geq  i \\
\end{aligned}
$$


```{code-cell} ipython3
from sympy import Symbol, Function, simplify, expand

last_ = Function('last')
start_ = Function('start')
first_ = Function('first')
k = Symbol('k')
l = Symbol('l')
s_E = Symbol('sigma_E')
d_E = Symbol('delta_E')
d_S = Symbol('delta_S')

E_k = last_(k) - d_S * start_(k-1) - d_E * s_E * (last_(k) - first_(k-1))
```

#### Off-diagonal Hessian entries

We can similarly deduce how to evaluate off-diagonal entries of the Hessian, though
we have not implemented all of these yet.

```{code-cell} ipython3
E_l = last_(l) - d_S * start_(l) - d_E * s_E * (last_(l) - first_(l-1))
prod = expand(E_k * E_l)
```

## Breslow with start times

```{code-cell} ipython3
prod.subs(d_S, 1).subs(d_E, 0).as_ordered_terms()
```

That is, the $(r,c)$ entry of the Hessian is
$$
-\frac{\partial^2}{\partial \eta_r \partial \eta_c}\left[\sum_{i=1}^n w_i d_i \log(R_i(\eta))\right]
$$
This is
$$
\sum_{i=1}^n \frac{1}{R_i(\eta)^2} \left(1_{\{r \geq \text{first}(i)\}} - 1_{\{r > \text{event\_map}(i)\}} \right)\left(1_{\{c \geq \text{first}(i)\}} - 1_{\{c > \text{event\_map}(i)\}} \right)
$$

+++

Consider multiplication on the right by the vector $(\zeta_c)_{1 \leq c \leq n}$.
The $r$-th entry of the product is 
$$
\sum_{c=1}^n \left[\sum_{i=1}^n \frac{1}{R_i(\eta)^2} \left(1_{\{r \geq \text{first}(i)\}} - 1_{\{r > \text{event\_map}(i)\}} \right)\left(1_{\{c \geq \text{first}(i)\}} - 1_{\{c > \text{event\_map}(i)\}} \right) \right] \zeta_c
$$

+++

Let's define the reversed cumsum
$$
{\cal S}(\zeta)[i] = \sum_{j:j\geq i} \zeta_j, 1 \leq i \leq n.
$$

+++

The $r$-th entry of the matrix vector product is
$$
\sum_{i=1}^n \frac{{\cal S}(\zeta)[\text{first}(i)] - {\cal S}(\zeta)[\text{event\_map(i)}+1]}{R_i(\eta)^2} \left(1_{\{r \geq \text{first}(i)\}} - 1_{\{r > \text{event\_map}(i)\}} \right)
$$
This is the vector `after_1st_cumsum` in the code.

+++

So, the $r$-th entry of the product can be expressed in terms of the cumsums of
the sequence
$$
i \mapsto \frac{{\cal S}(\zeta)[\text{first}(i)] - {\cal S}(\zeta)[\text{event\_map(i)}+1]}{R_i(\eta)^2}.
\overset{def}{=} G(\zeta,\eta, \delta_E=0)[i] \qquad (*) $$
Specifically, the $r$-th entry is the difference between the $\text{last}(r)$-th cumsum and 
$\text{start\_map}(i)-1$-st entry. The name of the variable used for this is `cumsum_2nd`.

Computing the sequence costs a reverse cumsum and then lookup. Completing the product requires
another cumsum and lookup.

+++

## Efron

It will be similar, but a little more tedious. First, there will be four terms in the
sequence analogous to $(*)$.
There will then be 4 different entries of the basic reversed cumsum.
 Set 
$$
\delta_E = \begin{cases} 0 & \text{Breslow} \\ 1 & \text{Efron} \end{cases}.
$$

+++

The analog of $(*)$ when $\delta_E=1$ is
$$
i \mapsto \frac{{\cal S}(\zeta)[\text{first}(i)] - {\cal S}(\zeta)[\text{event\_map(i)}+1] -  \sigma_i \cdot\left({\cal S}(\zeta)[\text{first}(i)] - {\cal S}(\zeta)[\text{last}(i)+1]\right)}{R_i(\eta)^2} \qquad (**)$$.

We see, then, that in either case we compute the reversed cumsum of
$$
i \mapsto \frac{{\cal S}(\zeta)[\text{first}(i)] - {\cal S}(\zeta)[\text{event\_map(i)}+1] - \delta_E \cdot \sigma_i \cdot\left({\cal S}(\zeta)[\text{first}(i)] - {\cal S}(\zeta)[\text{last}(i)+1]\right)}{R_i(\eta)^2}  \overset{def}{=} G(\eta,\zeta,\delta_E)[i].
$$

We see that Breslow indeed uses $G(\eta,\zeta,0)$. Its cost (besides the copy) is requires the same reversed cumsum as Breslow. 

Having formed this sequence `cumsum_2nd_0` in the code, the Efron version will again use the cumsums (of 
$G(\eta, \zeta,1)$ instead of $G(\eta,\zeta,0)$) but there is another cumsum needed, namely 
$$
i \mapsto \sigma_E(i) \cdot G(\eta,\zeta,1)[i].
$$
This is called `cumsum_2nd_1` in the code.

+++

Why another cumsum? Well, having defined $G(\eta,\zeta,\delta)$ we can see that, in the 
Efron case, the $r$-th entry of the product is
$$
\sum_{i=1}^n G(\eta,\zeta,1)[i] \left(1_{\{r \geq \text{first}(i)\}} - 1_{\{r > \text{event\_map}(i)\}} - \sigma_E(i) \cdot  \left( 1_{\{i \leq  \text{last}(r)\}} - 1_{\{i \leq \text{first}(r)-1\}}\right) \right)
$$

+++

This can be expressed in terms of the cumsums of the sequences
$$
\left(G(\eta,\zeta,1)[i] \right)_{1 \leq i \leq n}, \qquad \left(\sigma_E(i) \cdot G(\eta,\zeta,1)[i] \right)_{1 \leq i \leq n}
$$

```{code-cell} ipython3

```
