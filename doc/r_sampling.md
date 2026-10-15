---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: R
  language: R
  name: ir
---

# Sampling the score in R


```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 6, repr.plot.height = 4)
```

`CoxDeviance$sample_score()` draws the score of the Cox model at a linear
predictor, by one of four methods, and `CoxDeviance$wild_score()` gives the
wild draw for multipliers of your choosing. This vignette compares the
methods on data with heavy ties. For the basics of `CoxDeviance`, see
[Using coxdev from R](r_using_coxdev.md). This page is the vignette
`vignette("sampling", package = "coxdev")`, and matches the Python page
[Sampling the score](sampling.md).

```{code-cell} r
library(coxdev)
library(survival)
```

## A survival response

Survival times on a coarse grid, so that there are many ties, with delayed
entry and censoring.

```{code-cell} r
set.seed(1)
n <- 500
p <- 3
X <- matrix(rnorm(n * p), n, p, dimnames = list(NULL, c("x1", "x2", "x3")))
beta <- c(0.5, -0.5, 0.25)
time <- rexp(n, exp(drop(X %*% beta)))
cens <- rexp(n, 0.5)
stop <- ceiling(pmin(time, cens) * 4) / 4
status <- as.integer(time <= cens)
start <- floor(runif(n, 0, 0.5) * stop * 8) / 8
y <- Surv(start, stop, status)

cox <- CoxDeviance$new(y, tie_breaking = "efron")
cox
```

We sample at the estimate of the coefficients, from `glmnet` with no
penalty, `lambda = 0`, or from `coxph` when `glmnet` is not installed:

```{code-cell} r
if (requireNamespace("glmnet", quietly = TRUE) &&
    utils::packageVersion("glmnet") >= "5.0") {
  fit <- glmnet::glmnet(X, y, family = "cox", lambda = 0, cox.ties = "efron")
  b <- drop(as.matrix(coef(fit)))
} else {
  b <- coef(coxph(y ~ X, ties = "efron"))
}
b
```

## The methods

Each draw `r` has one entry per observation, and `crossprod(X, r)` is a
draw of the score of the coefficients. We draw at the estimate `b`, by four
methods:

- `"multinomial"`: the failures at each event time drawn with replacement,
  with probabilities proportional to `exp(eta)`. This is the sampling model
  of Breslow's likelihood, so the covariance is the Breslow information.
- `"fisher"`: the set of failures at each event time drawn from Fisher's
  noncentral hypergeometric distribution. This is the sampling model of the
  exact partial likelihood, whose information is smaller than Breslow's when
  ties are heavy.
- `"bernoulli"`: each subject at risk fails independently, with
  probability `min(1, d p)`, where `d` is the number of failures at the
  event time and `p` the subject's share of the risk. Its mean is zero only
  while `d p < 1`: under heavy ties the cap binds, and the draws are biased.
- `"wild"`: the score residuals times independent normals. Its covariance is
  the sandwich, which does not rely on the model being right.

```{code-cell} r
eta_hat <- drop(X %*% b)
methods <- c("multinomial", "fisher", "bernoulli", "wild")
draws <- lapply(methods, function(m) {
  crossprod(X, cox$sample_score(eta_hat, method = m, n_draws = 4000, seed = 1))
})
names(draws) <- methods
```

Here are the means of the draws, in standard deviations, and their
variances relative to the Breslow information:

```{code-cell} r
breslow <- CoxDeviance$new(y, tie_breaking = "breslow")
info_breslow <- crossprod(X, breslow$information(eta_hat) %*% X)
summary_table <- t(sapply(draws, function(d) {
  c(rowMeans(d) / sqrt(diag(info_breslow)), diag(cov(t(d))) / diag(info_breslow))
}))
colnames(summary_table) <- c(paste("mean", colnames(X)), paste("variance", colnames(X)))
round(summary_table, 2)
```

The multinomial draws have the Breslow information as covariance, and the
Fisher draws a smaller one, the information of the exact likelihood. The
Bernoulli draws are biased: at these ties, `d p` often exceeds 1, so the
cap `min(1, d p)` gives too few failures to the subjects most at risk.

The wild draws estimate the sandwich, `crossprod(U)` for the score
residuals `U` of the design, the middle of the robust variance
`solve(I) %*% crossprod(U) %*% solve(I)` of `coxph(robust = TRUE)`:

```{code-cell} r
U <- cox$score_residuals(eta_hat, X)
sandwich <- crossprod(U)
fit_robust <- coxph(y ~ X, ties = "efron", id = seq_len(n), robust = TRUE)
I_hat <- solve(fit_robust$naive.var)
rbind(wild = diag(cov(t(draws$wild))), sandwich = diag(sandwich),
      coxph = diag(I_hat %*% fit_robust$var %*% I_hat))
```

The distributions of the first coefficient's score:

```{code-cell} r
cols <- c(multinomial = "black", fisher = "firebrick", bernoulli = "darkorange", wild = "steelblue")
dens <- lapply(draws, function(d) density(d[1, ]))
plot(NULL, xlim = c(-1, 1) * 4 * sqrt(info_breslow[1, 1]),
     ylim = c(0, 1.35 * max(sapply(dens, function(d) max(d$y)))),
     xlab = "score of x1", ylab = "density")
for (m in methods) lines(dens[[m]], col = cols[m], lwd = 2)
curve(dnorm(x, sd = sqrt(info_breslow[1, 1])), add = TRUE, lty = 2)
legend("topleft", c(methods, "N(0, Breslow info.)"), cex = 0.8,
       col = c(cols, "black"), lty = c(rep(1, 4), 2), lwd = c(rep(2, 4), 1), bty = "n")
```

## Choosing the multipliers

`wild_score()` gives the wild draw for multipliers you choose: the sum over
observations of the multiplier times the score residual, `G %*% U`. With multipliers
all one it is the score itself. Other multipliers give other wild
bootstraps, here Rademacher signs in place of normals:

```{code-cell} r
all.equal(cox$wild_score(eta_hat, rep(1, n)), -cox$compute(eta_hat)$gradient / 2)
rademacher <- replicate(4000, {
  drop(crossprod(X, cox$wild_score(eta_hat, sample(c(-1, 1), n, replace = TRUE))))
})
rbind(rademacher = diag(cov(t(rademacher))), sandwich = diag(sandwich))
```

## Reproducibility

The same seed gives the same draws on every platform, and from the Python
package. With `seed = NULL`, the seed comes from R's generator, so
`set.seed()` makes the draws reproducible.

```{code-cell} r
identical(cox$sample_score(eta_hat, method = "fisher", seed = 7),
          cox$sample_score(eta_hat, method = "fisher", seed = 7))
```
