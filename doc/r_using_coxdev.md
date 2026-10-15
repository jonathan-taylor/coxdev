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

# Using coxdev from R


```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 6, repr.plot.height = 4)
```

`coxdev` computes the deviance of the Cox proportional hazards model, its
gradient and the diagonal of its Hessian with respect to the linear predictor,
the information as a matrix-free operator, the score residuals, the baseline
hazard, and draws of the score. This page fits a model by Newton's
method, computes robust standard errors and survival curves, and shows what
keeping the sorted data in a `CoxDeviance` object saves. Sampling the score
has [its own page](r_sampling.md). Both are the vignettes of the R package,
`vignette("coxdev", package = "coxdev")`, and match the Python pages
[Using coxdev](using_coxdev.md) and [Sampling the score](sampling.md).

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

The response can also be given as vectors, as in the Python package:
`CoxDeviance$new(event = stop, status = status, start = start)`.

## Deviance, gradient and information

`compute()` returns the deviance and its derivatives with respect to the
linear predictor `eta`. `information()` returns the information (half the
Hessian of the deviance) as an `n` by `n` operator. It is never formed: a
product with it costs about as much as a gradient. For a design `X`,
`crossprod(X, I %*% X)` is the information of the coefficients, so Newton's
method takes a few lines.

```{code-cell} r
b <- rep(0, p)
for (iter in 1:10) {
  eta <- drop(X %*% b)
  res <- cox$compute(eta)
  score <- -drop(crossprod(X, res$gradient)) / 2
  info <- crossprod(X, cox$information(eta) %*% X)
  step <- solve(info, score)
  b <- b + step
  if (max(abs(step)) < 1e-10) break
}
iter
b
```

These are the estimates of `coxph` from `survival`, with the same standard
errors, and of `glmnet` with no penalty, `lambda = 0`:

```{code-cell} r
fit <- coxph(y ~ X, ties = "efron")
estimates <- rbind(coxdev = b, coxph = coef(fit))
if (requireNamespace("glmnet", quietly = TRUE) &&
    utils::packageVersion("glmnet") >= "5.0") {
  fit_glmnet <- glmnet::glmnet(X, y, family = "cox", lambda = 0, cox.ties = "efron")
  estimates <- rbind(estimates, glmnet = drop(as.matrix(coef(fit_glmnet))))
}
estimates
rbind(coxdev = sqrt(diag(solve(info))), coxph = sqrt(diag(vcov(fit))))
```

## Robust standard errors and the baseline hazard

The score residuals of the design give the robust (sandwich) variance,
`solve(I) %*% crossprod(U) %*% solve(I)`:

```{code-cell} r
eta_hat <- drop(X %*% b)
U <- cox$score_residuals(eta_hat, X)
info_hat <- crossprod(X, cox$information(eta_hat) %*% X)
robust <- solve(info_hat, t(solve(info_hat, crossprod(U))))
fit_robust <- coxph(y ~ X, ties = "efron", id = seq_len(n), robust = TRUE)
rbind(coxdev = sqrt(diag(robust)), coxph = sqrt(diag(vcov(fit_robust))))
```

`baseline_hazard()` gives Efron's estimate of the cumulative baseline
hazard (Breslow's with `tie_breaking = "breslow"`), the
`basehaz(fit, centered = FALSE)` of `coxph`. The survival curve of a subject
with covariates `x0` is `exp(-cumulative_hazard * exp(sum(x0 * b)))`:

```{code-cell} r
H <- cox$baseline_hazard(eta_hat)
head(H)
bh <- basehaz(fit, centered = FALSE)
all.equal(H$cumulative_hazard, bh$hazard[match(H$time, bh$time)])

plot(NULL, xlim = c(0, max(H$time)), ylim = c(0, 1), xlab = "time", ylab = "survival")
for (x1 in c(-1, 0, 1)) {
  surv <- exp(-H$cumulative_hazard * exp(b[1] * x1))
  lines(c(0, H$time), c(1, surv), type = "s", lty = x1 + 2)
}
legend("topright", paste("x1 =", c(-1, 0, 1)), lty = 1:3, bty = "n")
```

## What keeping state saves

Creating a `CoxDeviance` sorts the data, once. Each `compute()` after that
is linear in `n`, and repeating a call with the same linear predictor and
weights returns the cached result. A stateless function has to sort on every
call. That cost is paid at every step of an optimizer, and at every
product with the information.

We time 50 evaluations of the deviance and gradient at 100,000 observations,
each at a different linear predictor.

```{code-cell} r
set.seed(2)
N <- 1e5
XN <- matrix(rnorm(N * p), N, p)
etaN <- drop(XN %*% beta)
stopN <- ceiling(rexp(N, exp(etaN)) * 20) / 20
statusN <- rbinom(N, 1, 0.7)
startN <- floor(runif(N, 0, 0.5) * stopN * 40) / 40
yN <- Surv(startN, stopN, statusN)
etas <- lapply(1:50, function(i) etaN + rnorm(N, sd = 0.1))

per_call <- function(expr) {
  t <- system.time(expr)[["elapsed"]]
  sprintf("%.1f ms", 1000 * t / 50)
}
setup <- system.time(coxN <- CoxDeviance$new(yN))[["elapsed"]]
timings <- c(
  "one object, reused" = per_call(for (e in etas) coxN$compute(e)),
  "a new object for each call" = per_call(for (e in etas) CoxDeviance$new(yN)$compute(e)),
  "one object, same eta again (cached)" = per_call(for (e in etas) coxN$compute(etas[[1]])),
  "one information product" = per_call({
    I <- coxN$information(etas[[1]])
    for (e in etas) I %*% XN[, 1]
  })
)
if (requireNamespace("glmnet", quietly = TRUE) &&
    utils::packageVersion("glmnet") >= "5.0") {
  timings["glmnet::coxnet.deviance, then glmnet:::coxgrad"] <- per_call(for (e in etas) {
    glmnet::coxnet.deviance(pred = e, y = yN, std.weights = FALSE, cox.ties = "efron")
    glmnet:::coxgrad(e, yN, rep(1, N), std.weights = FALSE, cox.ties = "efron")
  })
}
cat(sprintf("Time per call, n = %d. Creating the object (the sort) took %.0f ms.\n",
            N, 1000 * setup))
data.frame(time = timings)
```

The times depend on the machine, but the pattern is stable. Reusing the
object avoids the sort, which costs several times as much as an evaluation.
glmnet's functions run the same C++ code, but have to start from scratch on
every call. The information operator is what makes the Hessian usable at
this size: as a dense matrix it would take 8 N² bytes, 74.5 GB.
