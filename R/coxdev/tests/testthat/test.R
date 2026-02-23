library(coxdev)
set.seed(10101)
nobs <- 100; nvars <- 10
nzc <- nvars %/% 3
x <- matrix(rnorm(nobs * nvars), nobs, nvars)
beta <- rnorm(nzc)
fx <- x[, seq(nzc)] %*% beta / 3
hx <- exp(fx)
ty <- rexp(nobs,hx)
tcens <- rbinom(n = nobs, prob = 0.3, size = 1)
cox_deviance <- make_cox_deviance(event = ty,
                                  status = tcens,
                                  weight = rep(1.0, length(ty)),
                                  tie_breaking = 'efron')
result  <- cox_deviance$coxdev(linear_predictor = fx)
str(result)
tx  <- t(x)
h <- cox_deviance$information(fx)
I <- tx %*% h(x)
cov  <- solve(I)

