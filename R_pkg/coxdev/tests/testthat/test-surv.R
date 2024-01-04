context("Check against survival package")

tol  <- 1e-10
i  <- j <- 0
get_coxph <- function(event,
                      status,
                      X,
                      beta,
                      sample_weight,
                      start=None,
                      ties='efron') {

  if (length(start) == length(status)) start  <- as.numeric(start)
  start <- as.numeric(start)
  status <- as.numeric(status)
  event <- as.numeric(event)
  weight <- as.numeric(sample_weight)
  if (length(start) == length(status)) {
    y <- Surv(start, event, status)
  } else {
    y <- Surv(event, status)
  }
  F <- coxph(y ~ X, init=beta, weights=sample_weight, control=coxph.control(iter.max=0), ties=ties, robust=FALSE)
  G <- colSums(coxph.detail(F)$scor)
  D <- F$loglik
  cov <- vcov(F)
  list(G = -2 * G, D = -2 * D, cov = cov)
}

check_coxph <- function(tie_types,
                       tie_breaking,
                       sample_weight,
                       have_start_times,
                       nrep=5,
                       size=5,
                       tol=1e-10) {

  data <- simulate_df(tie_types,
                      nrep,
                      size)
  if (have_start_times)
       start <- data$start
  else
    start <- NA

  cox_deviance  <- make_cox_deviance(event = data$event, start = start, status = data$status, weight = weight, tie_breaking = tie_breaking)
  n <- nrow(data)
  p <- n %/% 2
  X <- matrix(rnorm(n * p), nrow = n)
  beta <- rnorm(p) / sqrt(n)
  weight <- sample_weight(n)
  tX  <- t(X)
  C <- cox_deviance$coxdev(X %*% beta, weight)
  h <- cox_deviance$information(X %*% beta, weight)
  I <- tX %*% h(X)
  expect_true(all_close(I, t(I)),
              info = "Information matrix not symmetric")

  ## stopifnot(all_close(I, t(I)))
  new_cov  <- solve(I)

  coxph_result  <- get_coxph(event = data$event,
                             status = data$status,
                             beta  = beta,
                             sample_weight = weight,
                             start = start,
                             ties = tie_breaking,
                             X = X)
  G_coxph <- coxph_result$G
  D_coxph <- coxph_result$D[1]
  cov_coxph  <- coxph_result$cov
  ## stopifnot(all_close(D_coxph, C$deviance - 2 * C$loglik_sat))
  ## stopifnot(rel_diff_norm(G_coxph, tX %*% C$gradient) < tol)
  ## stopifnot(rel_diff_norm(new_cov, cov_coxph) < tol)

  expect_true(all_close(D_coxph, C$deviance - 2 * C$loglik_sat),
              info = "Deviance mismatch")
  if (rel_diff_norm(G_coxph, tX %*% C$gradient) >= tol) {
    saveRDS(list(event = data$event,
                 status = data$status,
                 beta  = beta,
                 sample_weight = weight,
                 start = start,
                 ties = tie_breaking,
                 X = X), sprintf("~/tmp/g%d.RDS", i))
      i <<- i + 1
    }
  expect_true(rel_diff_norm(G_coxph, tX %*% C$gradient) < tol,
              info = "Gradient mismatch")

  if (rel_diff_norm(new_cov, cov_coxph) >= tol) {
    saveRDS(list(event = data$event,
                 status = data$status,
                 beta  = beta,
                 sample_weight = weight,
                 start = start,
                 ties = tie_breaking,
                 X = X), sprintf("~/tmp/h%d.RDS", j))
    j <<- j + 1
  }
  expect_true(rel_diff_norm(new_cov, cov_coxph) < tol,
              info = "Covariance mismatch")
}

for (tie_type in all_combos) {
  for (tie_breaking in c('efron', 'breslow')) {
    for (sample_weight in list(just_ones, sample_weights)) {
      for (have_start_time in c(TRUE, FALSE)) {
        check_coxph(tie_type,
                    tie_breaking,
                    sample_weight,
                    have_start_time,
                    nrep=5,
                    size=5,
                    tol=1e-10)
      }
    }
  }
}



