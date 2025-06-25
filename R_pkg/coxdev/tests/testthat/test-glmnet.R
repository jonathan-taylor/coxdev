context("Check against glmnet package")
tol  <- 1e-10

get_glmnet_result <- function(event,
                              status,
                              start,
                              eta,
                              weight) {
  event <- as.numeric(event)
  status <- as.numeric(status)
  weight <- as.numeric(weight)
  eta <- as.numeric(eta)
  if (length(start) == length(status)) {
    y <- survival::Surv(start, event, status)
    D_R <- glmnet:::coxnet.deviance3(pred=eta, y=y, weight=weight, std.weights=FALSE)
    G_R <- glmnet:::coxgrad3(eta, y, weight, std.weights=FALSE, diag.hessian=TRUE)
    H_R <- attr(G_R, 'diag_hessian')
  } else {
    y <- Surv(event, status)
    D_R <- glmnet:::coxnet.deviance2(pred=eta, y=y, weight=weight, std.weights=FALSE)
    G_R <- glmnet:::coxgrad2(eta, y, weight, std.weights=FALSE, diag.hessian=TRUE)
    H_R <- attr(G_R, 'diag_hessian')
  }
  list(D = D_R, G = as.numeric(-2 * G_R), H = -2 * H_R)
}


check_glmnet <- function(tie_types,
                        sample_weight,
                        have_start_times,
                        nrep=5,
                        size=5,
                        tol=1e-10) {

  data <- simulate_df(tie_types,
                      nrep,
                      size)

  n <- nrow(data)
  eta <- rnorm(n)
  weight <- sample_weight(n)
  if (have_start_times) {
    start <- data$start
  } else {
    start <- NA
  }
  glmnet_result <- get_glmnet_result(data$event,
                                     data$status,
                                     start,
                                     eta,
                                     weight)
  D_glmnet  <- glmnet_result$D
  G_glmnet <- glmnet_result$G
  H_glmnet <- glmnet_result$H

  cox_deviance <- make_cox_deviance(event = data$event, start = start, status = data$status,
                                    weight = weight, tie_breaking = 'breslow')
  C <- cox_deviance$coxdev(eta, weight)

  expect_true(all_close(D_glmnet, C$deviance),
              info = "Deviance mismatch")
  expect_true(rel_diff_norm(G_glmnet, C$gradient) < tol,
              info = "Gradient mismatch")
  expect_true(rel_diff_norm(H_glmnet, C$diag_hessian) < tol,
              info = "Covariance mismatch")
  ## stopifnot(all_close(D_glmnet, C$deviance))
  ## stopifnot(rel_diff_norm(G_glmnet, C$gradient) < tol)
  ## stopifnot(rel_diff_norm(H_glmnet, C$diag_hessian) < tol)
}

for (tie_type in all_combos) {
  for (sample_weight in list(just_ones, sample_weights)) {
    for (have_start_time in c(TRUE, FALSE)) {
      check_glmnet(tie_type,
                   sample_weight,
                   have_start_time,
                   nrep=5,
                   size=5,
                   tol=1e-10)
    }
  }
}
