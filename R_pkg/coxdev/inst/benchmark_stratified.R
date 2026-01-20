#!/usr/bin/env Rscript
#' Benchmark: C++ Stratified vs R Loop Implementation
#'
#' Compares the C++ stratified Cox implementation against
#' an R loop that calls the unstratified version per stratum
#' (similar to what glmnet does internally).

library(coxdev)

# R loop implementation (baseline - similar to glmnet's approach)
stratified_r_loop <- function(event, status, strata, eta, weights = NULL,
                               start = NULL, tie_breaking = "efron") {
  unique_strata <- sort(unique(strata))
  n <- length(event)
  if (is.null(weights)) weights <- rep(1, n)

  total_dev <- 0
  gradient <- numeric(n)
  diag_hessian <- numeric(n)


  for (s in unique_strata) {
    idx <- which(strata == s)
    cox_s <- make_cox_deviance(
      event = event[idx],
      status = status[idx],
      start = if (!is.null(start)) start[idx] else NULL,
      tie_breaking = tie_breaking
    )
    result <- cox_s$coxdev(eta[idx], weights[idx])
    total_dev <- total_dev + result$deviance
    gradient[idx] <- result$gradient
    diag_hessian[idx] <- result$diag_hessian
  }

  list(deviance = total_dev, gradient = gradient, diag_hessian = diag_hessian)
}

benchmark_single <- function(n, n_strata, n_calls = 50, with_ties = FALSE,
                              with_start = FALSE, seed = 42) {
  set.seed(seed)

  if (with_ties) {
    event <- sample(1:(n/10), n, replace = TRUE)
  } else {
    event <- rexp(n, rate = 0.2)
  }

  status <- rbinom(n, 1, 0.7)
  strata <- sample(1:n_strata, n, replace = TRUE)
  eta <- rnorm(n) * 0.5
  weights <- runif(n, 0.5, 2.0)

  if (with_start) {
    start <- runif(n, 0, 2)
    event <- start + abs(event)
  } else {
    start <- NULL
  }

  # Initialize C++ version
  cpp_init_time <- system.time({
    cpp_cox <- make_stratified_cox_deviance(
      event = event,
      status = status,
      strata = strata,
      start = start
    )
  })["elapsed"]

  # Warm up
  cpp_cox$coxdev(eta, weights)
  stratified_r_loop(event, status, strata, eta, weights, start)

  # Benchmark C++ calls
  cpp_call_time <- system.time({
    for (i in 1:n_calls) {
      cpp_result <- cpp_cox$coxdev(eta, weights)
    }
  })["elapsed"] / n_calls

  # Benchmark R loop calls
  r_call_time <- system.time({
    for (i in 1:n_calls) {
      r_result <- stratified_r_loop(event, status, strata, eta, weights, start)
    }
  })["elapsed"] / n_calls

  # Verify results match
  stopifnot(abs(cpp_result$deviance - r_result$deviance) < 1e-10)

  list(
    n = n,
    n_strata = n_strata,
    with_ties = with_ties,
    with_start = with_start,
    cpp_init_ms = cpp_init_time * 1000,
    cpp_call_ms = cpp_call_time * 1000,
    r_call_ms = r_call_time * 1000,
    speedup = r_call_time / cpp_call_time
  )
}

run_benchmarks <- function() {
  cat(strrep("=", 80), "\n")
  cat("BENCHMARK: C++ Stratified vs R Loop Implementation\n")
  cat(strrep("=", 80), "\n\n")

  results <- list()

  # Varying number of observations with fixed strata
  cat("### Varying n (observations), fixed 10 strata ###\n")
  cat(sprintf("%8s | %12s | %12s | %8s\n", "n", "R loop (ms)", "C++ (ms)", "Speedup"))
  cat(strrep("-", 50), "\n")
  for (n in c(100, 500, 1000, 5000, 10000)) {
    r <- benchmark_single(n, n_strata = 10, n_calls = 30)
    results <- c(results, list(r))
    cat(sprintf("%8d | %12.4f | %12.4f | %7.2fx\n",
                n, r$r_call_ms, r$cpp_call_ms, r$speedup))
  }

  # Varying number of strata with fixed observations
  cat("\n### Varying strata count, fixed n=1000 ###\n")
  cat(sprintf("%8s | %12s | %12s | %8s\n", "strata", "R loop (ms)", "C++ (ms)", "Speedup"))
  cat(strrep("-", 50), "\n")
  for (n_strata in c(2, 5, 10, 50, 100, 200)) {
    r <- benchmark_single(n = 1000, n_strata = n_strata, n_calls = 30)
    results <- c(results, list(r))
    cat(sprintf("%8d | %12.4f | %12.4f | %7.2fx\n",
                n_strata, r$r_call_ms, r$cpp_call_ms, r$speedup))
  }

  # Many small strata (worst case for R loop overhead)
  cat("\n### Many small strata (5 obs each) ###\n")
  cat(sprintf("%8s | %12s | %12s | %8s\n", "strata", "R loop (ms)", "C++ (ms)", "Speedup"))
  cat(strrep("-", 50), "\n")
  for (n_strata in c(10, 50, 100, 200, 500)) {
    n <- n_strata * 5
    r <- benchmark_single(n = n, n_strata = n_strata, n_calls = 30)
    results <- c(results, list(r))
    cat(sprintf("%8d | %12.4f | %12.4f | %7.2fx\n",
                n_strata, r$r_call_ms, r$cpp_call_ms, r$speedup))
  }

  # With ties (Efron)
  cat("\n### With many ties (Efron), n=1000, 10 strata ###\n")
  r <- benchmark_single(n = 1000, n_strata = 10, n_calls = 30, with_ties = TRUE)
  results <- c(results, list(r))
  cat(sprintf("R loop: %.4f ms | C++: %.4f ms | Speedup: %.2fx\n",
              r$r_call_ms, r$cpp_call_ms, r$speedup))

  # With start times (left truncation)
  cat("\n### With start times (left truncation), n=1000, 10 strata ###\n")
  r <- benchmark_single(n = 1000, n_strata = 10, n_calls = 30, with_start = TRUE)
  results <- c(results, list(r))
  cat(sprintf("R loop: %.4f ms | C++: %.4f ms | Speedup: %.2fx\n",
              r$r_call_ms, r$cpp_call_ms, r$speedup))

  # Large scale test
  cat("\n### Large scale: n=50000, 500 strata ###\n")
  r <- benchmark_single(n = 50000, n_strata = 500, n_calls = 10)
  results <- c(results, list(r))
  cat(sprintf("R loop: %.4f ms | C++: %.4f ms | Speedup: %.2fx\n",
              r$r_call_ms, r$cpp_call_ms, r$speedup))

  cat("\n", strrep("=", 80), "\n")
  cat("SUMMARY\n")
  cat(strrep("=", 80), "\n")

  speedups <- sapply(results, function(r) r$speedup)
  cat(sprintf("Speedup range: %.2fx - %.2fx\n", min(speedups), max(speedups)))
  cat(sprintf("Mean speedup: %.2fx\n", mean(speedups)))

  invisible(results)
}

# Run if executed directly
if (!interactive()) {
  run_benchmarks()
}
