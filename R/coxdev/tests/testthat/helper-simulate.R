# Survival data with the kinds of ties of tests/simulate.py: ties among start
# times, among event times, and between them.

sample_times <- function(size = 1) {
  W <- stats::runif(size) + 0.5
  i <- size %/% 3 + 1
  W[i] <- W[i] + 2
  j <- seq_len(size)
  mid <- j > size %/% 3 & j <= size %/% 2
  W[mid] <- W[mid] + 1
  W
}

sample_weights <- function(size = 1) {
  W <- stats::rpois(size, 2) + stats::runif(size)
  W[size %/% 3 + 1] <- W[size %/% 3 + 1] + 2
  W
}

sample_weights_zeros <- function(size = 1) {
  W <- sample_weights(size)
  W[sample(size, size %/% 5)] <- 0
  W
}

just_ones <- function(size = 1) rep(1, size)

# (start_count, event_count): 0 no ties, 1 a single tie, 2 many ties
simulate_ties <- function(start_count, event_count, size = 5) {
  size <- stats::rpois(1, size) + 1
  key <- paste(start_count, event_count)
  if (key %in% c("0 1", "1 0")) {
    start <- sample_times(size)
    event <- start + sample_times(size)
  } else if (key == "0 2") {
    event <- rep(sample_times(), size)
    start <- event - sample_times(size)
    shift <- min(start) + sample_times()
    event <- event + shift
    start <- start + shift
  } else if (key == "2 0") {
    start <- rep(sample_times(), size)
    event <- start + sample_times(size)
  } else if (key == "1 1") {
    start <- event <- numeric(0)
    for (i in seq_len(size)) {
      U <- sample_times()
      start <- c(start, U - sample_times(), U)
      event <- c(event, U, U + sample_times())
    }
  } else if (key == "2 1") {
    start <- rep(sample_times(), size)
    event <- start + sample_times(size)
    event <- c(event, start[1])
    start <- c(start, start[1] - sample_times())
  } else if (key == "1 2") {
    event <- rep(sample_times(), size)
    start <- event - sample_times(size)
    start <- c(start, event[1])
    event <- c(event, event[1] + sample_times())
  } else if (key == "2 2") {
    U <- sample_times()
    event <- rep(U, size)
    start <- event - sample_times(size)
    size2 <- stats::rpois(1, size) + 1
    start2 <- rep(U, size2)
    event2 <- start2 + sample_times(size2)
    start <- c(start, start2)
    event <- c(event, event2)
  }
  data.frame(start = start, event = event,
             status = stats::rbinom(length(start), 1, 0.5))
}

dataset_types <- list(c(0, 1), c(1, 0), c(1, 1), c(0, 2), c(2, 0), c(2, 1), c(1, 2), c(2, 2))
all_combos <- unlist(lapply(seq_along(dataset_types), function(k) {
  utils::combn(length(dataset_types), k, simplify = FALSE)
}), recursive = FALSE)

# with some observations after the last failure, which carry no information
simulate_df <- function(combo, nrep = 5, size = 5) {
  dfs <- list()
  for (k in combo) {
    for (r in seq_len(nrep)) {
      dfs[[length(dfs) + 1]] <- simulate_ties(dataset_types[[k]][1], dataset_types[[k]][2], size)
    }
  }
  df <- do.call(rbind, dfs)
  start <- max(df$event) + stats::rexp(5)
  rbind(df, data.frame(start = start, event = start + stats::rexp(5), status = 0))
}

rel_diff <- function(a, b) sqrt(sum((a - b)^2)) / sqrt(sum(b^2))

# survival data on a coarse grid of times, as in tests/test_sample_score.py
cox_data <- function(n = 150, p = 3, start = FALSE, ties = TRUE) {
  X <- matrix(stats::rnorm(n * p), n, p)
  beta <- seq(0.5, -0.5, length.out = p)
  T <- stats::rexp(n, exp(drop(X %*% beta)))
  C <- stats::rexp(n, 1 / 1.5)
  event <- pmin(T, C)
  status <- as.integer(T <= C)
  if (ties) event <- ceiling(event * 4) / 4
  entry <- NULL
  if (start) {
    entry <- stats::runif(n, 0, 0.5) * event
    if (ties) entry <- floor(entry * 8) / 8
  }
  list(X = X, event = event, status = status, start = entry)
}

# the information of the coefficients of X
coef_information <- function(cox, X, eta, weight = NULL) {
  crossprod(X, cox$information(eta, weight)$multiply(X))
}
