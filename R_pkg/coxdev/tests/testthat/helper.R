# Helper functions for coxdev tests
# This file is automatically sourced before tests run

library(survival)
library(glmnet)

all_close <- function(a, b, rtol = 1e-05, atol = 1e-08) {
  all(abs(a - b) <= (atol + rtol * abs(b)))
}

rel_diff_norm <- function(a, b) {
  a <- as.matrix(a)
  b <- as.matrix(b)
  norm(a - b, 'F') / norm(b, 'F')
}

# Function to sample weights
# @param size Number of weights to generate
# @param zero_weight_fraction Fraction of weights to set to zero (0.0 to 1.0). Default 0.0.
sample_weights <- function(size = 1, zero_weight_fraction = 0.0) {
  W <- rpois(size, lambda = 2) + runif(size)
  W[(size %/% 3) + 1] <- W[(size %/% 3) + 1] + 2
  W[((size %/% 3) + 1):(size %/% 2)] <- W[((size %/% 3) + 1):(size %/% 2)] + 1

  if (zero_weight_fraction > 0) {
    n_zero <- max(1, as.integer(size * zero_weight_fraction))
    n_zero <- min(n_zero, size - 1)  # Keep at least one non-zero
    zero_idx <- sample(size, n_zero)
    W[zero_idx] <- 0.0
  }

  W
}

# Ones for weights
just_ones <- function(size) rep(1.0, size)

# Weight function factories for parametrized tests
sample_weights_with_zeros <- function(size) sample_weights(size, zero_weight_fraction = 0.2)

# Function to sample times
sample_times <- function(size = 1) {
  W <- runif(size) + 0.5
  W[(size %/% 3) + 1] <- W[(size %/% 3) + 1] + 2
  W[((size %/% 3) + 1):(size %/% 2)] <- W[((size %/% 3) + 1):(size %/% 2)] + 1
  return(W)
}

simulate <- function(start_count, event_count, size = 1) {
  size <- rpois(1, lambda = size) + 1

  if (start_count == 0 && event_count == 0) {
    return(NULL)
  }

  # Single start or event
  if ((start_count == 0 && event_count == 1) || (start_count == 1 && event_count == 0)) {
    start <- sample_times(size = size)
    event <- start + sample_times(size = size)
  }

  # Ties in event but not starts
  if (start_count == 0 && event_count == 2) {
    event <- rep(sample_times(), size)
    start <- event - sample_times(size = size)
    min_start <- min(start)
    E <- sample_times()
    event <- event + min_start + E
    start <- start + min_start + E
  }

  # Ties in starts but not events
  if (start_count == 2 && event_count == 0) {
    start <- rep(sample_times(), size)
    event <- start + sample_times(size = size)
  }

  # Single tie in start and event
  if (start_count == 1 && event_count == 1) {
    start <- c()
    event <- c()
    for (i in 1:size) {
      U <- sample_times()
      start <- c(start, U - sample_times(), U)
      event <- c(event, U, U + sample_times())
    }
  }

  # Multiple starts at single event
  if (start_count == 2 && event_count == 1) {
    start <- rep(sample_times(), size)
    event <- start + sample_times(size = size)
    E <- sample_times()
    event <- c(event, start[1])
    start <- c(start, start[1] - sample_times())
  }

  # Multiple events at single start
  if (start_count == 1 && event_count == 2) {
    event <- rep(sample_times(), size)
    start <- event - sample_times(size = size)
    E <- sample_times()
    start <- c(start, event[1])
    event <- c(event, event[1] + sample_times())
  }

  # Multiple events and starts
  if (start_count == 2 && event_count == 2) {
    U <- sample_times()
    event <- rep(U, size)
    start <- event - sample_times(size = size)
    size2 <- rpois(1, lambda = size) + 1
    start2 <- rep(U, size2)
    event2 <- start2 + sample_times(size = size2)
    start <- c(start, start2)
    event <- c(event, event2)
  }

  size <- length(start)
  status <- sample(c(0, 1), size, replace = TRUE)
  return(data.frame(start = start, event = event, status = status))
}

simulate_df <- function(tie_types, nrep, size, noinfo = TRUE) {
  dfs <- list()

  for (tie_type in tie_types) {
    for (i in 1:nrep) {
      dfs[[length(dfs) + 1]] <- simulate(tie_type[1], tie_type[2], size = size)
    }
  }

  df <- do.call(rbind, dfs)

  # Include some points with no failures
  if (noinfo) {
    max_event <- max(df$event, na.rm = TRUE)
    start <- max_event + rexp(5, rate = 1)
    event <- start + rexp(5, rate = 1)
    df_noinfo <- data.frame(start = start, event = event, status = rep(0, 5))
    df <- rbind(df, df_noinfo)
  }
  return(df)
}

# Define dataset types
dataset_types <- list(c(0, 1), c(1, 0), c(1, 1), c(0, 2), c(2, 0), c(2, 1), c(1, 2), c(2, 2))

all_combos <- do.call(c, lapply(seq_along(dataset_types), combn, x = dataset_types,
                                simplify = FALSE))

# Set seed for reproducibility
set.seed(0)
