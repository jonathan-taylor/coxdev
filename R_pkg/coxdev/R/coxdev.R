## #' @export
## preprocess <- function(start, event, status) {
##   # Convert inputs to vectors
##   start <- as.numeric(start)
##   event <- as.numeric(event)
##   status <- as.numeric(status)
##   nevent <- length(status)

##   # Perform stacking of arrays
##   stacked_time <- c(start, event)
##   stacked_status_c <- c(rep(1, nevent), 1 - status) # complement of status
##   stacked_is_start <- c(rep(1, nevent), rep(0, nevent))
##   stacked_index <- c(seq_len(nevent), seq_len(nevent))

##   # Perform the joint sort
##   order_indices <- order(stacked_time, stacked_status_c, stacked_is_start)
##   sorted_time <- stacked_time[order_indices]
##   sorted_status <- 1 - stacked_status_c[order_indices]
##   sorted_is_start <- stacked_is_start[order_indices]
##   sorted_index <- stacked_index[order_indices]

##   # Initialize variables for loop
##   event_count <- 0
##   start_count <- 0
##   event_order <- numeric(0)
##   start_order <- numeric(0)
##   start_map <- numeric(0)
##   event_map <- numeric(0)
##   first <- numeric(0)
##   which_event <- -1
##   first_event <- -1
##   num_successive_event <- 1
##   last_row <- NULL

##   # Loop through sorted data
##   for (i in seq_along(sorted_time)) {
##     s_time <- sorted_time[i]
##     s_status <- sorted_status[i]
##     s_is_start <- sorted_is_start[i]
##     s_index <- sorted_index[i]

##     if (s_is_start == 1) {
##       start_order <- c(start_order, s_index)
##       start_map <- c(start_map, event_count)
##       start_count <- start_count + 1
##     } else {
##       if (s_status == 1) {
##         if (!is.null(last_row) && s_time != last_row[1]) {
##           first_event <- first_event + num_successive_event
##           num_successive_event <- 1
##           which_event <- which_event + 1
##         } else {
##           num_successive_event <- num_successive_event + 1
##         }
##         first <- c(first, first_event)
##       } else {
##         first_event <- first_event + num_successive_event
##         num_successive_event <- 1
##         first <- c(first, first_event)
##       }

##       event_map <- c(event_map, start_count)
##       event_order <- c(event_order, s_index)
##       event_count <- event_count + 1
##     }
##     last_row <- c(s_time, s_status, s_is_start, s_index)
##   }

##   # Reset start_map to original order and set to event order
##   start_map_cp <- start_map
##   start_map[start_order] <- start_map_cp

##   s_status <- status[event_order]
##   s_first <- first
##   s_start_map <- start_map[event_order]
##   s_event_map <- event_map

##   s_event <- event[event_order]
##   s_start <- event[start_order]

##   # Compute `last`
##   last <- numeric(0)
##   last_event <- nevent - 1
##   s_first_len <- length(s_first)
##   for (i in seq_along(s_first)) {
##     f <- s_first[s_first_len - i + 1]
##     last <- c(last, last_event)
##     if (f - (nevent - i) == 0) {
##       last_event <- f - 1
##     }
##   }
##   s_last <- rev(last)

##   den <- s_last + 1 - s_first
##   s_scaling <- (seq_len(nevent) - 1 - s_first) / den

##   # Prepare the output list
##   preproc <- list(
##     start = s_start,
##     event = s_event,
##     first = s_first,
##     last = s_last,
##     scaling = s_scaling,
##     start_map = s_start_map,
##     event_map = s_event_map,
##     status = s_status
##   )

##   return(list(preproc, event_order, start_order))
## }

#' Make cox deviance object
#' @param event the event vector of times
#' @param start the start vector, if start/stop. Use `NA` for just
#'   right censored data
#' @param status the status vector indicating event or censoring
#' @param tie_breaking default 'efron'
#' @param weight the sample weight the vector of sample weights,
#'   default all ones
#' @return a list of two functions named `coxdev` and `information`
#'   each of which takes a linear predictor as argument, along with
#'   weights
#' @examples
#' set.seed(10101)
#' nobs <- 100; nvars <- 10
#' nzc <- nvars %/% 3
#' x <- matrix(rnorm(nobs * nvars), nobs, nvars)
#' beta <- rnorm(nzc)
#' fx <- x[, seq(nzc)] %*% beta / 3
#' hx <- exp(fx)
#' ty <- rexp(nobs,hx)
#' tcens <- rbinom(n = nobs, prob = 0.3, size = 1)
#' cox_deviance <- make_cox_deviance(event = ty,
#'                                   status = tcens,
#'                                   weight = rep(1.0, length(ty)),
#'                                   tie_breaking = 'efron')
#' result  <- cox_deviance$coxdev(linear_predictor = fx)
#' str(result)
#' tx  <- t(x)
#' h <- cox_deviance$information(fx)
#' I <- tx %*% h(x)  ## I should be symmetric
#' cov  <- solve(I)
#' @export
make_cox_deviance <- function(event,
                              start = NA,
                              status,
                              tie_breaking = c('efron', 'breslow'),
                              weight = rep(1.0, length(event))) {
  # Unified implementation: use stratified code with single stratum
  # This reduces code duplication and ensures consistent behavior
  tie_breaking <- match.arg(tie_breaking)
  n <- length(event)

  # Create single-stratum vector (all observations in stratum 1)
  strata <- rep(1L, n)

  # Call stratified implementation
  result <- make_stratified_cox_deviance(
    event = event,
    start = start,
    status = status,
    strata = strata,
    tie_breaking = tie_breaking
  )

  # Return only coxdev and information for backward compatibility
  # (stratified version also returns n_strata and n_total)
  list(coxdev = result$coxdev, information = result$information)
}

#' Make stratified cox deviance object (C++ implementation)
#'
#' Creates a stratified Cox deviance calculator using pure C++ implementation
#' for all strata processing. This is faster than the R-loop version when
#' there are many strata.
#'
#' @param event the event vector of times
#' @param start the start vector, if start/stop. Use `NA` for just
#'   right censored data
#' @param status the status vector indicating event or censoring
#' @param strata the strata vector (integer or factor)
#' @param tie_breaking default 'efron'
#' @return a list of two functions named `coxdev` and `information`
#'   each of which takes a linear predictor as argument, along with
#'   weights
#' @examples
#' set.seed(10101)
#' nobs <- 100
#' x <- matrix(rnorm(nobs * 5), nobs, 5)
#' beta <- c(1, -0.5, 0.3, 0, 0)
#' fx <- x %*% beta
#' hx <- exp(fx)
#' ty <- rexp(nobs, hx)
#' tcens <- rbinom(n = nobs, prob = 0.3, size = 1)
#' strata <- sample(1:3, nobs, replace = TRUE)
#' cox_strat <- make_stratified_cox_deviance(
#'   event = ty,
#'   status = tcens,
#'   strata = strata,
#'   tie_breaking = 'efron'
#' )
#' result <- cox_strat$coxdev(linear_predictor = fx)
#' str(result)
#' @export
make_stratified_cox_deviance <- function(event,
                                          start = NA,
                                          status,
                                          strata,
                                          tie_breaking = c('efron', 'breslow')) {

  tie_breaking <- match.arg(tie_breaking)
  efron <- tie_breaking == 'efron'

  event <- as.numeric(event)
  nevent <- length(event)
  status <- as.integer(status)
  strata <- as.integer(strata)

  if (length(start) != length(status)) {
    start <- rep(-Inf, nevent)
  } else {
    start <- as.numeric(start)
  }

  # Preprocess and create the XPtr
  strat_data_ptr <- .preprocess_stratified(start, event, status, strata, efron)

  # Store last eta and weights for information function
  last_eta <- NULL
  last_weights <- NULL

  coxdev <- function(linear_predictor, sample_weight = NULL) {
    if (is.null(sample_weight)) {
      sample_weight <- rep(1.0, length(linear_predictor))
    } else {
      sample_weight <- as.numeric(sample_weight)
    }

    result <- .cox_dev_stratified(strat_data_ptr,
                                   as.numeric(linear_predictor),
                                   sample_weight)

    # Store for information function
    last_eta <<- linear_predictor
    last_weights <<- sample_weight

    list(
      linear_predictor = linear_predictor,
      sample_weight = sample_weight,
      loglik_sat = result$loglik_sat,
      deviance = result$deviance,
      gradient = result$gradient,
      diag_hessian = result$diag_hessian
    )
  }

  information <- function(eta, sample_weight = NULL) {
    # Compute deviance to update buffers
    coxdev_result <- coxdev(eta, sample_weight)

    matvec <- function(arg) {
      # Handle both vector and matrix
      arg <- as.matrix(-arg)  # Negate for information matrix convention
      apply(arg, 2, function(v) {
        .hessian_matvec_stratified(strat_data_ptr,
                                    as.numeric(v),
                                    as.numeric(eta),
                                    coxdev_result$sample_weight)
      })
    }
    matvec
  }

  list(
    coxdev = coxdev,
    information = information,
    n_strata = .get_n_strata(strat_data_ptr),
    n_total = .get_n_total(strat_data_ptr)
  )
}
