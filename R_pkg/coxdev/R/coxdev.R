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
                              start = NA, # if NA, indicates just right censored data
                              status,
                              tie_breaking = c('efron', 'breslow'),
                              weight = rep(1.0, length(event))) {

  tie_breaking  <- match.arg(tie_breaking)

  event <- as.numeric(event)
  nevent <- length(event)
  status <- as.integer(status)
  if (length(start) != length(status)) {
    start <- rep(-Inf, nevent)
    have_start_times <- FALSE
  } else {
    start  <- as.numeric(start)
    have_start_times <- TRUE
  }

  ## prep_result  <- preprocess(start, event, status) # R version of preprocess
  ## event_order  <- as.integer(prep_result[[2L]])  - 1L  ## for R 1-based indexing!
  ## start_order  <- as.integer(prep_result[[3L]])  - 1L  ## for R 1-based indexing!
  prep_result  <- .preprocess(start, event, status)  # C version of preprocess
  event_order  <- as.integer(prep_result[[2L]])
  start_order  <- as.integer(prep_result[[3L]])
  preproc  <- prep_result[[1L]]
  efron  <- (tie_breaking == 'efron') && (norm(matrix(preproc$scaling), "2") > 0)
  status <- preproc[['status']]
  event <- preproc[['event']]
  start <- preproc[['start']]
  first <- preproc[['first']]
  last <- preproc[['last']]
  scaling <- preproc[['scaling']]
  event_map <- preproc[['event_map']]
  start_map <- preproc[['start_map']]

  ## This check is now moved in to C++ code.
  ## first_start <- first[start_map + 1]  ## Only used for next check!
  ## if (!all(first_start == start_map)) {
  ##   stop('first_start disagrees with start_map')
  ## }

  n <- length(status)

  # allocate necessary memory

  T_1_term <- numeric(n)
  T_2_term <- numeric(n)
  # event_reorder_buffers = np.zeros((3, n))
  event_reorder_buffers <- lapply(seq_len(3), function(x) numeric(n))
  # forward_cumsum_buffers = np.zeros((5, n+1))
  forward_cumsum_buffers <- lapply(seq_len(5), function(x) numeric(n + 1))
  forward_scratch_buffer <- numeric(n)
  # reverse_cumsum_buffers = np.zeros((4, n+1))
  reverse_cumsum_buffers <- lapply(seq_len(4), function(x) numeric(n + 1))
  # risk_sum_buffers = np.zeros((2, n))
  risk_sum_buffers <- list(numeric(n), numeric(n))
  hess_matvec_buffer <- numeric(n)
  grad_buffer <- numeric(n)
  diag_hessian_buffer <- numeric(n)
  diag_part_buffer <- numeric(n)
  w_avg_buffer <- numeric(n)
  exp_w_buffer <- numeric(n)

  coxdev <- function (linear_predictor, sample_weight = NULL) {
    if (is.null(sample_weight)) {
      sample_weight  <- rep(1.0, length(linear_predictor))
    } else {
      sample_weight  <- as.numeric(sample_weight)
    }
    loglik_sat  <- .compute_sat_loglik(first,
                                       last,
                                       sample_weight,
                                       event_order,
                                       status,
                                       forward_cumsum_buffers[[1]])
    eta <- linear_predictor - mean(linear_predictor)
    exp_w_buffer <<- sample_weight * exp(eta) ## Note the double arrow

    ## The C++ code has to be modified for R lists!
    deviance  <- .cox_dev(eta,
                          sample_weight,
                          exp_w_buffer,
                          event_order,
                          start_order,
                          status,
                          first,
                          last,
                          scaling,
                          event_map,
                          start_map,
                          loglik_sat,
                          T_1_term,
                          T_2_term,
                          grad_buffer,
                          diag_hessian_buffer,
                          diag_part_buffer,
                          w_avg_buffer,
                          event_reorder_buffers,
                          risk_sum_buffers, #[[1]] is for coxdev, [[2]] is for hessian...
                          forward_cumsum_buffers,
                          forward_scratch_buffer,
                          reverse_cumsum_buffers, #[1:3] are for risk sums, [4:5] used for hessian risk*arg sums
                          have_start_times,
                          efron)
    list(linear_predictor = linear_predictor,
         sample_weight = sample_weight,
         loglik_sat = loglik_sat,
         deviance = deviance,
         gradient = grad_buffer,
         diag_hessian = diag_hessian_buffer)
  }
  information  <- function(eta, sample_weight = NULL) {

    coxdev_result <- coxdev(eta, sample_weight)

    event_cumsum <- reverse_cumsum_buffers[[1L]]
    start_cumsum <- reverse_cumsum_buffers[[2L]]
    risk_sums  <- risk_sum_buffers[[1L]]

    matvec <- function(arg) {
      # Have to handle both a vector or a matrix
      arg <- as.matrix(-arg)
      apply(arg, 2, .hessian_matvec,
            eta = eta,
            sample_weight = coxdev_result$sample_weight,
            risk_sums = risk_sums,
            diag_part = diag_part_buffer,
            w_avg = w_avg_buffer,
            exp_w = exp_w_buffer,
            event_cumsum = event_cumsum,
            start_cumsum = start_cumsum,
            event_order = event_order,
            start_order = start_order,
            status = status,
            first = first,
            last = last,
            scaling = scaling,
            event_map = event_map,
            start_map = start_map,
            risk_sum_buffers = risk_sum_buffers,
            forward_cumsum_buffers = forward_cumsum_buffers,
            forward_scratch_buffer = forward_scratch_buffer,
            reverse_cumsum_buffers = reverse_cumsum_buffers,
            hess_matvec_buffer = hess_matvec_buffer,
            have_start_times = have_start_times,
            efron = efron)
    }
    matvec
  }
  list(coxdev = coxdev, information = information)
}
