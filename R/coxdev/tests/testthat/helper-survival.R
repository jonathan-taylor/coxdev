# survival's model.matrix() of a fitted coxph model fails on some R-devel
# builds (October 2026): stats::delete.response() stops with "subscript out
# of bounds", as in survival's own CRAN check on
# r-devel-linux-x86_64-debian-gcc. Every survival function that rebuilds the
# model matrix (basehaz, residuals, coxph.detail, predict, ...) fails with it,
# so the tests that compare with them skip where it does. TODO: remove, with
# the skip_if_survival_broken() calls, once survival's R-devel checks pass.
survival_works <- local({
  works <- NULL
  function() {
    if (is.null(works)) {
      d <- data.frame(time = c(1, 2, 3, 4), status = c(1, 0, 1, 1), x = c(0.1, 0.4, -0.2, 0.3))
      fit <- survival::coxph(survival::Surv(time, status) ~ x, data = d)
      probes <- list(function() stats::model.matrix(fit),
                     function() survival::basehaz(fit, centered = FALSE),
                     function() stats::residuals(fit, type = "score"))
      works <<- all(vapply(probes, function(f) !inherits(try(f(), silent = TRUE), "try-error"),
                           logical(1)))
    }
    works
  }
})

skip_if_survival_broken <- function() {
  testthat::skip_if_not(survival_works(),
                        "survival's model.matrix() of a coxph fit fails on this R (R-devel, October 2026)")
}
