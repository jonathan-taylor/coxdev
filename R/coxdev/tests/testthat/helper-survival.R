# survival::basehaz() fails on some R-devel builds (October 2026): survfit()
# of a coxph fit stops in stats::delete.response() with "subscript out of
# bounds", as in survival's own CRAN check on r-devel-linux-x86_64-debian-gcc.
# Until that is fixed, the comparisons with basehaz() are skipped where it
# fails. TODO: remove, using survival::basehaz() directly, once survival's
# R-devel checks pass again.
basehaz_or_null <- function(fit) {
  tryCatch(survival::basehaz(fit, centered = FALSE), error = function(e) NULL)
}
