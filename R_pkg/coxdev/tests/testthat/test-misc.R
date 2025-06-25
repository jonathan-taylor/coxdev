## py_run_file(file = 'preprocess.py')
## data <- simulate_df(all_combos[[length(all_combos)]],
##                     nrep = 5,
##                     size = 5)
## p  <- py$py_preprocess(as.numeric(data$start), as.numeric(data$event), as.numeric(data$status))

## test_simple_coxph <- function(nrep=5,
##                               size=5,
##                               tol=1e-10) {
##   test_coxph(all_combos[[length(all_combos)]],
##              'efron',
##              sample_weights,
##              TRUE,
##              nrep=5,
##              size=5,
##              tol=1e-10)
## }
## test_simple_glmnet <- function(nrep=5,
##                                size=5,
##                                tol=1e-10) {
##   test_glmnet(all_combos[[length(all_combos)]],
##               sample_weights,
##               TRUE,
##               nrep=5,
##               size=5,
##               tol=1e-10)
## }



