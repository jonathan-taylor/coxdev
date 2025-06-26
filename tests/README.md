# Tests Directory

This directory contains tests for the coxdev package.

## Test Data

The test files now generate data programmatically instead of loading from external files:

- `test_bad.py` and `test_bad.R` generate problematic test cases using `generate_problematic_test_data()`
- `simulate.py` provides functions to generate various types of survival data with different tie patterns
- All test data is generated with reproducible random seeds

## Test Files

- `test_compareR.py` - Tests comparing against R's coxph and glmnet implementations
- `test_cumsums.py` - Tests for cumulative sum calculations
- `test_bad.py` - Tests for problematic edge cases (Python version)
- `test_bad.R` - Tests for problematic edge cases (R version)
- `simulate.py` - Data generation utilities for testing

## Jupyter Notebooks

- `test_cox_hessian.ipynb` - Interactive testing of Cox model Hessian calculations
- `Weird_coxph.ipynb` - Investigation of unusual coxph behavior
- `cox_tiesR.ipynb` - Testing tie-breaking methods

## Cleanup

The following files are now ignored by git and should not be committed:
- `*.RDS` and `*.rds` files (R data files)
- `*.csv` files (data files)
- `*.py~` and `*.Rmd~` files (backup files)
- `.ipynb_checkpoints/` directory
- `__pycache__/` directory
- `ipython_log.py*` files
