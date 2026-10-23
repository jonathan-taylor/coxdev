#!/bin/sh
# Copy the shared C++ library (../../cpp) into the R package, so that the
# package is self-contained, as R CMD build and CRAN require. Run from
# anywhere after editing anything under cpp/; CI checks the copies agree.
set -e
cd "$(dirname "$0")/.."
cp ../../cpp/include/coxdev.h inst/include/coxdev.h
cp ../../cpp/include/coxdev_impl.h inst/include/coxdev_impl.h
cp ../../cpp/src/coxdev.cpp src/coxdev.cpp
