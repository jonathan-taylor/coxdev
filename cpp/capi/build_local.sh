#!/bin/sh
# Build libcoxdev_c and the C test with CMake. Usage: build_local.sh [build-dir] [extra cmake args...]
# e.g. build_local.sh build-asan -DCMAKE_C_FLAGS=-fsanitize=address -DCMAKE_CXX_FLAGS=-fsanitize=address
set -e
here=$(cd "$(dirname "$0")" && pwd)
build=${1:-$here/build}
[ $# -gt 0 ] && shift
cmake -S "$here" -B "$build" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$build" -j
echo "built $(ls "$build"/libcoxdev_c.*)"
