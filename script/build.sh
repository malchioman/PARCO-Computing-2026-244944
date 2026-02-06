#!/bin/bash
set -e

command -v mpicc  >/dev/null 2>&1 || { echo "ERROR: mpicc not found. Run ./scripts/env.sh first."; exit 1; }
command -v mpic++ >/dev/null 2>&1 || { echo "ERROR: mpic++ not found. Run ./scripts/env.sh first."; exit 1; }

(
  cd "$(dirname "$0")/.."
  rm -rf build
  mkdir -p build bin
  cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpic++ \
    -DPARCO_BUILD_MPI=ON \
    -DPARCO_BUILD_OMP=ON \
    -DPARCO_MARCH_NATIVE=OFF
  cmake --build build -j
)

echo
echo "=== BUILD COMPLETED ==="
echo "Now you can run the code"
echo
