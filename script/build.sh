#!/bin/bash
set -e

module purge
module load gcc91
module load openmpi-3.0.0--gcc-9.1.0
module load cmake-3.15.4

export OMPI_CXX=/apps/gcc-9.1.0/local/bin/g++-9.1.0
export OMPI_CC=/apps/gcc-9.1.0/local/bin/gcc-9.1.0

cd "$(dirname "$0")/.."

rm -rf build bin
mkdir -p build

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpic++ \
  -DPARCO_BUILD_MPI=ON \
  -DPARCO_BUILD_OMP=ON \
  -DPARCO_MARCH_NATIVE=OFF

cmake --build build -j
