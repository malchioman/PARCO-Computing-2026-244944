#!/bin/bash
set -euo pipefail

# ===========================
# Init Environment Modules (bash-safe)
# ===========================
if [ -f /etc/profile ]; then
  source /etc/profile
fi

if [ -f /usr/share/Modules/init/bash ]; then
  source /usr/share/Modules/init/bash
elif [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
else
  echo "[fatal] Cannot initialize environment modules (no init script found)" >&2
  exit 1
fi

# ===========================
# Modules
# ===========================
module purge
module load gcc91
module load openmpi-3.0.0--gcc-9.1.0
module load cmake-3.15.4

export OMPI_CC=/apps/gcc-9.1.0/local/bin/gcc-9.1.0
export OMPI_CXX=/apps/gcc-9.1.0/local/bin/g++-9.1.0

# repo root (script sta in script/)
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_DIR"
rm -rf build
mkdir -p build bin matrices

cmake -S . -B build -Wno-dev \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpic++ \
  -DPARCO_BUILD_MPI=ON \
  -DPARCO_BUILD_OMP=ON \
  -DPARCO_MARCH_NATIVE=OFF \
  -DPARCO_FETCH_DATASETS=ON

cmake --build build -j

# prova a scaricare dataset (se il nodo non ha internet non deve rompere la build)
cmake --build build --target dataset_kron_g500_logn21 || true

echo
echo "=== BUILD COMPLETATA ==="
echo "Binari in: $REPO_DIR/bin"
echo "Matrices dir: $REPO_DIR/matrices"
echo "Se il download e' riuscito: $REPO_DIR/matrices/kron_g500-logn21.mtx"
echo "Ora puoi usare mpirun direttamente."
echo
