#!/bin/bash
set -e

# inizializza moduli
source /etc/profile
source /etc/profile.d/modules.sh

module purge
module load gcc91
module load openmpi-3.0.0--gcc-9.1.0
module load cmake-3.15.4

export OMPI_CC=/apps/gcc-9.1.0/local/bin/gcc-9.1.0
export OMPI_CXX=/apps/gcc-9.1.0/local/bin/g++-9.1.0
export OMPI_MCA_mpi_cuda_support=0
export OMPI_MCA_oob=tcp
export OMPI_MCA_btl="^openib"
export OMPI_MCA_orte_base_help_aggregate=1

echo
echo "=== ENVIRONMENT READY ==="
echo

# replaces the old shell with a new one with the new environment
exec bash