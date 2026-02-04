#!/bin/bash
#PBS -N spmv_strong
#PBS -l select=2:ncpus=72:mpiprocs=72:ompthreads=1
#PBS -l walltime=00:20:00
#PBS -j oe
#PBS -o logs/spmv_strong.out

set -euo pipefail
cd "$PBS_O_WORKDIR"
mkdir -p logs results

module purge
module load gcc91
module load openmpi-3.0.0--gcc-9.1.0
module load cmake-3.15.4

# --- OpenMP pinning (anche se threads=1, ok) ---
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# --- eseguibile ---
export EXE=./bin/spmv_mpi
export OUTDIR=./results

# --- benchmark params (coerenti con D1) ---
export THREADS=1
export SCHED=static
export CHUNK=64
export REPEATS=10
export TRIALS=5

# Se su PBS non vuoi che OpenMPI provi a usare infiniband:
export OMPI_MCA_btl=^openib

# Importante: usa lo slotfile PBS (OpenMPI lo rileva spesso da solo)
# ma se serve: --hostfile $PBS_NODEFILE

export RUNNER="mpirun -np"

# ranks da testare
export PROCS_LIST="1 2 4 8 16 32 64 128"

# matrice (metti quella reale che userai nello strong scaling)
MATRIX=matrices/kron_g500-logn21.mtx

./scripts/run_strong.sh "$MATRIX"
