#!/usr/bin/env bash
set -euo pipefail

OUTDIR=${OUTDIR:-results}
mkdir -p "$OUTDIR"
OUT="$OUTDIR/cluster_info_$(date +%Y%m%d_%H%M%S).txt"

{
  echo "==== CLUSTER INFO ===="
  echo "date: $(date)"
  echo "hostname: $(hostname)"
  echo

  echo "---- OS / kernel ----"
  uname -a || true
  cat /etc/os-release 2>/dev/null || true
  echo

  echo "---- CPU ----"
  lscpu || true
  echo

  echo "---- NUMA ----"
  numactl --hardware 2>/dev/null || true
  echo

  echo "---- MEMORY ----"
  free -h || true
  echo

  echo "---- LIMITS ----"
  ulimit -a || true
  echo

  echo "---- MODULES ----"
  (module --version 2>/dev/null || true)
  (module list 2>&1 || true)
  echo

  echo "---- COMPILER ----"
  (which gcc && gcc --version) || true
  (which g++ && g++ --version) || true
  echo

  echo "---- MPI ----"
  (which mpirun && mpirun --version) || true
  (which mpicc && mpicc --version) || true
  (ompi_info --version 2>/dev/null || true)
  echo

  echo "---- ENV (selected) ----"
  env | egrep '^(OMP|OMPI|I_MPI|MPICH|SLURM|PBS|PMI|UCX|FI_)' | sort || true
  echo
} | tee "$OUT"

echo "Saved: $OUT"
