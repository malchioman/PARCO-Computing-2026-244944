#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXE="$REPO_ROOT/bin/spmv_mpi"
OUT="$REPO_ROOT/results/strong.txt"

export MATRICES_DIR="$REPO_ROOT/bin/matrices"

THREADS=1
SCHED=static
CHUNK=64
REPEATS=10
TRIALS=5

MTX="strong_matrix.mtx"
P_LIST=(1 2 4 8 16 32 64 128)

{
  echo "==== Strong Scaling Run ===="
  echo "date: $(date)"
  echo "host: $(hostname)"
  echo "matrix: $MTX"
  echo "threads: $THREADS"
  echo "schedule: $SCHED chunk=$CHUNK"
  echo "repeats: $REPEATS trials: $TRIALS"
  echo
  printf "%-6s %-12s %-14s %-12s %-12s %-12s %-12s\n" \
    "P" "p90_e2e_ms" "p90_comp_ms" "p90_comm_ms" \
    "gflops_e2e" "gflops_comp" "speedup"
} > "$OUT"

T1=""

for P in "${P_LIST[@]}"; do
  OUTRUN=$(
    mpirun -np "$P" \
      --bind-to none \
      --map-by slot \
      --oversubscribe \
      "$EXE" "$MTX" "$THREADS" "$SCHED" "$CHUNK" "$REPEATS" "$TRIALS" --no-validate
  )

  p90=$(echo "$OUTRUN" | awk -F': ' '/P90 execution time/{print $2}' | awk '{print $1}')
  p90c=$(echo "$OUTRUN" | awk -F': ' '/Compute-only P90 time/{print $2}' | awk '{print $1}')
  p90m=$(echo "$OUTRUN" | awk -F': ' '/Comm-only P90 time/{print $2}' | awk '{print $1}')
  gfe=$(echo "$OUTRUN" | awk -F': ' '/Throughput/{print $2}' | awk '{print $1}')
  gfc=$(echo "$OUTRUN" | awk -F': ' '/Compute-only GFLOPS/{print $2}' | awk '{print $1}')

  if [[ -z "$T1" ]]; then T1="$p90"; fi
  speedup=$(awk "BEGIN {printf \"%.3f\", $T1/$p90}")

  printf "%-6d %-12.3f %-14.3f %-12.3f %-12.3f %-12.3f %-12.3f\n" \
    "$P" "$p90" "$p90c" "$p90m" "$gfe" "$gfc" "$speedup" >> "$OUT"
done
