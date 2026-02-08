#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXE="$REPO_ROOT/bin/spmv_mpi"
OUTDIR="$REPO_ROOT/results"
OUT="$OUTDIR/strong.txt"

mkdir -p "$OUTDIR"
export MATRICES_DIR="$REPO_ROOT/bin/matrices"

# ---------------- CONFIG ----------------
MATRIX="${1:?Usage: strong.sh <matrix.mtx> [threads] [sched] [chunk] [repeats] [trials]}"
THREADS="${2:-1}"
SCHED="${3:-static}"
CHUNK="${4:-64}"
REPEATS="${5:-10}"
TRIALS="${6:-5}"

P_LIST=(1 2 4 8 16 32 64 128)
# ---------------------------------------

{
  echo "==== Strong Scaling Run ===="
  echo "date: $(date)"
  echo "host: $(hostname)"
  echo "exe:  $EXE"
  echo "matrix: $MATRIX"
  echo "threads: $THREADS"
  echo "schedule: $SCHED chunk=$CHUNK"
  echo "repeats: $REPEATS trials: $TRIALS"
  echo "P list: ${P_LIST[*]}"
  echo
  printf "%-4s %-12s %-14s %-12s %-12s %-12s %-14s %-14s %-12s %-12s\n" \
    "P" "p90_e2e_ms" "p90_comp_ms" "p90_comm_ms" \
    "gflops_e2e" "gbps_e2e" "gflops_comp" "gbps_comp" \
    "commKiB_max" "memMiB_max"
} > "$OUT"

for P in "${P_LIST[@]}"; do
  echo "[run] P=$P ..." >&2

  OUTRUN=$(
    cd "$REPO_ROOT"
    mpirun -np "$P" --bind-to none "$EXE" "$MATRIX" "$THREADS" "$SCHED" "$CHUNK" "$REPEATS" "$TRIALS --no-validate"
  )

  p90_e2e=$(echo "$OUTRUN" | awk -F': ' '/P90 execution time/{print $2}' | awk '{print $1}' | tail -n1)
  p90_comp=$(echo "$OUTRUN" | awk -F': ' '/Compute-only P90 time/{print $2}' | awk '{print $1}' | tail -n1)
  p90_comm=$(echo "$OUTRUN" | awk -F': ' '/Comm-only P90 time/{print $2}' | awk '{print $1}' | tail -n1)

  gflops_e2e=$(echo "$OUTRUN" | awk -F': ' '/Throughput/{print $2}' | awk '{print $1}' | tail -n1)
  gbps_e2e=$(echo "$OUTRUN" | awk -F': ' '/Estimated bandwidth/{print $2}' | awk '{print $1}' | tail -n1)

  gflops_comp=$(echo "$OUTRUN" | awk -F': ' '/Compute-only GFLOPS/{print $2}' | awk '{print $1}' | tail -n1)
  gbps_comp=$(echo "$OUTRUN" | awk -F': ' '/Compute-only BW/{print $2}' | awk '{print $1}' | tail -n1)

  commKiB_max=$(
    echo "$OUTRUN" |
    awk '/Per-rank max \(KiB\): total=/{for(i=1;i<=NF;i++) if($i ~ /^total=/){sub(/^total=/,"",$i); print $i; exit}}'
  )
  memMiB_max=$(
    echo "$OUTRUN" |
    awk '/Per-rank max \(MiB\): total=/{for(i=1;i<=NF;i++) if($i ~ /^total=/){sub(/^total=/,"",$i); print $i; exit}}'
  )


  printf "%-4d %-12.3f %-14.3f %-12.3f %-12.3f %-12.3f %-14.3f %-14.3f %-12.3f %-12.3f\n" \
    "$P" "${p90_e2e:-0}" "${p90_comp:-0}" "${p90_comm:-0}" \
    "${gflops_e2e:-0}" "${gbps_e2e:-0}" "${gflops_comp:-0}" "${gbps_comp:-0}" \
    "${commKiB_max:-0}" "${memMiB_max:-0}" >> "$OUT"

done

echo "[done] wrote $OUT" >&2