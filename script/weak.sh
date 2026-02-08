#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXE="$REPO_ROOT/bin/spmv_mpi"
GEN="$REPO_ROOT/bin/custom_matrix"
OUTDIR="$REPO_ROOT/results"
OUT="$OUTDIR/weak.txt"

mkdir -p "$OUTDIR" "$REPO_ROOT/bin/matrices"
export MATRICES_DIR="$REPO_ROOT/bin/matrices"

# ---------------- CONFIG ----------------
THREADS=1
SCHED=static
CHUNK=64
REPEATS=10
TRIALS=5

ROWS_PER_RANK="${ROWS_PER_RANK:-16384}"
NNZ_PER_RANK="${NNZ_PER_RANK:-1000000}"

P_LIST=(1 2 4 8 16 32 64 128)
# ---------------------------------------

{
  echo "==== Weak Scaling Run ===="
  echo "date: $(date)"
  echo "host: $(hostname)"
  echo "exe:  $EXE"
  echo "gen:  $GEN"
  echo "threads: $THREADS"
  echo "schedule: $SCHED chunk=$CHUNK"
  echo "repeats: $REPEATS trials: $TRIALS"
  echo "rows_per_rank: $ROWS_PER_RANK"
  echo "nnz_per_rank:  $NNZ_PER_RANK"
  echo "P list: ${P_LIST[*]}"
  echo
 printf "%-6s %-10s %-10s %-12s %-12s %-14s %-12s %-12s %-12s %-14s %-14s %-12s\n" \
   "P" "rows" "cols" "nnz" \
   "p90_e2e_ms" "p90_comp_ms" "p90_comm_ms" \
   "gflops_e2e" "gflops_comp" "commKiB_max" "commKiB_avg" "memMiB_max"
} > "$OUT"

for P in "${P_LIST[@]}"; do
  rows=$((ROWS_PER_RANK * P))
  cols=$rows
  nnz=$((NNZ_PER_RANK * P))

  mtx="weak_P${P}_rpr${ROWS_PER_RANK}_nnzpr${NNZ_PER_RANK}.mtx"
  mtx_path="$REPO_ROOT/bin/matrices/$mtx"

  echo "[gen] P=$P -> $mtx" >&2
  if [[ ! -f "$mtx_path" ]]; then
    "$GEN" "$mtx_path" "$rows" "$cols" "$nnz" >/dev/null
  fi


  echo "[run] P=$P" >&2

  MPI_EXTRA=()
  if [[ -n "${PBS_NODEFILE:-}" ]]; then
    MPI_EXTRA+=( --hostfile "$PBS_NODEFILE" --bind-to core --map-by slot )
  else
    MPI_EXTRA+=( --bind-to none --map-by slot --oversubscribe )
  fi

  OUTRUN=$(
    mpirun -np "$P" \
      "${MPI_EXTRA[@]}" \
      "$EXE" "$mtx" "$THREADS" "$SCHED" "$CHUNK" "$REPEATS" "$TRIALS" --no-validate
  )

  p90_e2e=$(echo "$OUTRUN" | awk -F': ' '/P90 execution time/{print $2}' | awk '{print $1}')
  p90_comp=$(echo "$OUTRUN" | awk -F': ' '/Compute-only P90 time/{print $2}' | awk '{print $1}')
  p90_comm=$(echo "$OUTRUN" | awk -F': ' '/Comm-only P90 time/{print $2}' | awk '{print $1}')
  gflops_e2e=$(echo "$OUTRUN" | awk -F': ' '/Throughput/{print $2}' | awk '{print $1}')
  gflops_comp=$(echo "$OUTRUN" | awk -F': ' '/Compute-only GFLOPS/{print $2}' | awk '{print $1}')

  commKiB_max=$(
    echo "$OUTRUN" |
    awk '/Per-rank max \(KiB\): total=/{for(i=1;i<=NF;i++) if($i ~ /^total=/){sub(/^total=/,"",$i); print $i; exit}}'
  )
  commKiB_avg=$(
    echo "$OUTRUN" |
    awk '/Per-rank max \(KiB\): total=/{for(i=1;i<=NF;i++) if($i ~ /^avg=/){sub(/^avg=/,"",$i); print $i; exit}}'
  )
  memMiB_max=$(
    echo "$OUTRUN" |
    awk '/Per-rank max \(MiB\): total=/{for(i=1;i<=NF;i++) if($i ~ /^total=/){sub(/^total=/,"",$i); print $i; exit}}'
  )


  printf "%-6d %-10d %-10d %-12d %-12.3f %-14.3f %-12.3f %-12.3f %-12.3f %-14.3f %-14.3f %-12.3f\n" \
    "$P" "$rows" "$cols" "$nnz" \
    "${p90_e2e:-0}" "${p90_comp:-0}" "${p90_comm:-0}" \
    "${gflops_e2e:-0}" "${gflops_comp:-0}" "${commKiB_max:-0}" "${commKiB_avg:-0}" "${memMiB_max:-0}" >> "$OUT"
done

echo "[done] wrote $OUT" >&2
