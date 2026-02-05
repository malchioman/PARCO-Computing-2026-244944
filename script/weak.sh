#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXE="$REPO_ROOT/bin/spmv_mpi"
GEN="$REPO_ROOT/bin/custom_matrix"
OUTDIR="$REPO_ROOT/results"
OUT="$OUTDIR/weak.txt"

mkdir -p "$OUTDIR"
export MATRICES_DIR="$REPO_ROOT/bin/matrices"
mkdir -p "$MATRICES_DIR"

# ---------------- CONFIG ----------------
THREADS="${1:-1}"
SCHED="${2:-static}"
CHUNK="${3:-64}"
REPEATS="${4:-10}"
TRIALS="${5:-5}"

ROWS_PER_RANK="${6:-16384}"
NNZ_PER_RANK="${7:-1000000}"

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
  printf "%-4s %-10s %-10s %-12s %-12s %-14s %-12s %-12s %-12s %-14s %-14s\n" \
    "P" "rows" "cols" "nnz" \
    "p90_e2e_ms" "p90_comp_ms" "p90_comm_ms" \
    "gflops_e2e" "gbps_e2e" "gflops_comp" "gbps_comp"
} > "$OUT"

for P in "${P_LIST[@]}"; do
  rows=$((ROWS_PER_RANK * P))
  cols=$rows
  nnz=$((NNZ_PER_RANK * P))

  mtx="weak_P${P}_rpr${ROWS_PER_RANK}_nnzpr${NNZ_PER_RANK}.mtx"
  mtx_path="$MATRICES_DIR/$mtx"

  echo "[gen] P=$P -> $mtx (rows=$rows cols=$cols nnz=$nnz)" >&2

  # Il TUO custom_matrix: ./custom_matrix <out_path_or_name> <rows> <cols> <nnz>
  ( cd "$REPO_ROOT" && "$GEN" "$mtx_path" "$rows" "$cols" "$nnz" >/dev/null )

  if [[ ! -f "$mtx_path" ]]; then
    echo "[fatal] generator did not create: $mtx_path" >&2
    exit 1
  fi

  echo "[run] P=$P ..." >&2
  OUTRUN=$(
    cd "$REPO_ROOT"
    mpirun -np "$P" --bind-to none "$EXE" "$mtx" "$THREADS" "$SCHED" "$CHUNK" "$REPEATS" "$TRIALS"
  )

  p90_e2e=$(echo "$OUTRUN" | awk -F': ' '/P90 execution time/{print $2}' | awk '{print $1}' | tail -n1)
  p90_comp=$(echo "$OUTRUN" | awk -F': ' '/Compute-only P90 time/{print $2}' | awk '{print $1}' | tail -n1)
  p90_comm=$(echo "$OUTRUN" | awk -F': ' '/Comm-only P90 time/{print $2}' | awk '{print $1}' | tail -n1)

  gflops_e2e=$(echo "$OUTRUN" | awk -F': ' '/Throughput/{print $2}' | awk '{print $1}' | tail -n1)
  gbps_e2e=$(echo "$OUTRUN" | awk -F': ' '/Estimated bandwidth/{print $2}' | awk '{print $1}' | tail -n1)

  gflops_comp=$(echo "$OUTRUN" | awk -F': ' '/Compute-only GFLOPS/{print $2}' | awk '{print $1}' | tail -n1)
  gbps_comp=$(echo "$OUTRUN" | awk -F': ' '/Compute-only BW/{print $2}' | awk '{print $1}' | tail -n1)

  printf "%-4d %-10d %-10d %-12d %-12.3f %-14.3f %-12.3f %-12.3f %-12.3f %-14.3f %-14.3f\n" \
    "$P" "$rows" "$cols" "$nnz" \
    "${p90_e2e:-0}" "${p90_comp:-0}" "${p90_comm:-0}" \
    "${gflops_e2e:-0}" "${gbps_e2e:-0}" "${gflops_comp:-0}" "${gbps_comp:-0}" >> "$OUT"
done

echo "[done] wrote $OUT" >&2