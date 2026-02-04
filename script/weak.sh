#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXE="$REPO_ROOT/bin/spmv_mpi"
GEN="$REPO_ROOT/bin/custom_matrix"
OUTDIR="$REPO_ROOT/results"
OUT="$OUTDIR/weak.txt"

mkdir -p "$OUTDIR" "$REPO_ROOT/bin/matrices"
export MATRICES_DIR="$REPO_ROOT/bin/matrices"

THREADS="${THREADS:-1}"
SCHED="${SCHED:-static}"
CHUNK="${CHUNK:-64}"
REPEATS="${REPEATS:-10}"
TRIALS="${TRIALS:-5}"

ROWS_PER_RANK="${ROWS_PER_RANK:-16384}"
NNZ_PER_RANK="${NNZ_PER_RANK:-1000000}"

P_LIST=(${PROCS_LIST:-1 2 4 8 16 32 64 128})

[[ -x "$EXE" ]] || { echo "[fatal] missing EXE: $EXE" >&2; exit 1; }
[[ -x "$GEN" ]] || { echo "[fatal] missing GEN: $GEN" >&2; exit 1; }

{
  echo "==== Weak Scaling Run ===="
  echo "date: $(date)"
  echo "host: $(hostname)"
  echo "exe: $EXE"
  echo "gen: $GEN"
  echo "threads: $THREADS"
  echo "schedule: $SCHED chunk=$CHUNK"
  echo "repeats: $REPEATS trials: $TRIALS"
  echo "rows_per_rank: $ROWS_PER_RANK"
  echo "nnz_per_rank:  $NNZ_PER_RANK"
  echo "P list: ${P_LIST[*]}"
  echo
  printf "%-4s %-10s %-10s %-12s %-12s %-14s %-12s %-12s %-12s %-14s %-12s\n" \
    "P" "rows" "cols" "nnz" \
    "p90_e2e_ms" "p90_compute_ms" "p90_comm_ms" \
    "gflops_e2e" "gbps_e2e" "gflops_compute" "gbps_compute"
} > "$OUT"

for P in "${P_LIST[@]}"; do
  rows=$((ROWS_PER_RANK * P))
  cols=$rows
  nnz=$((NNZ_PER_RANK * P))

  mtx="weak_P${P}_rpr${ROWS_PER_RANK}_nnzpr${NNZ_PER_RANK}.mtx"
  mtx_rel="bin/matrices/$mtx"              # <=== IMPORTANTE: path passato al generator
  mtx_path="$REPO_ROOT/$mtx_rel"

  echo "[gen] P=$P -> $mtx_path (rows=$rows cols=$cols nnz=$nnz)" >&2

  # custom_matrix: <out_path_or_name> <rows> <cols> <nnz>
  ( cd "$REPO_ROOT" && "$GEN" "$mtx_rel" "$rows" "$cols" "$nnz" >/dev/null )

  [[ -f "$mtx_path" ]] || { echo "[fatal] generator did not create: $mtx_path" >&2; exit 1; }

  echo "[run] P=$P ..." >&2

  MAP_ARGS=()
  if [[ "$P" -le 72 ]]; then
    MAP_ARGS=(--map-by "ppr:${P}:node:pe=1" --bind-to core)
  else
    MAP_ARGS=(--map-by "ppr:72:node:pe=1" --bind-to core)
  fi

  set +e
  OUTRUN=$(
    cd "$REPO_ROOT"
    mpirun -np "$P" "${MAP_ARGS[@]}" \
      "$EXE" "$mtx" "$THREADS" "$SCHED" "$CHUNK" "$REPEATS" "$TRIALS" --no-validate 2>&1
  )
  RC=$?
  set -e
  if [[ $RC -ne 0 ]]; then
    echo "[fatal] mpirun failed at P=$P (rc=$RC). Output:" >> "$OUT"
    echo "$OUTRUN" >> "$OUT"
    exit $RC
  fi

  p90_e2e=$(echo "$OUTRUN" | awk -F':' '/P90 execution time/ {gsub(/ ms/,"",$2); sub(/^[ \t]+/,"",$2); print $2}' | tail -n1)
  p90_comp=$(echo "$OUTRUN" | awk -F':' '/Compute-only P90 time/ {gsub(/ ms/,"",$2); sub(/^[ \t]+/,"",$2); print $2}' | tail -n1)
  p90_comm=$(echo "$OUTRUN" | awk -F':' '/Comm-only P90 time/ {gsub(/ ms/,"",$2); sub(/^[ \t]+/,"",$2); print $2}' | tail -n1)

  gflops_e2e=$(echo "$OUTRUN" | awk -F':' '/Throughput/ {gsub(/ GFLOPS/,"",$2); sub(/^[ \t]+/,"",$2); print $2}' | tail -n1)
  gbps_e2e=$(echo "$OUTRUN" | awk -F':' '/Estimated bandwidth/ {gsub(/ GB\/s/,"",$2); sub(/^[ \t]+/,"",$2); print $2}' | tail -n1)

  gflops_comp=$(echo "$OUTRUN" | awk -F':' '/Compute-only GFLOPS/ {gsub(/ GFLOPS/,"",$2); sub(/^[ \t]+/,"",$2); print $2}' | tail -n1)
  gbps_comp=$(echo "$OUTRUN" | awk -F':' '/Compute-only BW/ {gsub(/ GB\/s/,"",$2); sub(/^[ \t]+/,"",$2); print $2}' | tail -n1)

  printf "%-4d %-10d %-10d %-12d %-12.3f %-14.3f %-12.3f %-12.3f %-12.3f %-14.3f %-12.3f\n" \
    "$P" "$rows" "$cols" "$nnz" \
    "${p90_e2e:-0}" "${p90_comp:-0}" "${p90_comm:-0}" \
    "${gflops_e2e:-0}" "${gbps_e2e:-0}" "${gflops_comp:-0}" "${gbps_comp:-0}" >> "$OUT"
done

echo "[done] wrote $OUT" >&2