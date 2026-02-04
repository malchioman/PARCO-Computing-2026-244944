#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXE="$REPO_ROOT/bin/spmv_mpi"
GEN="$REPO_ROOT/bin/custom_matrix"
OUTDIR="$REPO_ROOT/results"
OUT="$OUTDIR/weak.txt"

mkdir -p "$OUTDIR"
mkdir -p "$REPO_ROOT/bin/matrices"
export MATRICES_DIR="$REPO_ROOT/bin/matrices"

# ---------------- CONFIG ----------------
# Args override env, env override defaults
THREADS="${1:-${THREADS:-1}}"
SCHED="${2:-${SCHED:-static}}"
CHUNK="${3:-${CHUNK:-64}}"
REPEATS="${4:-${REPEATS:-10}}"
TRIALS="${5:-${TRIALS:-5}}"

ROWS_PER_RANK="${6:-${ROWS_PER_RANK:-16384}}"
NNZ_PER_RANK="${7:-${NNZ_PER_RANK:-1000000}}"

P_LIST=(${PROCS_LIST:-1 2 4 8 16 32 64 128})
# ---------------------------------------

if [[ ! -x "$EXE" ]]; then
  echo "[fatal] executable not found: $EXE" >&2
  exit 1
fi
if [[ ! -x "$GEN" ]]; then
  echo "[fatal] generator not found: $GEN" >&2
  exit 1
fi

# Header
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
  printf "%-4s  %-8s %-8s %-10s  %-11s %-13s %-11s  %-11s %-11s %-13s %-11s  %-12s %-12s\n" \
    "P" "rows" "cols" "nnz" \
    "p90_e2e_ms" "p90_compute_ms" "p90_comm_ms" \
    "gflops_e2e" "gbps_e2e" "gflops_compute" "gbps_compute" \
    "commKiB_max" "memMiB_max"
} > "$OUT"

for P in "${P_LIST[@]}"; do
  rows=$((ROWS_PER_RANK * P))
  cols=$rows
  nnz=$((NNZ_PER_RANK * P))

  mtx="weak_P${P}_rpr${ROWS_PER_RANK}_nnzpr${NNZ_PER_RANK}.mtx"
  mtx_path="$REPO_ROOT/bin/matrices/$mtx"

  echo "[gen] P=$P -> $mtx_path (rows=$rows cols=$cols nnz=$nnz)" >&2

  # Prova firma: custom_matrix rows cols nnz filename
  set +e
  ( cd "$REPO_ROOT" && "$GEN" "$rows" "$cols" "$nnz" "$mtx" >/dev/null 2>&1 )
  rc=$?
  set -e

  # Fallback: se non supporta args, prova solo filename (oppure genera default e rinomina)
  if [[ $rc -ne 0 ]]; then
    echo "[gen] generator did not accept (rows cols nnz filename), trying fallback..." >&2

    set +e
    ( cd "$REPO_ROOT" && "$GEN" "$mtx" >/dev/null 2>&1 )
    rc2=$?
    set -e

    if [[ $rc2 -ne 0 ]]; then
      echo "[fatal] custom_matrix failed (both signatures). Update custom_matrix to accept args." >&2
      exit 1
    fi
  fi

  if [[ ! -f "$mtx_path" ]]; then
    echo "[fatal] generator did not create: $mtx_path" >&2
    exit 1
  fi

  echo "[run] P=$P ..." >&2

  # Mapping come strong: fino a 72 su 1 nodo, oltre usa entrambi
  MAP_ARGS=()
  if [[ "$P" -le 72 ]]; then
    MAP_ARGS=(--map-by "ppr:${P}:node:pe=1" --bind-to core)
  else
    MAP_ARGS=(--map-by "ppr:72:node:pe=1" --bind-to core)
  fi

  set +e
  OUTRUN=$(cd "$REPO_ROOT" && mpirun -np "$P" "${MAP_ARGS[@]}" "$EXE" "$mtx" "$THREADS" "$SCHED" "$CHUNK" "$REPEATS" "$TRIALS" --no-validate 2>&1)
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

  commKiB_max=$(
    echo "$OUTRUN" | awk '
      /Per-rank max \(KiB\)/{
        for(i=1;i<=NF;i++){
          if($i ~ /^total=/){gsub("total=","",$i); print $i}
        }
      }' | tail -n1
  )

  memMiB_max=$(
    echo "$OUTRUN" | awk '
      /Per-rank max \(MiB\)/{
        for(i=1;i<=NF;i++){
          if($i ~ /^total=/){gsub("total=","",$i); print $i}
        }
      }' | tail -n1
  )

  # Se qualcosa manca, fallisci mostrando l'output
  if [[ -z "${p90_e2e:-}" || -z "${gflops_e2e:-}" || -z "${gbps_e2e:-}" ]]; then
    echo "[fatal] parsing failed at P=$P. Full program output:" >> "$OUT"
    echo "$OUTRUN" >> "$OUT"
    exit 2
  fi

  printf "%-4d  %-8d %-8d %-10d  %-11.3f %-13.3f %-11.3f  %-11.3f %-11.3f %-13.3f %-11.3f  %-12.3f %-12.3f\n" \
    "$P" "$rows" "$cols" "$nnz" \
    "${p90_e2e}" "${p90_comp:-0}" "${p90_comm:-0}" \
    "${gflops_e2e}" "${gbps_e2e}" "${gflops_comp:-0}" "${gbps_comp:-0}" \
    "${commKiB_max:-0}" "${memMiB_max:-0}" >> "$OUT"
done

echo "[done] wrote $OUT" >&2