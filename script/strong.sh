#!/usr/bin/env bash
set -euo pipefail

# ------------------ Paths ------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXE="${EXE:-"$REPO_ROOT/bin/spmv_mpi"}"
MATRIX="${1:-"bin/matrices/kron_g500-logn21.mtx"}"

OUTDIR="$REPO_ROOT/results"
OUTFILE="$OUTDIR/strong.txt"
mkdir -p "$OUTDIR"

# Make spmv_mpi find matrices even if you pass only filename
export MATRICES_DIR="${MATRICES_DIR:-"$REPO_ROOT/bin/matrices"}"

# ------------------ Config ------------------
THREADS="${THREADS:-1}"
SCHED="${SCHED:-static}"
CHUNK="${CHUNK:-64}"
REPEATS="${REPEATS:-10}"
TRIALS="${TRIALS:-5}"

# P list (override with PROCS_LIST env if you want)
PROCS_LIST=(${PROCS_LIST:-1 2 4 8 16 32 64 128})

# OpenMP + OpenMPI safe defaults
export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMPI_MCA_btl=${OMPI_MCA_btl:-"^openib"}

# ------------------ Checks ------------------
if [[ ! -x "$EXE" ]]; then
  echo "[fatal] executable not found: $EXE" >&2
  exit 1
fi

# ------------------ Header ------------------
{
  echo "==== Strong Scaling Run ===="
  echo "date: $(date)"
  echo "host: $(hostname)"
  echo "exe: $EXE"
  echo "matrix: $MATRIX"
  echo "threads: $THREADS"
  echo "schedule: $SCHED chunk=$CHUNK"
  echo "repeats: $REPEATS trials: $TRIALS"
  echo "P list: ${PROCS_LIST[*]}"
  echo
  printf "%-4s %-12s %-14s %-12s %-12s %-12s %-14s %-12s %-12s %-14s %-14s %-14s\n" \
    "P" \
    "p90_e2e_ms" "p90_compute_ms" "p90_comm_ms" \
    "gflops_e2e" "gbps_e2e" "gflops_compute" "gbps_compute" \
    "commKiB_min" "commKiB_avg" "commKiB_max" \
    "memMiB_max"
} > "$OUTFILE"

# ------------------ Helpers ------------------
get_first_number_after_colon() {
  # reads stdin; finds first match; prints numeric
  awk -F':' '
    { # line contains "... : <num> ..."
      if (NF>=2) {
        x=$2
        gsub(/^[ \t]+/,"",x)
        # keep only first token
        split(x,a," ")
        print a[1]
        exit
      }
    }'
}

get_number_from_line_matching() {
  # $1 = regex
  # reads stdin
  local re="$1"
  awk -v re="$re" '
    $0 ~ re {
      # find first float/int in the line
      for(i=1;i<=NF;i++){
        if($i ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/){
          print $i; exit
        }
      }
    }'
}

get_triplet_total_from_stats_line() {
  # expects line like: "Per-rank max (KiB): total=123.4 ..."
  # prints value after total=
  awk '
    /Per-rank/ && /total=/ {
      for(i=1;i<=NF;i++){
        if($i ~ /total=/){
          gsub(/.*total=/,"",$i)
          gsub(/[,;]/,"",$i)
          print $i
          exit
        }
      }
    }'
}

# ------------------ Run loop ------------------
for P in "${PROCS_LIST[@]}"; do
  # PBS mapping: <=72 -> single node, else use both nodes
  MAP_ARGS=()
  if [[ "$P" -le 72 ]]; then
    MAP_ARGS=(--map-by "ppr:${P}:node:pe=1" --bind-to core)
  else
    MAP_ARGS=(--map-by "ppr:72:node:pe=1" --bind-to core)
  fi

  echo "[run] P=$P ..." >&2

  set +e
  OUT=$(mpirun -np "$P" "${MAP_ARGS[@]}" \
        "$EXE" "$MATRIX" "$THREADS" "$SCHED" "$CHUNK" "$REPEATS" "$TRIALS" --no-validate 2>&1)
  RC=$?
  set -e

  if [[ $RC -ne 0 ]]; then
    {
      echo "[fatal] mpirun failed at P=$P (rc=$RC). Output:"
      echo "$OUT"
    } >> "$OUTFILE"
    exit $RC
  fi

  # ---- core metrics (robust parsing) ----
  p90_e2e=$(echo "$OUT" | awk -F':' '/P90 execution time/ {gsub(/ ms/,"",$2); sub(/^[ \t]+/,"",$2); print $2}' | tail -n1)
  gflops_e2e=$(echo "$OUT" | awk -F':' '/Throughput/ {gsub(/ GFLOPS/,"",$2); sub(/^[ \t]+/,"",$2); print $2}' | tail -n1)
  gbps_e2e=$(echo "$OUT" | awk -F':' '/Estimated bandwidth/ {gsub(/ GB\/s/,"",$2); sub(/^[ \t]+/,"",$2); print $2}' | tail -n1)

  p90_compute=$(echo "$OUT" | awk -F':' '/Compute-only P90 time/ {gsub(/ ms/,"",$2); sub(/^[ \t]+/,"",$2); print $2}' | tail -n1)
  gflops_compute=$(echo "$OUT" | awk -F':' '/Compute-only GFLOPS/ {gsub(/ GFLOPS/,"",$2); sub(/^[ \t]+/,"",$2); print $2}' | tail -n1)
  gbps_compute=$(echo "$OUT" | awk -F':' '/Compute-only BW/ {gsub(/ GB\/s/,"",$2); sub(/^[ \t]+/,"",$2); print $2}' | tail -n1)

  p90_comm=$(echo "$OUT" | awk -F':' '/Comm-only P90 time/ {gsub(/ ms/,"",$2); sub(/^[ \t]+/,"",$2); print $2}' | tail -n1)

  # ---- comm volume / memory footprint (optional if printed) ----
  # We try to parse lines of the form:
  #   "Per-rank min (KiB): total=..."
  #   "Per-rank avg (KiB): total=..."
  #   "Per-rank max (KiB): total=..."
  #   "Per-rank max (MiB): total=..."
  comm_min=$(echo "$OUT" | awk '/Per-rank min \(KiB\)/{print}' | get_triplet_total_from_stats_line | tail -n1)
  comm_avg=$(echo "$OUT" | awk '/Per-rank avg \(KiB\)/{print}' | get_triplet_total_from_stats_line | tail -n1)
  comm_max=$(echo "$OUT" | awk '/Per-rank max \(KiB\)/{print}' | get_triplet_total_from_stats_line | tail -n1)
  mem_max=$(echo "$OUT" | awk '/Per-rank max \(MiB\)/{print}' | get_triplet_total_from_stats_line | tail -n1)

  # defaults if not present
  p90_e2e=${p90_e2e:-0}
  p90_compute=${p90_compute:-0}
  p90_comm=${p90_comm:-0}
  gflops_e2e=${gflops_e2e:-0}
  gbps_e2e=${gbps_e2e:-0}
  gflops_compute=${gflops_compute:-0}
  gbps_compute=${gbps_compute:-0}
  comm_min=${comm_min:-0}
  comm_avg=${comm_avg:-0}
  comm_max=${comm_max:-0}
  mem_max=${mem_max:-0}

  printf "%-4d %-12.3f %-14.3f %-12.3f %-12.3f %-12.3f %-14.3f %-12.3f %-12.3f %-14.3f %-14.3f %-14.3f\n" \
    "$P" \
    "$p90_e2e" "$p90_compute" "$p90_comm" \
    "$gflops_e2e" "$gbps_e2e" "$gflops_compute" "$gbps_compute" \
    "$comm_min" "$comm_avg" "$comm_max" \
    "$mem_max" >> "$OUTFILE"
done

echo "Saved results to: $OUTFILE" >&2