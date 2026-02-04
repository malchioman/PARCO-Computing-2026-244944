#!/usr/bin/env bash
set -euo pipefail
export LC_NUMERIC=C

EXE=${EXE:-"./bin/spmv_mpi"}
GEN=${GEN:-"./bin/custom_matrix"}

THREADS=${THREADS:-1}
SCHED=${SCHED:-static}
CHUNK=${CHUNK:-64}
REPEATS=${REPEATS:-10}
TRIALS=${TRIALS:-5}

ROWS_PER_RANK=${ROWS_PER_RANK:-16384}
NNZ_PER_RANK=${NNZ_PER_RANK:-1000000}

PROCS_LIST=(${PROCS_LIST:-1 2 4 8 16 32 64 128})

OUTDIR="results"
OUTFILE="${OUTDIR}/weak.txt"
mkdir -p "$OUTDIR"

export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMPI_MCA_btl=${OMPI_MCA_btl:-"^openib"}

if [[ ! -x "$EXE" ]]; then
  echo "[fatal] executable not found: $EXE" >&2
  exit 1
fi
if [[ ! -x "$GEN" ]]; then
  echo "[fatal] generator not found/executable: $GEN" >&2
  exit 1
fi

# --- pretty table formatting ---
HDR_FMT="%-4s %10s %10s %12s %12s %14s %12s %12s %12s %15s %15s\n"
ROW_FMT="%-4s %10s %10s %12s %12s %14s %12s %12s %12s %15s %15s\n"

extract_number() {
  echo "$OUT" | awk -F':' -v pat="$1" '
    $0 ~ pat {
      s=$2
      gsub(/[^0-9eE\.\+\-]/, "", s)
      if (s != "") { print s; exit }
    }'
}

extract_M() {
  echo "$OUT" | awk '
    /Dimensions/ {
      s=$0
      gsub(/[^0-9]/, " ", s)
      split(s,a," ")
      for(i=1;i<=length(a);i++) if(a[i]!=""){ print a[i]; exit }
    }'
}

extract_nnz_used() {
  echo "$OUT" | awk '
    /Non-zero entries/ {
      split($0, parts, ":")
      s=parts[2]
      gsub(/[^0-9]/, " ", s)
      split(s,a," ")
      for(i=1;i<=length(a);i++) if(a[i]!=""){ print a[i]; exit }
    }'
}

calc_gflops() {
  awk -v nnz="$1" -v tms="$2" 'BEGIN{
    if (tms<=0) { print "inf"; exit }
    printf "%.6f", (2.0*nnz)/((tms/1000.0)*1e9)
  }'
}

calc_gbps() {
  awk -v nnz="$1" -v M="$2" -v tms="$3" 'BEGIN{
    if (tms<=0) { print "inf"; exit }
    bytes = nnz*(8+4+8) + M*8
    printf "%.6f", bytes/((tms/1000.0)*1e9)
  }'
}

# Header
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
  echo "P list: ${PROCS_LIST[*]}"
  echo
  printf "$HDR_FMT" \
    "P" "rows" "cols" "nnz" \
    "p90_e2e_ms" "p90_compute_ms" "p90_comm_ms" \
    "gflops_e2e" "gbps_e2e" "gflops_compute" "gbps_compute"
} > "$OUTFILE"

for P in "${PROCS_LIST[@]}"; do
  # Weak scaling: local rows constant => global rows = rows_per_rank * P
  M=$((ROWS_PER_RANK * P))
  N=$M
  NNZ=$((NNZ_PER_RANK * P))

  OUTNAME="weak_P${P}_rpr${ROWS_PER_RANK}_nnzpr${NNZ_PER_RANK}.mtx"
  OUTPATH="bin/matrices/${OUTNAME}"

  echo "[gen] P=$P -> $OUTPATH (rows=$M cols=$N nnz=$NNZ)" >&2
  "$GEN" "$OUTPATH" "$M" "$N" "$NNZ" >/dev/null

  if [[ ! -f "$OUTPATH" ]]; then
    {
      echo
      echo "[fatal] generator did not create expected file: $OUTPATH"
    } >> "$OUTFILE"
    exit 2
  fi

  echo "[run] P=$P ..." >&2

  MAP_ARGS=()
  if [[ "$P" -le 72 ]]; then
    MAP_ARGS=(--map-by "ppr:${P}:node:pe=1" --bind-to core)
  else
    MAP_ARGS=(--map-by "ppr:72:node:pe=1" --bind-to core)
  fi

  set +e
  OUT=$(mpirun -np "$P" "${MAP_ARGS[@]}" "$EXE" "$OUTPATH" "$THREADS" "$SCHED" "$CHUNK" "$REPEATS" "$TRIALS" --no-validate 2>&1)
  RC=$?
  set -e

  if [[ $RC -ne 0 ]]; then
    {
      echo
      echo "[fatal] mpirun failed at P=$P (rc=$RC). Output:"
      echo "$OUT"
    } >> "$OUTFILE"
    exit $RC
  fi

  # parse times
  P90_E2E=$(extract_number "P90 execution time")
  P90_COMP=$(extract_number "Compute-only P90 time")
  P90_COMM=$(extract_number "Comm-only P90 time")
  [[ -n "${P90_COMP:-}" ]] || P90_COMP="$P90_E2E"
  [[ -n "${P90_COMM:-}" ]] || P90_COMM="0"

  # parse M + nnz_used from program output (robusto)
  M_OUT=$(extract_M)
  NNZ_USED=$(extract_nnz_used)
  [[ -n "${M_OUT:-}" && -n "${NNZ_USED:-}" ]] || { echo "[fatal] parsing M/nnz failed at P=$P" >&2; exit 2; }

  GF_E2E=$(calc_gflops "$NNZ_USED" "$P90_E2E")
  GB_E2E=$(calc_gbps   "$NNZ_USED" "$M_OUT" "$P90_E2E")
  GF_COMP=$(calc_gflops "$NNZ_USED" "$P90_COMP")
  GB_COMP=$(calc_gbps   "$NNZ_USED" "$M_OUT" "$P90_COMP")

  printf "$ROW_FMT" \
    "$P" "$M_OUT" "$M_OUT" "$NNZ_USED" \
    "$P90_E2E" "$P90_COMP" "$P90_COMM" \
    "$GF_E2E" "$GB_E2E" "$GF_COMP" "$GB_COMP" >> "$OUTFILE"
done

echo "Saved results to: $OUTFILE" >&2
