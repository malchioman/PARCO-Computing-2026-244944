#!/usr/bin/env bash
set -euo pipefail
export LC_NUMERIC=C

EXE=${EXE:-"./bin/spmv_mpi"}
MATRIX=${1:-"bin/matrices/kron_g500-logn21.mtx"}

THREADS=${THREADS:-1}
SCHED=${SCHED:-static}
CHUNK=${CHUNK:-64}
REPEATS=${REPEATS:-10}
TRIALS=${TRIALS:-5}

PROCS_LIST=(${PROCS_LIST:-1 2 4 8 16 32 64 128})

OUTDIR="results"
OUTFILE="${OUTDIR}/strong.txt"
mkdir -p "$OUTDIR"

export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMPI_MCA_btl=${OMPI_MCA_btl:-"^openib"}

if [[ ! -x "$EXE" ]]; then
  echo "[fatal] executable not found: $EXE" >&2
  exit 1
fi

# Header
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
  printf "P\tp90_e2e_ms\tp90_compute_ms\tp90_comm_ms\tgflops_e2e\tgbps_e2e\tgflops_compute\tgbps_compute\n"
} > "$OUTFILE"

extract_number() {
  # $1 = regex/pattern, returns first numeric token after ':'
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
      # prende il primo numero dopo i ":" (nnz_used), ignorando header=...
      split($0, parts, ":")
      s=parts[2]
      gsub(/[^0-9]/, " ", s)
      split(s,a," ")
      for(i=1;i<=length(a);i++) if(a[i]!=""){ print a[i]; exit }
    }'
}

calc_gflops() {
  # nnz, t_ms
  awk -v nnz="$1" -v tms="$2" 'BEGIN{
    if (tms<=0) { print "inf"; exit }
    printf "%.6f", (2.0*nnz)/((tms/1000.0)*1e9)
  }'
}

calc_gbps() {
  # nnz, M, t_ms  (byte model: nnz*(8+4+8) + M*8)
  awk -v nnz="$1" -v M="$2" -v tms="$3" 'BEGIN{
    if (tms<=0) { print "inf"; exit }
    bytes = nnz*(8+4+8) + M*8
    printf "%.6f", bytes/((tms/1000.0)*1e9)
  }'
}

for P in "${PROCS_LIST[@]}"; do
  echo "[run] P=$P ..." >> "$OUTFILE"

  MAP_ARGS=()
  if [[ "$P" -le 72 ]]; then
    MAP_ARGS=(--map-by "ppr:${P}:node:pe=1" --bind-to core)
  else
    MAP_ARGS=(--map-by "ppr:72:node:pe=1" --bind-to core)
  fi

  set +e
  OUT=$(mpirun -np "$P" "${MAP_ARGS[@]}" "$EXE" "$MATRIX" "$THREADS" "$SCHED" "$CHUNK" "$REPEATS" "$TRIALS" --no-validate 2>&1)
  RC=$?
  set -e

  if [[ $RC -ne 0 ]]; then
    echo "[fatal] mpirun failed at P=$P (rc=$RC). Output:" >> "$OUTFILE"
    echo "$OUT" >> "$OUTFILE"
    exit $RC
  fi

  # Times (ms)
  P90_E2E=$(extract_number "P90 execution time")
  P90_COMP=$(extract_number "Compute-only P90 time")
  P90_COMM=$(extract_number "Comm-only P90 time")

  # Backward compatibility (se non trovi le nuove righe)
  if [[ -z "${P90_E2E:-}" ]]; then
    echo "[fatal] parsing failed at P=$P (missing P90 execution time). Full output:" >> "$OUTFILE"
    echo "$OUT" >> "$OUTFILE"
    exit 2
  fi
  if [[ -z "${P90_COMP:-}" ]]; then P90_COMP="$P90_E2E"; fi
  if [[ -z "${P90_COMM:-}" ]]; then P90_COMM="0"; fi

  # Parse M and nnz_used from program output (serve per ricalcolare GF/GB su entrambe le metriche)
  M=$(extract_M)
  NNZ=$(extract_nnz_used)

  if [[ -z "${M:-}" || -z "${NNZ:-}" ]]; then
    echo "[fatal] parsing failed at P=$P (missing Dimensions or Non-zero entries). Full output:" >> "$OUTFILE"
    echo "$OUT" >> "$OUTFILE"
    exit 2
  fi

  GF_E2E=$(calc_gflops "$NNZ" "$P90_E2E")
  GB_E2E=$(calc_gbps   "$NNZ" "$M" "$P90_E2E")
  GF_COMP=$(calc_gflops "$NNZ" "$P90_COMP")
  GB_COMP=$(calc_gbps   "$NNZ" "$M" "$P90_COMP")

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$P" "$P90_E2E" "$P90_COMP" "$P90_COMM" \
    "$GF_E2E" "$GB_E2E" "$GF_COMP" "$GB_COMP" >> "$OUTFILE"
done

echo "Saved results to: $OUTFILE"
