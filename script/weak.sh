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

# Weak scaling knobs (work per rank ~ costante)
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
  echo "[fatal] matrix generator not found: $GEN" >&2
  exit 1
fi

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
  printf "P\trows\tcols\tnnz\tp90_e2e_ms\tp90_compute_ms\tp90_comm_ms\tgflops_e2e\tgbps_e2e\tgflops_compute\tgbps_compute\n"
} > "$OUTFILE"

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

for P in "${PROCS_LIST[@]}"; do
  # Weak scaling: rows e nnz crescono linearmente con P
  ROWS=$(( ROWS_PER_RANK * P ))
  COLS=$ROWS
  NNZ=$(( NNZ_PER_RANK  * P ))

  # Nome matrice unico
  NAME="weak_P${P}_rpr${ROWS_PER_RANK}_nnzpr${NNZ_PER_RANK}.mtx"

  echo "[gen] P=$P -> $NAME (rows=$ROWS cols=$COLS nnz=$NNZ)" >> "$OUTFILE"
  "$GEN" "$ROWS" "$COLS" "$NNZ" "$NAME" >/dev/null

  MATRIX="bin/matrices/$NAME"

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

  P90_E2E=$(extract_number "P90 execution time")
  P90_COMP=$(extract_number "Compute-only P90 time")
  P90_COMM=$(extract_number "Comm-only P90 time")

  if [[ -z "${P90_E2E:-}" ]]; then
    echo "[fatal] parsing failed at P=$P (missing P90 execution time). Full output:" >> "$OUTFILE"
    echo "$OUT" >> "$OUTFILE"
    exit 2
  fi
  if [[ -z "${P90_COMP:-}" ]]; then P90_COMP="$P90_E2E"; fi
  if [[ -z "${P90_COMM:-}" ]]; then P90_COMM="0"; fi

  # Prendiamo M e nnz_usato dal programma (è più robusto che fidarsi di NNZ teorico)
  M_OUT=$(extract_M)
  NNZ_OUT=$(extract_nnz_used)
  if [[ -z "${M_OUT:-}" || -z "${NNZ_OUT:-}" ]]; then
    echo "[fatal] parsing failed at P=$P (missing Dimensions or Non-zero entries). Full output:" >> "$OUTFILE"
    echo "$OUT" >> "$OUTFILE"
    exit 2
  fi

  GF_E2E=$(calc_gflops "$NNZ_OUT" "$P90_E2E")
  GB_E2E=$(calc_gbps   "$NNZ_OUT" "$M_OUT" "$P90_E2E")
  GF_COMP=$(calc_gflops "$NNZ_OUT" "$P90_COMP")
  GB_COMP=$(calc_gbps   "$NNZ_OUT" "$M_OUT" "$P90_COMP")

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$P" "$ROWS" "$COLS" "$NNZ_OUT" \
    "$P90_E2E" "$P90_COMP" "$P90_COMM" \
    "$GF_E2E" "$GB_E2E" "$GF_COMP" "$GB_COMP" >> "$OUTFILE"
done

echo "Saved results to: $OUTFILE"
