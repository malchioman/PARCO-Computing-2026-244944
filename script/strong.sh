#!/usr/bin/env bash
set -euo pipefail

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
  printf "P\tp90_ms\tgflops\tgbps\n"
} > "$OUTFILE"

for P in "${PROCS_LIST[@]}"; do
  echo "[run] P=$P ..." >> "$OUTFILE"

  # Per strong scaling “pulito”: fino a 72 metti tutto su 1 nodo, oltre lasciamo che usi entrambi.
  MAP_ARGS=()
  if [[ "$P" -le 72 ]]; then
    MAP_ARGS=(--map-by "ppr:${P}:node:pe=1" --bind-to core)
  else
    MAP_ARGS=(--map-by "ppr:72:node:pe=1" --bind-to core)
  fi

  # IMPORTANT: in PBS NON forzare hostfile, lascia che OpenMPI usi l’allocazione PBS.
  set +e
  OUT=$(mpirun -np "$P" "${MAP_ARGS[@]}" "$EXE" "$MATRIX" "$THREADS" "$SCHED" "$CHUNK" "$REPEATS" "$TRIALS" --no-validate 2>&1)
  RC=$?
  set -e

  if [[ $RC -ne 0 ]]; then
    echo "[fatal] mpirun failed at P=$P (rc=$RC). Output:" >> "$OUTFILE"
    echo "$OUT" >> "$OUTFILE"
    exit $RC
  fi

  P90=$(echo "$OUT" | awk -F':' '/P90 execution time/ {gsub(/ ms/,"",$2); sub(/^[ \t]+/,"",$2); print $2}')
  GF=$(echo "$OUT"  | awk -F':' '/Throughput/        {gsub(/ GFLOPS/,"",$2); sub(/^[ \t]+/,"",$2); print $2}')
  GB=$(echo "$OUT"  | awk -F':' '/Estimated bandwidth/ {gsub(/ GB\/s/,"",$2); sub(/^[ \t]+/,"",$2); print $2}')

  if [[ -z "${P90:-}" || -z "${GF:-}" || -z "${GB:-}" ]]; then
    echo "[fatal] parsing failed at P=$P. Full program output:" >> "$OUTFILE"
    echo "$OUT" >> "$OUTFILE"
    exit 2
  fi

  printf "%s\t%s\t%s\t%s\n" "$P" "$P90" "$GF" "$GB" >> "$OUTFILE"
done

echo "Saved results to: $OUTFILE"
