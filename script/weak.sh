#!/usr/bin/env bash
set -euo pipefail

EXE=${EXE:-"./bin/spmv_mpi"}
GEN=${GEN:-"./bin/custom_matrix"}

# Weak scaling idea:
# keep rows-per-rank constant => total rows M = BASE_ROWS_PER_RANK * P
BASE_ROWS_PER_RANK=${BASE_ROWS_PER_RANK:-4096}
NNZ_PER_ROW=${NNZ_PER_ROW:-16}

THREADS=${THREADS:-1}
SCHED=${SCHED:-static}
CHUNK=${CHUNK:-64}
REPEATS=${REPEATS:-10}
TRIALS=${TRIALS:-5}

PROCS_LIST=(${PROCS_LIST:-1 2 4 8 16 32 64 128})

OUTDIR="results"
OUTFILE="${OUTDIR}/weak.txt"
mkdir -p "$OUTDIR" "bin/matrices"

export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# If not set in PBS, keep safe defaults:
export OMPI_MCA_oob=${OMPI_MCA_oob:-tcp}
export OMPI_MCA_btl=${OMPI_MCA_btl:-self,tcp}
export OMPI_MCA_mpi_cuda_support=${OMPI_MCA_mpi_cuda_support:-0}

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
  echo "exe: $EXE"
  echo "gen: $GEN"
  echo "rows_per_rank: $BASE_ROWS_PER_RANK"
  echo "nnz_per_row: $NNZ_PER_ROW"
  echo "threads: $THREADS"
  echo "schedule: $SCHED chunk=$CHUNK"
  echo "repeats: $REPEATS trials: $TRIALS"
  echo "P list: ${PROCS_LIST[*]}"
  echo
  printf "P\tp90_ms\tgflops\tgbps\n"
} > "$OUTFILE"

for P in "${PROCS_LIST[@]}"; do
  M=$((BASE_ROWS_PER_RANK * P))
  N=$M
  MAT="bin/matrices/weak_M${M}_k${NNZ_PER_ROW}.mtx"

  echo "[gen] P=$P -> $MAT (M=$M, N=$N, nnz_per_row=$NNZ_PER_ROW)" >> "$OUTFILE"
  # Generate only if missing (set FORCE_GEN=1 to regenerate)
  if [[ "${FORCE_GEN:-0}" == "1" || ! -f "$MAT" ]]; then
    "$GEN" "$MAT" "$M" "$N" "$NNZ_PER_ROW" >> "$OUTFILE" 2>&1
  fi

  echo "[run] P=$P ..." >> "$OUTFILE"

  MAP_ARGS=()
  if [[ "$P" -le 72 ]]; then
    MAP_ARGS=(--map-by "ppr:${P}:node:pe=1" --bind-to core)
  else
    MAP_ARGS=(--map-by "ppr:72:node:pe=1" --bind-to core)
  fi

  set +e
  OUT=$(mpirun -np "$P" "${MAP_ARGS[@]}" \
        "$EXE" "$MAT" "$THREADS" "$SCHED" "$CHUNK" "$REPEATS" "$TRIALS" --no-validate 2>&1)
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
