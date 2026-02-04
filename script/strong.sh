#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config (puoi cambiare qui)
# =========================
EXE=${EXE:-"./bin/spmv_mpi"}
MATRIX=${1:-"DIMACS10/kron_g500-logn21"}

THREADS=${THREADS:-1}
SCHED=${SCHED:-static}     # static|dynamic|guided
CHUNK=${CHUNK:-64}
REPEATS=${REPEATS:-10}
TRIALS=${TRIALS:-5}

PROCS_LIST=(${PROCS_LIST:-1 2 4 8 16 32 64 128})

OUTDIR="results"
OUTFILE="${OUTDIR}/strong.txt"

# =========================
# Checks
# =========================
mkdir -p "$OUTDIR"

if [[ ! -x "$EXE" ]]; then
  echo "[fatal] executable not found or not executable: $EXE" >&2
  exit 1
fi

if [[ ! -f "$MATRIX" ]]; then
  echo "[fatal] matrix not found: $MATRIX" >&2
  echo "        (pass it as argument: ./script/strong.sh <matrix_path>)" >&2
  exit 1
fi

# =========================
# OpenMP pinning
# =========================
export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# OpenMPI: spesso serve su UniTN cluster
export OMPI_MCA_btl=${OMPI_MCA_btl:-"^openib"}

# =========================
# mpirun command (PBS-aware)
# =========================
MPI_BASE=(mpirun)
if [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE:-}" ]]; then
  MPI_BASE+=(--hostfile "$PBS_NODEFILE")
fi

# =========================
# Header
# =========================
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
  echo "P\tp90_ms\tgflops\tgbps"
} > "$OUTFILE"

# =========================
# Loop
# =========================
for P in "${PROCS_LIST[@]}"; do
  echo "[run] P=$P ..." | tee -a "$OUTFILE" >/dev/null

  # Run (validation off per benchmark)
  OUT="$("${MPI_BASE[@]}" -np "$P" "$EXE" "$MATRIX" "$THREADS" "$SCHED" "$CHUNK" "$REPEATS" "$TRIALS" --no-validate)"

  # Parse numbers from your program output
  P90=$(echo "$OUT" | awk -F':' '/P90 execution time/ {gsub(/ ms/,"",$2); gsub(/^[ \t]+/,"",$2); print $2}')
  GF=$(echo "$OUT"  | awk -F':' '/Throughput/        {gsub(/ GFLOPS/,"",$2); gsub(/^[ \t]+/,"",$2); print $2}')
  GB=$(echo "$OUT"  | awk -F':' '/Estimated bandwidth/ {gsub(/ GB\/s/,"",$2); gsub(/^[ \t]+/,"",$2); print $2}')

  if [[ -z "${P90:-}" || -z "${GF:-}" || -z "${GB:-}" ]]; then
    echo "[fatal] parsing failed for P=$P. Full output:" >> "$OUTFILE"
    echo "$OUT" >> "$OUTFILE"
    exit 2
  fi

  echo -e "${P}\t${P90}\t${GF}\t${GB}" >> "$OUTFILE"
done

echo
echo "Saved results to: $OUTFILE"
