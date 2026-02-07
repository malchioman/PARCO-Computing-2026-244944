#!/bin/bash
set -euo pipefail

command -v qsub >/dev/null 2>&1 || {
  echo "[fatal] qsub not found: esegui questo script sul CLUSTER (login node), non in locale/WSL."
  exit 127
}

# Usage:
#   ./script/spmv_mpi.sh <NP> <THREADS> <matrix.mtx> [sched] [chunk] [repeats] [trials] [flags...]
NP="${1:?Usage: spmv_mpi.sh <NP> <THREADS> <matrix.mtx> [sched] [chunk] [repeats] [trials] [flags...]}"
THREADS="${2:?Usage: spmv_mpi.sh <NP> <THREADS> <matrix.mtx> [sched] [chunk] [repeats] [trials] [flags...]}"
MATRIX_IN="${3:?Usage: spmv_mpi.sh <NP> <THREADS> <matrix.mtx> [sched] [chunk] [repeats] [trials] [flags...]}"
SCHED="${4:-static}"
CHUNK="${5:-64}"
REPEATS="${6:-10}"
TRIALS="${7:-5}"
shift 7 || true
FLAGS="$*"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PBS_SCRIPT="$REPO_ROOT/script/spmv_mpi.pbs"
EXE="$REPO_ROOT/bin/spmv_mpi"

MATRICES_DIR="$REPO_ROOT/bin/matrices"
mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/results"

# accetta sia "foo.mtx" che "bin/matrices/foo.mtx" -> usa sempre basename
MATRIX="$(basename "$MATRIX_IN")"

# check matrice SOLO in bin/matrices
if [[ ! -f "$MATRICES_DIR/$MATRIX" ]]; then
  echo "[fatal] matrix not found: $MATRIX"
  echo "expected here: $MATRICES_DIR/$MATRIX"
  exit 3
fi

# risorse per nodo (come nel tuo strong.pbs: ncpus=72)
NCPUS_NODE="${NCPUS_NODE:-72}"
QUEUE="${QUEUE:-short_cpuQ}"
WALLTIME="${WALLTIME:-01:00:00}"
QUIET="${QUIET:-1}"

if (( THREADS > NCPUS_NODE )); then
  echo "[fatal] THREADS=$THREADS > NCPUS_NODE=$NCPUS_NODE"
  exit 4
fi

# ranks per node = floor(NCPUS_NODE / THREADS)
RPN=$(( NCPUS_NODE / THREADS ))
if (( RPN > NP )); then RPN=$NP; fi
if (( RPN < 1 )); then RPN=1; fi

# nodes = ceil(NP / RPN)
NODES=$(( (NP + RPN - 1) / RPN ))

# ncpus per chunk (no oversubscribe)
NCPUS_CHUNK=$(( RPN * THREADS ))

TS="$(date +%Y%m%d_%H%M%S)"
PBS_OUT="$REPO_ROOT/logs/spmv_mpi_pbs_np${NP}_t${THREADS}_${TS}.out"

echo "Submitting:"
echo "  repo        : $REPO_ROOT"
echo "  exe         : $EXE"
echo "  matrices    : $MATRICES_DIR"
echo "  matrix      : $MATRIX"
echo "  NP          : $NP"
echo "  threads     : $THREADS"
echo "  select      : ${NODES} x (ncpus=${NCPUS_CHUNK}, mpiprocs=${RPN})"
echo "  queue       : $QUEUE"
echo "  walltime    : $WALLTIME"
echo "  quiet       : $QUIET"
echo "  pbs out     : $PBS_OUT"
echo "  program log : logs/spmv_mpi_run_<JOBID>.log"
echo

JOBID=$(qsub \
  -q "$QUEUE" \
  -l "select=${NODES}:ncpus=${NCPUS_CHUNK}:mpiprocs=${RPN}" \
  -l "walltime=${WALLTIME}" \
  -o "$PBS_OUT" -j oe \
  -v "REPO_ROOT=${REPO_ROOT},EXE=${EXE},NP=${NP},THREADS=${THREADS},MATRIX=${MATRIX},SCHED=${SCHED},CHUNK=${CHUNK},REPEATS=${REPEATS},TRIALS=${TRIALS},FLAGS=${FLAGS},RPN=${RPN},QUIET=${QUIET}" \
  "$PBS_SCRIPT"
)

echo "Job submitted: $JOBID"
echo "PBS log      : $PBS_OUT"
echo "Program log  : $REPO_ROOT/logs/spmv_mpi_run_${JOBID}.log"
