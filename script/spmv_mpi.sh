#!/bin/bash
set -euo pipefail

command -v qsub >/dev/null 2>&1 || {
  echo "[fatal] qsub not found: esegui questo script sul CLUSTER (login node), non in locale/WSL."
  exit 127
}

NP="${1:?Usage: spmv_mpi.sh <NP> <THREADS> <matrix.mtx> [sched] [chunk] [repeats] [trials] [flags...]}"
THREADS="${2:?Usage: spmv_mpi.sh <NP> <THREADS> <matrix.mtx> [sched] [chunk] [repeats] [trials] [flags...]}"
MATRIX_IN="${3:?Usage: spmv_mpi.sh <NP> <THREADS> <matrix.mtx> [sched] [chunk] [repeats] [trials] [flags...]}"
SCHED="${4:-static}"
CHUNK="${5:-64}"
REPEATS="${6:-10}"
TRIALS="${7:-5}"
shift 7 || true
FLAGS="$*"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PBS_SCRIPT="$SCRIPT_DIR/spmv_mpi.pbs"
EXE="$REPO_ROOT/bin/spmv_mpi"

MATRICES_DIR="$REPO_ROOT/bin/matrices"
mkdir -p "$REPO_ROOT/results/pbs_out"

MATRIX="$(basename "$MATRIX_IN")"

if [[ ! -f "$MATRICES_DIR/$MATRIX" ]]; then
  echo "[fatal] matrix not found: $MATRIX"
  echo "expected here: $MATRICES_DIR/$MATRIX"
  exit 3
fi

NCPUS_NODE="${NCPUS_NODE:-72}"
QUEUE="${QUEUE:-short_cpuQ}"
WALLTIME="${WALLTIME:-01:00:00}"

if (( THREADS > NCPUS_NODE )); then
  echo "[fatal] THREADS=$THREADS > NCPUS_NODE=$NCPUS_NODE"
  exit 4
fi

RPN=$(( NCPUS_NODE / THREADS ))
if (( RPN > NP )); then RPN=$NP; fi
if (( RPN < 1 )); then RPN=1; fi

NODES=$(( (NP + RPN - 1) / RPN ))
NCPUS_CHUNK=$(( RPN * THREADS ))

TS="$(date +%Y%m%d_%H%M%S)"
PBS_OUT="$REPO_ROOT/results/pbs_out/spmv_mpi_np${NP}_t${THREADS}_${TS}.out"

JOBID=$(qsub \
  -q "$QUEUE" \
  -l "select=${NODES}:ncpus=${NCPUS_CHUNK}:mpiprocs=${RPN}" \
  -l "walltime=${WALLTIME}" \
  -o "$PBS_OUT" -j oe \
  -v "REPO_ROOT=${REPO_ROOT},EXE=${EXE},NP=${NP},THREADS=${THREADS},MATRIX=${MATRIX},SCHED=${SCHED},CHUNK=${CHUNK},REPEATS=${REPEATS},TRIALS=${TRIALS},FLAGS=${FLAGS},RPN=${RPN}" \
  "$PBS_SCRIPT"
)

echo "Job submitted: $JOBID"
echo "PBS output: $PBS_OUT"
