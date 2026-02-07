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

MATRICES_DIR="$REPO_ROOT/bin/matrices"
MATRIX="$(basename "$MATRIX_IN")"

# matrice SOLO in bin/matrices
if [[ ! -f "$MATRICES_DIR/$MATRIX" ]]; then
  echo "[fatal] matrix not found: $MATRIX (expected in $MATRICES_DIR)"
  exit 3
fi

# risorse (come strong: nodi da 72 core)
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

# MASTER file unico (corto)
MASTER_OUT_REL="results/pbs_out/spmv_mpi.out"
MASTER_OUT="$REPO_ROOT/$MASTER_OUT_REL"
mkdir -p "$(dirname "$MASTER_OUT")"
touch "$MASTER_OUT"

# submit: PBS stdout/stderr buttati (gestiamo tutto noi nel master file)
JOBID=$(qsub \
  -q "$QUEUE" \
  -l "select=${NODES}:ncpus=${NCPUS_CHUNK}:mpiprocs=${RPN}" \
  -l "walltime=${WALLTIME}" \
  -o /dev/null -e /dev/null \
  -v "REPO_ROOT=${REPO_ROOT},NP=${NP},THREADS=${THREADS},MATRIX=${MATRIX},SCHED=${SCHED},CHUNK=${CHUNK},REPEATS=${REPEATS},TRIALS=${TRIALS},FLAGS=${FLAGS},RPN=${RPN},MASTER_OUT=${MASTER_OUT}" \
  "$PBS_SCRIPT"
)

# append SUBITO: anche se resta in coda
ts="$(date '+%Y-%m-%d %H:%M:%S')"
line="[submitted] $ts job=$JOBID NP=$NP threads=$THREADS matrix=$MATRIX sched=$SCHED chunk=$CHUNK repeats=$REPEATS trials=$TRIALS"

LOCK="${MASTER_OUT}.lock"
if command -v flock >/dev/null 2>&1; then
  { flock -w 30 9; echo "$line" >> "$MASTER_OUT"; } 9>"$LOCK"
else
  echo "$line" >> "$MASTER_OUT"
fi

# output bello in terminale
echo "[submitted] $JOBID  ->  $MASTER_OUT_REL"
