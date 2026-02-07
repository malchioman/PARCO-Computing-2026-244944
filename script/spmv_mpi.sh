#!/bin/bash
set -euo pipefail

command -v qsub >/dev/null 2>&1 || {
  echo "[fatal] qsub not found: questo script va eseguito sul CLUSTER (login node), non in locale/WSL."
  exit 127
}

# Uso:
#   ./script/spmv_mpi.sh <NP> <THREADS> <matrix.mtx> [sched] [chunk] [repeats] [trials] [flags...]
#
# Esempio:
#   ./script/spmv_mpi.sh 128 1 kron_g500-logn21.mtx static 64 10 5 --sort-rows

NP="${1:?Usage: spmv_mpi.sh <NP> <THREADS> <matrix.mtx> [sched] [chunk] [repeats] [trials] [flags...]}"
THREADS="${2:?Usage: spmv_mpi.sh <NP> <THREADS> <matrix.mtx> [sched] [chunk] [repeats] [trials] [flags...]}"
MATRIX="${3:?Usage: spmv_mpi.sh <NP> <THREADS> <matrix.mtx> [sched] [chunk] [repeats] [trials] [flags...]}"
SCHED="${4:-static}"
CHUNK="${5:-64}"
REPEATS="${6:-10}"
TRIALS="${7:-5}"
shift 7 || true
FLAGS="$*"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PBS_SCRIPT="$REPO_ROOT/script/spmv_mpi.pbs"
EXE="$REPO_ROOT/bin/spmv_mpi"

mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/results"
export MATRICES_DIR="$REPO_ROOT/bin/matrices"

# ---- Risorse per nodo (PBS Pro select=...) ----
# Default: nodo da 72 core come nel tuo strong.pbs
NCPUS_NODE="${NCPUS_NODE:-72}"

if (( THREADS > NCPUS_NODE )); then
  echo "[fatal] THREADS=$THREADS > NCPUS_NODE=$NCPUS_NODE"
  exit 2
fi

# ranks per node (evita oversubscribe): floor(NCPUS_NODE / THREADS)
RPN_MAX=$(( NCPUS_NODE / THREADS ))
RPN=$RPN_MAX
if (( RPN > NP )); then RPN=$NP; fi
if (( RPN < 1 )); then RPN=1; fi

# nodes = ceil(NP / RPN)
NODES=$(( (NP + RPN - 1) / RPN ))

# core per chunk realmente richiesti
NCPUS_CHUNK=$(( RPN * THREADS ))

QUEUE="${QUEUE:-short_cpuQ}"
WALLTIME="${WALLTIME:-01:00:00}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="$REPO_ROOT/logs/spmv_mpi_np${NP}_t${THREADS}_$(basename "$MATRIX")_${TS}.out"

# QUIET=1 (default) => niente output del programma, solo risultati in results/
QUIET="${QUIET:-1}"

echo "Submitting:"
echo "  repo      : $REPO_ROOT"
echo "  exe       : $EXE"
echo "  matrix    : $MATRIX"
echo "  NP        : $NP"
echo "  threads   : $THREADS"
echo "  per node  : NCPUS_NODE=$NCPUS_NODE -> RPN=$RPN"
echo "  select    : $NODES x (ncpus=$NCPUS_CHUNK, mpiprocs=$RPN)"
echo "  queue     : $QUEUE"
echo "  walltime  : $WALLTIME"
echo "  quiet     : $QUIET"
echo "  out log   : $OUT"
echo

qsub -q "$QUEUE" -l "select=${NODES}:ncpus=${NCPUS_CHUNK}:mpiprocs=${RPN}" -l "walltime=${WALLTIME}" \
  -o "$OUT" -j oe \
  -v "REPO_ROOT=${REPO_ROOT},EXE=${EXE},MATRICES_DIR=${REPO_ROOT}/bin/matrices,NP=${NP},THREADS=${THREADS},MATRIX=${MATRIX},SCHED=${SCHED},CHUNK=${CHUNK},REPEATS=${REPEATS},TRIALS=${TRIALS},FLAGS=${FLAGS},RPN=${RPN},QUIET=${QUIET}" \
  "$PBS_SCRIPT"
