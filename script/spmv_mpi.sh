#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <NP_ranks> <threads> <matrix.mtx> [sched] [chunk] [repeats] [trials] [flags...]"
  echo "Esempio: $0 128 1 my.mtx static 64 10 5 --sort-rows"
  exit 1
fi

NP="$1"
THREADS="$2"
MATRIX="$3"
SCHED="${4:-static}"
CHUNK="${5:-64}"
REPEATS="${6:-10}"
TRIALS="${7:-5}"
shift 7 || true
FLAGS=("$@")

# Dove si trova questo script? -> scripts/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
JOB="$SCRIPT_DIR/spmv_job.pbs"

QUEUE="${QUEUE:-short_cpuQ}"
WALLTIME="${WALLTIME:-01:00:00}"

# core per nodo (ppn) reale del cluster
PPN="${PPN:-8}"
MEM_PER_NODE_GB="${MEM_PER_NODE_GB:-8}"

RPN=$(( PPN / THREADS ))
if [[ "$RPN" -lt 1 ]]; then
  echo "[fatal] THREADS=$THREADS > PPN=$PPN -> richiedi troppi thread per nodo"
  exit 2
fi

NODES=$(( (NP + RPN - 1) / RPN ))
TOTAL_MEM_GB=$(( NODES * MEM_PER_NODE_GB ))

ARGS="$MATRIX $THREADS $SCHED $CHUNK $REPEATS $TRIALS"
for f in "${FLAGS[@]}"; do ARGS+=" $f"; done

echo "Submitting from: $SCRIPT_DIR"
echo "Repo root      : $ROOT"
echo "NP=$NP THREADS=$THREADS PPN=$PPN -> RPN=$RPN -> NODES=$NODES"
echo "mem=${TOTAL_MEM_GB}gb walltime=$WALLTIME queue=$QUEUE"
echo "ARGS: $ARGS"

qsub \
  -q "$QUEUE" \
  -l "nodes=${NODES}:ppn=${PPN},mem=${TOTAL_MEM_GB}gb,walltime=${WALLTIME}" \
  -v "ROOT=${ROOT},NP=${NP},PPN=${PPN},ARGS=${ARGS},EXE=${ROOT}/bin/spmv_mpi,MATRICES_DIR=${ROOT}/bin/matrices" \
  "$JOB"
