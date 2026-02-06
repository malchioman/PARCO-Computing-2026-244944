#!/bin/bash
set -euo pipefail

command -v qsub >/dev/null 2>&1 || {
  echo "[fatal] qsub non trovato: questo script va eseguito sul CLUSTER (login node), non in locale/WSL."
  exit 127
}

if [[ $# -lt 3 ]]; then
  echo "Usage:"
  echo "  $0 <NP_ranks> <threads> <matrix.mtx> [sched] [chunk] [repeats] [trials] [flags...]"
  echo ""
  echo "Example:"
  echo "  $0 128 1 my_matrix.mtx static 64 10 5 --sort-rows"
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

# --- repo layout: scripts/.. = repo root ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
JOB="$SCRIPT_DIR/spmv_job.pbs"

QUEUE="${QUEUE:-short_cpuQ}"
WALLTIME="${WALLTIME:-01:00:00}"

# Core per nodo (ppn) reale del cluster (override: PPN=16 ./submit_spmv.sh ...)
PPN="${PPN:-8}"

# Memoria “per nodo” (GB). Il submit la scala con NODES.
MEM_PER_NODE_GB="${MEM_PER_NODE_GB:-8}"

# ranks per node = floor(ppn / threads)
RPN=$(( PPN / THREADS ))
if [[ "$RPN" -lt 1 ]]; then
  echo "[fatal] THREADS=$THREADS > PPN=$PPN -> non mappabile senza oversubscribe"
  exit 2
fi

# nodes = ceil(np / rpn)
NODES=$(( (NP + RPN - 1) / RPN ))

# memoria totale (molti PBS/Torque interpretano mem come totale job)
TOTAL_MEM_GB=$(( NODES * MEM_PER_NODE_GB ))

# Argomenti al main
ARGS="$MATRIX $THREADS $SCHED $CHUNK $REPEATS $TRIALS"
for f in "${FLAGS[@]}"; do ARGS+=" $f"; done

mkdir -p "$ROOT/logs"

echo "Submitting:"
echo "  ROOT=$ROOT"
echo "  NP=$NP THREADS=$THREADS PPN=$PPN -> RPN=$RPN -> NODES=$NODES"
echo "  mem=${TOTAL_MEM_GB}gb walltime=$WALLTIME queue=$QUEUE"
echo "  ARGS: $ARGS"

JOBID=$(qsub \
  -q "$QUEUE" \
  -l "nodes=${NODES}:ppn=${PPN},mem=${TOTAL_MEM_GB}gb,walltime=${WALLTIME}" \
  -v "ROOT=${ROOT},NP=${NP},PPN=${PPN},ARGS=${ARGS},EXE=${ROOT}/bin/spmv_mpi,MATRICES_DIR=${ROOT}/bin/matrices" \
  "$JOB")

echo "Job submitted: $JOBID"
LOG="$ROOT/logs/spmv_${JOBID}.log"
echo "Log file     : $LOG"

# --- STAMPA SEMPRE (default) ---
# Se vuoi disabilitare: FOLLOW=0 ./scripts/submit_spmv.sh ...
FOLLOW="${FOLLOW:-1}"
if [[ "$FOLLOW" == "1" ]]; then
  echo "[follow] streaming output (Ctrl+C per smettere di seguire; il job continua)"
  while [[ ! -f "$LOG" ]]; do sleep 0.2; done
  tail -f "$LOG"
fi
