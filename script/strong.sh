#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXE="$REPO_ROOT/bin/spmv_mpi"
OUTDIR="$REPO_ROOT/results"
OUT="$OUTDIR/strong.txt"

mkdir -p "$OUTDIR"
export MATRICES_DIR="$REPO_ROOT/bin/matrices"

# ---------------- CONFIG ----------------
MATRIX_NAME="${1:-kron_g500-logn21.mtx}"   # oppure quello che usi tu
THREADS="${2:-1}"
SCHED="${3:-static}"
CHUNK="${4:-64}"
REPEATS="${5:-10}"
TRIALS="${6:-5}"

P_LIST=(1 2 4 8 16 32 64 128)
# ---------------------------------------

{
  echo "==== Strong Scaling Run ===="
  echo "date: $(date)"
  echo "host: $(hostname)"
  echo "exe:  $EXE"
  echo "matrix: $MATRIX_NAME"
  echo "threads: $THREADS"
  echo "schedule: $SCHED chunk=$CHUNK"
  echo "repeats: $REPEATS trials: $TRIALS"
  echo "P list: ${P_LIST[*]}"
  echo
  printf "%-6s %-12s %-14s %-12s %-12s %-12s %-12s %-12s %-14s %-14s %-14s %-14s\n" \
    "P" "p90_e2e_ms" "p90_comp_ms" "p90_comm_ms" \
    "gflops_e2e" "gbps_e2e" "gflops_comp" "gbps_comp" \
    "commKiB_max" "memMiB_max" "speedup_e2e" "eff_e2e"
} > "$OUT"

T1_E2E=""

for P in "${P_LIST[@]}"; do
  echo "[run] P=$P ..." >&2

  # Esegui da repo root (così path relativi stabili)
  OUTRUN=$(
    cd "$REPO_ROOT"
    mpirun -np "$P" --bind-to none "$EXE" "$MATRIX_NAME" "$THREADS" "$SCHED" "$CHUNK" "$REPEATS" "$TRIALS"
  )

  # ---- parse tempi / perf ----
  p90_e2e=$(echo "$OUTRUN" | awk -F': ' '/P90 execution time/{print $2}' | awk '{print $1}' | tail -n1)
  p90_comp=$(echo "$OUTRUN" | awk -F': ' '/Compute-only P90 time/{print $2}' | awk '{print $1}' | tail -n1)
  p90_comm=$(echo "$OUTRUN" | awk -F': ' '/Comm-only P90 time/{print $2}' | awk '{print $1}' | tail -n1)

  gflops_e2e=$(echo "$OUTRUN" | awk -F': ' '/Throughput/{print $2}' | awk '{print $1}' | tail -n1)
  gbps_e2e=$(echo "$OUTRUN" | awk -F': ' '/Estimated bandwidth/{print $2}' | awk '{print $1}' | tail -n1)

  gflops_comp=$(echo "$OUTRUN" | awk -F': ' '/Compute-only GFLOPS/{print $2}' | awk '{print $1}' | tail -n1)
  gbps_comp=$(echo "$OUTRUN" | awk -F': ' '/Compute-only BW/{print $2}' | awk '{print $1}' | tail -n1)

  # comm total per iter (per-rank max total=XXX)
  commKiB_max=$(
    echo "$OUTRUN" | awk '
      /Per-rank max \(KiB\)/{
        # riga tipo: Per-rank max (KiB)   : sent=..., recv=..., total=123.456
        for(i=1;i<=NF;i++){
          if($i ~ /^total=/){gsub("total=","",$i); print $i}
        }
      }' | tail -n1
  )

  # mem total max
  memMiB_max=$(
    echo "$OUTRUN" | awk '
      /Per-rank max \(MiB\)/{
        # riga tipo: Per-rank max (MiB)   : total=12.345  [CSR=...
        for(i=1;i<=NF;i++){
          if($i ~ /^total=/){gsub("total=","",$i); print $i}
        }
      }' | tail -n1
  )

  # speedup/eff e2e (solo strong)
  if [[ "$P" -eq 1 ]]; then
    T1_E2E="$p90_e2e"
    speedup="1.000"
    eff="1.000"
  else
    speedup=$(awk -v t1="$T1_E2E" -v tp="$p90_e2e" 'BEGIN{printf "%.3f", (tp>0)?(t1/tp):0}')
    eff=$(awk -v s="$speedup" -v p="$P" 'BEGIN{printf "%.3f", (p>0)?(s/p):0}')
  fi

  printf "%-6d %-12.3f %-14.3f %-12.3f %-12.3f %-12.3f %-12.3f %-12.3f %-14.3f %-14.3f %-14.3f %-14.3f\n" \
    "$P" "$p90_e2e" "$p90_comp" "$p90_comm" \
    "$gflops_e2e" "$gbps_e2e" "$gflops_comp" "$gbps_comp" \
    "${commKiB_max:-0}" "${memMiB_max:-0}" "$speedup" "$eff" >> "$OUT"
done

echo "[done] wrote $OUT" >&2
