#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXE="$SCRIPT_DIR/bin/spmv_csr.exe"
MAT="$SCRIPT_DIR/bin/matrices/irreg_50k.mtx"
THREADS="${1:-20}"   # se non passi un argomento → 20 thread
"$EXE" "$MAT" "$THREADS" dynamic 32 10 5
