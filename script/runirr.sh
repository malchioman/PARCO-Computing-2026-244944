#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

EXE="$SCRIPT_DIR/../bin/spmv_csr"
MAT="$SCRIPT_DIR/../bin/matrices/irreg_50k.mtx"

#Numero di threads (Default 20)
THREADS="${1:-20}"

"$EXE" "$MAT" "$THREADS" dynamic 32 10 5
