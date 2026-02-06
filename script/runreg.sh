#!/usr/bin/bash
# Regular matrix: static scheduling

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

EXE="$SCRIPT_DIR/../bin/spmv_csr"
MAT="$SCRIPT_DIR/../bin/matrices/reg_150k.mtx"

# Numero di thread (default 20)
THREADS="${1:-20}"

# Esegui
"$EXE" "$MAT" "$THREADS" static 64 10 5
