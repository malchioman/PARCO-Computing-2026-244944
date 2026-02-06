#!/usr/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

EXE="$SCRIPT_DIR/../bin/spmv_csr"
MAT="$SCRIPT_DIR/../bin/matrices/reg_150k.mtx"

"$EXE" "$MAT" 1 static 64 10 5
