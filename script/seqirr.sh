#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

EXE="$SCRIPT_DIR/../bin/spmv_csr"
MAT="$SCRIPT_DIR/../bin/matrices/irreg_50k.mtx"

"$EXE" "$MAT" 1 dynamic 32 10 5
