#!/usr/bin/env pwsh
# 1 thread, matrice regolare, static(64)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Exe = Join-Path $ScriptDir "bin\spmv_csr.exe"
$Mat = Join-Path $ScriptDir "bin\matrices\reg_150k.mtx"
& $Exe $Mat 1 "static" 64 10 5
