#!/usr/bin/env pwsh
# 1 thread, matrice irregolare, dynamic(32)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Exe = Join-Path $ScriptDir "bin\spmv_csr.exe"
$Mat = Join-Path $ScriptDir "bin\matrices\irreg_50k.mtx"
& $Exe $Mat 1 "dynamic" 32 10 5
