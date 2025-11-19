# Numero di thread (default 20 se non passato)
param(
    [int]$Threads = 20
)

#  Regular matrix: static scheduling

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

$Exe = Join-Path $ScriptDir "..\bin\spmv_csr.exe"
$Mat = Join-Path $ScriptDir "..\bin\matrices\reg_150k.mtx"

& $Exe $Mat $Threads "static" 64 10 5
