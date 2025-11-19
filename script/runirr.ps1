param(
    [int]$Threads = 20
)

# Irregular matrix: dynamic scheduling

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

$Exe = Join-Path $ScriptDir "..\bin\spmv_csr.exe"
$Mat = Join-Path $ScriptDir "..\bin\matrices\irreg_50k.mtx"

& $Exe $Mat $Threads "dynamic" 32 10 5
