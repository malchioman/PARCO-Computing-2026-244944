
# Numero di thread (default 20 se non passato)
param(
    [int]$Threads = 20
)
# === Regular matrix: static scheduling ===

# Ottieni il percorso della cartella dove si trova questo script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Percorsi relativi
$Exe = Join-Path $ScriptDir "bin\spmv_csr"
$Mat = Join-Path $ScriptDir "bin\matrices\reg_150k.mtx"


# Esegui
& $Exe $Mat $Threads "static" 64 10 5
