# Trae a tu compu los reportes y el ledger del paper trading que corre en la VM.
#
#     powershell -ExecutionPolicy Bypass -File deploy\oracle\traer_resultados.ps1 -Ip <IP_PUBLICA> -Llave <ruta\a\la\llave.key>
#
# Copia reportes/paper (sesiones y logs) y el ledger a reportes\paper_vm\ en esta carpeta.
param(
    [Parameter(Mandatory = $true)][string]$Ip,
    [Parameter(Mandatory = $true)][string]$Llave
)
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$dest = Join-Path $root "reportes\paper_vm"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$remote = "ubuntu@${Ip}:TP-MAPF/ita_options_pipeline"
scp -i $Llave -r "${remote}/reportes/paper" $dest
scp -i $Llave "${remote}/data/paper/ledger.sqlite" $dest
Write-Host "Resultados en $dest"
