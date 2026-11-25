# Corrida diaria del paper trading, para lanzar a mano antes de la apertura:
#
#     powershell -ExecutionPolicy Bypass -File scripts\paper_daily.ps1
#
# El loop espera a que abra el mercado,
# opera y monitorea durante toda la rueda, y termina después del cierre dejando
# el reporte de sesión en reportes\paper. El log de consola queda en
# reportes\paper\logs\AAAA-MM-DD.log.
#
# Corte manual: crear un archivo llamado KILL en esta carpeta detiene las
# entradas y cierra todas las posiciones en el ciclo siguiente.

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$logDir = Join-Path $root "reportes\paper\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ((Get-Date -Format "yyyy-MM-dd") + ".log")

# Parte 2: el binomial (CRR) y Black-Scholes compiten en la misma cuenta con la
# misma estrategia de valor relativo (modo maker): 3 estructuras abiertas como
# máximo por modelo, 1 unidad, payoff acotado, vencimientos entre 21 y 75 días.
python -m ita_options.pipeline paper-run --live `
    --strategy rv_compite --cap-per-model 3 --contracts 1 --max-open 6 `
    --min-edge 10 --take-profit 0.6 --fill-timeout 180 --rest-seconds 1800 `
    --kill-switch 250 --daily-loss 1000 `
    --min-dte 21 --max-dte 75 --cycle-seconds 120 `
    --hedge delta_gamma --hedge-band 10 --hedge-max-contracts 5 *>> $log
