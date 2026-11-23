#!/usr/bin/env bash
# Corrida diaria del paper trading en la VM (equivalente a scripts/paper_daily.ps1).
#
# La lanza cron de lunes a viernes a las 9:20 de Nueva York. El loop espera la
# apertura, opera y monitorea toda la rueda, y termina después del cierre
# dejando el reporte de sesión en reportes/paper. Log: reportes/paper/logs/AAAA-MM-DD.log
#
# Corte manual: `touch ~/TP-MAPF/ita_options_pipeline/KILL` detiene las entradas
# y cierra todas las posiciones en el ciclo siguiente.
set -u
cd "$(dirname "$0")/../.." || exit 1
mkdir -p reportes/paper/logs
log="reportes/paper/logs/$(date +%F).log"

# Un solo proceso a la vez: si un feriado dejó el anterior esperando la
# apertura, la corrida nueva no se superpone.
exec 9>/tmp/paper_daily.lock
if ! flock -n 9; then
    echo "$(date -Is) ya hay una corrida en curso; se omite" >> "$log"
    exit 0
fi

# Trae las correcciones que se suban al repo (sólo avance rápido; si falla, se
# corre la versión que ya está).
git -C .. pull --ff-only >> "$log" 2>&1 || echo "$(date -Is) git pull falló; sigo con la versión local" >> "$log"

# Parte 2: el binomial (CRR) y Black-Scholes compiten en la misma cuenta con la
# misma estrategia de valor relativo (modo maker): 3 estructuras abiertas como
# máximo por modelo, 1 unidad, payoff acotado, vencimientos entre 21 y 75 días.
.venv/bin/python -m ita_options.pipeline paper-run --live \
    --strategy rv_compite --cap-per-model 3 --contracts 1 --max-open 6 \
    --min-edge 10 --take-profit 0.6 --fill-timeout 180 \
    --kill-switch 250 --daily-loss 1000 \
    --min-dte 21 --max-dte 75 --cycle-seconds 120 \
    --hedge delta_gamma --hedge-band 10 --hedge-max-contracts 5 >> "$log" 2>&1
echo "$(date -Is) fin de la corrida (código $?)" >> "$log"
