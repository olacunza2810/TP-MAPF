#!/usr/bin/env bash
# Prepara una VM Ubuntu (Oracle Cloud Always Free) para correr el paper trading.
#
#     bash setup_vm.sh
#
# Hace: paquetes del sistema, zona horaria de Nueva York, swap (para la VM de
# 1 GB), clon del repo, entorno virtual, dividendos semilla, tests y cron.
# No toca las claves: el .env lo creás vos después (ver README.md).
set -euo pipefail

REPO_URL="https://github.com/olacunza2810/TP-MAPF.git"
HOME_DIR="$HOME"
REPO="$HOME_DIR/TP-MAPF"
APP="$REPO/ita_options_pipeline"

echo "== Paquetes del sistema"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y git python3 python3-venv python3-pip cron util-linux

echo "== Zona horaria (cron y logs en hora de Nueva York)"
sudo timedatectl set-timezone America/New_York

echo "== Swap de 2 GB (la VM chica tiene 1 GB de RAM)"
if ! swapon --show | grep -q /swapfile; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab > /dev/null
fi

echo "== Repo"
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" pull --ff-only
else
    git clone "$REPO_URL" "$REPO"
fi

echo "== Entorno virtual"
cd "$APP"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"

echo "== Dividendos semilla (para el modelo con dividendo discreto)"
mkdir -p data/dolthub data/paper
cp deploy/oracle/seed/dividends.parquet deploy/oracle/seed/dividends_polygon.parquet data/dolthub/

echo "== Tests"
.venv/bin/python -m pytest -q

echo "== Cron: lunes a viernes 9:20 (Nueva York)"
chmod +x deploy/oracle/paper_daily.sh
line="20 9 * * 1-5 $APP/deploy/oracle/paper_daily.sh"
# Sin crontab previo, `crontab -l` y `grep` salen con error: con pipefail eso
# cortaba el script en silencio, así que se toleran.
{ crontab -l 2>/dev/null | grep -v "paper_daily.sh" || true; echo "$line"; } | crontab -
sudo systemctl enable --now cron
crontab -l

cat <<EOF

Listo. Falta un solo paso: crear $APP/.env con tus claves de Alpaca PAPER:

    nano $APP/.env

con estas tres líneas (sin comillas):

    ALPACA_API_KEY=...
    ALPACA_SECRET_KEY=...
    ALPACA_PAPER=true

y después:

    chmod 600 $APP/.env
    cd $APP && .venv/bin/python -m ita_options.pipeline paper-run --strategy rv_compite --max-cycles 1

(sin --live: es una prueba en seco que sólo verifica la conexión).
EOF
