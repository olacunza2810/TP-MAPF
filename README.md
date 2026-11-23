# TP MAPF 2026 — Grupo 3, Defensa (ETF de referencia: ITA)

Matemática Aplicada a Problemas Financieros, Universidad de San Andrés.
Agustín Herrera, Octavio Lacunza, Carolina Grunewald, Delfina González y Tiziana Brisighelli.

Pipeline de punta a punta para detectar y operar inconsistencias de precios en
opciones sobre **RTX**, **BA** y **LMT**, constituyentes individuales del ITA:
datos con bid/ask observado, calibración del binomial americano (CRR),
detectores de arbitraje, backtest sin lookahead, walk-forward in-sample /
out-of-sample y ejecución en paper trading (Alpaca) con monitoreo y kill-switch.

## Dónde está cada cosa

| Qué | Dónde |
|---|---|
| Informe consolidado (fuente de los números) | [`ita_options_pipeline/informe_proyecto.md`](ita_options_pipeline/informe_proyecto.md) |
| Resultados reproducibles (CSV y gráficos) | [`ita_options_pipeline/reportes_dolthub/`](ita_options_pipeline/reportes_dolthub/) |
| Código del paquete | [`ita_options_pipeline/ita_options/`](ita_options_pipeline/ita_options/) |
| Scripts de descarga y análisis | [`ita_options_pipeline/scripts/`](ita_options_pipeline/scripts/) |
| Tests | [`ita_options_pipeline/tests/`](ita_options_pipeline/tests/) |
| Registro de uso de IA | [`AI_LOG.md`](AI_LOG.md) |
| Estado y traspaso | [`ita_options_pipeline/ESTADO.md`](ita_options_pipeline/ESTADO.md) |
| Entrega parcial de septiembre (histórica) | [`entrega_parcial_sep/`](entrega_parcial_sep/) |

## Correcciones de la entrega parcial

| Devolución | Qué se hizo | Código | Resultado |
|---|---|---|---|
| 1. Bid/ask real | Descarga de la foto diaria de DoltHub (mar-nov 2026, 181 ruedas, 95.290 filas) y backtest contra bid/ask observado | `scripts/download_dolthub.py`, `ita_options/dolthub.py` | `reportes_dolthub/arbitraje_*` |
| 2. Paridad con dividendos discretos y fondeo | Cotas americanas $S-D-K \le C-P \le S-Ke^{-r_b\tau}$, dividendos point-in-time por fecha de declaración, carry y asignación anticipada en el backtest; corregido el signo del préstamo de la acción | `ita_options/arbitrage.py`, `ita_options/backtest.py` | `tests/test_parity.py`, `tests/test_dolthub_carry.py` |
| 3. Paper trading | Estrategia de valor relativo contra el binomial calibrado en verticales y butterflies (payoff acotado, orden multipata límite, 1 unidad) | `ita_options/execution/model_rv.py`, `paper-run --strategy rv_crr` | `reportes/paper/` |
| 4. Walk-forward IS/OOS | Calibración en mar-jun y evaluación mes a mes de julio a noviembre, con corte limpio entre muestras | `ita_options/rv_backtest.py`, `scripts/run_dolthub_analysis.py` | `reportes_dolthub/rv_walkforward.csv` |
| 5. Ajuste de la calibración | Sonrisa paramétrica por vencimiento, IV contra moneyness y plazo, error de repricing in-sample y fuera de muestra | `scripts/run_calibration.py` | `reportes_dolthub/calibracion_rmse.csv`, `iv_vs_*.png` |
| 6. Monitoreo como proceso | `paper-run` corre toda la rueda: marca, reconcilia el ledger con el broker y aplica kill-switch, pérdida diaria y corte manual sobre posiciones reales | `ita_options/execution/live_runner.py` | `reportes/paper/session_*` |
| 7. Una sola versión | Informe reescrito con los números de `reportes_dolthub/`; la entrega parcial quedó archivada aparte | `informe_proyecto.md` | — |

## Cómo reproducir

Requiere Python 3.11 o superior. Desde `ita_options_pipeline/`:

```bash
pip install -e ".[dev]"
python -m pytest                                         # 120 tests, sin red ni credenciales

# 1. Datos (red: DoltHub y Yahoo; ~20 minutos)
python scripts/download_dolthub.py --start 2026-03-02 --end 2026-11-20

# 2. Curación, detectores, backtest de arbitraje, valuaciones y walk-forward
python scripts/run_dolthub_analysis.py --out reportes_dolthub

# 3. Calibración (punto 5) y sensibilidad exploratoria
python scripts/run_calibration.py --out reportes_dolthub
python scripts/run_rv_sensitivity.py --out reportes_dolthub
```

## Paper trading

Las claves de Alpaca **paper** van en un archivo `ita_options_pipeline/.env`
(está en `.gitignore`):

```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_PAPER=true
```

Diagnóstico y corrida de una rueda completa (arrancarla antes de la apertura;
termina sola después del cierre):

```bash
python -m ita_options.pipeline doctor
python -m ita_options.pipeline paper-run --live --strategy rv_compite --cap-per-model 3 --contracts 1 --max-open 6 \
    --min-edge 10 --take-profit 0.6 --fill-timeout 180 --kill-switch 250 --daily-loss 1000 \
    --min-dte 21 --max-dte 75 --cycle-seconds 120
```

Sin `--live` el loop sólo registra las órdenes que habría enviado. Crear un
archivo `KILL` en la carpeta detiene las entradas y cierra todo en el ciclo
siguiente.
