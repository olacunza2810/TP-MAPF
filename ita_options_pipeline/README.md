# `ita_options` — paquete del Grupo 3 (Defensa / ITA)

Valuación, detección, backtest, coberturas, estrés y ejecución en paper trading
de opciones sobre **RTX**, **BA** y **LMT**, acciones individuales del ETF ITA.
Este README describe el código; los resultados y su interpretación están en
[`informe_proyecto.md`](informe_proyecto.md) y en
[`resumen_presentacion/`](resumen_presentacion/).

## Instalación

Requiere Python 3.11 o superior.

```bash
pip install -e ".[dev]"             # paquete + pytest
pip install -e ".[dev,analisis]"    # + matplotlib y lo necesario para regenerar gráficos y el PDF del informe
python -m pytest                    # 191 tests, sin red ni credenciales (~30 s)
ita-options demo                    # pipeline completo sobre un mercado sintético, offline
```

Si el comando `ita-options` no está en el PATH, usar
`python -m ita_options.pipeline <subcomando>`.

## Mapa del código

```
ita_options/
├── Datos
│   ├── dolthub.py          Curación de la foto diaria de DoltHub: spot implícito, dividendos point-in-time
│   ├── clients.py          Acceso asincrónico a Alpaca (rate limit y reintentos)
│   ├── ingest.py           Universo, grabación point-in-time y backfill
│   ├── enrich.py           Alineación anti-lookahead, mid/spread, DTE, IV
│   ├── daily.py            Modo diario sobre velas EOD de Polygon (Parte 1, histórico)
│   ├── filters.py          Filtros de liquidez con reporte auditable
│   ├── schemas.py          Contrato de columnas y tipos
│   └── storage.py          Parquet particionado y manifiesto reproducible
├── Valuación
│   ├── volatility.py       Black-Scholes, árbol CRR americano e inversión de IV
│   ├── greeks.py           BS y CRR con dividendo discreto escrowed, y sus griegas
│   └── escrowed.py         Precio e IV con dividendo discreto para toda la cadena
├── Estrategia y backtest
│   ├── arbitrage.py        Cinco detectores model-free y uno contra el modelo
│   ├── backtest.py         Motor event-driven con barrera anti-lookahead
│   ├── rv_backtest.py      Backtest y walk-forward de la estrategia de valor relativo
│   └── evaluation.py       Métricas, split IS/OOS y Sharpe deflactado
├── Parte 2
│   ├── model_comparison.py BS contra binomial: prima de ejercicio y Diebold-Mariano (paso 3)
│   ├── hedging.py          Backtest de coberturas con griegas (paso 4)
│   ├── stress.py           Monte Carlo GBM / Heston / Merton correlacionado (paso 5)
│   └── replay.py           Simulación histórica 2020–2026 del sistema completo
├── execution/              Paper trading en Alpaca
│   ├── live_runner.py      Loop de la rueda: datos, señales, ejecución, cobertura y riesgo
│   ├── model_rv.py         Señales de valor relativo y competencia entre modelos (rv_compite)
│   ├── router.py           Órdenes mleg: espera bloqueante o maker publicada (--rest-seconds)
│   ├── hedger.py           Cobertura delta y delta-gamma en vivo, un libro por modelo
│   ├── ledger.py           Registro SQLite de estrategias, órdenes y eventos; reconciliación
│   ├── risk.py             Controles previos al envío
│   ├── orders.py           Construcción de órdenes multipata
│   ├── parity.py           Secuencia de la paridad put-call en patas separadas
│   ├── live_data.py        Universo, snapshots y cadena curada en vivo
│   └── broker.py           Interfaz mínima del broker
├── calibration.py, demo.py Mercado sintético y demo offline
├── doctor.py               Diagnóstico de conectividad, permisos y datos
├── config.py               Configuración tipada
└── pipeline.py             CLI (ita-options)
```

Las capas de valuación y estrategia no dependen de la red: por eso los
detectores, los árboles, las coberturas y el loop en vivo (contra un broker
simulado, `tests/fake_broker.py`) se testean sin credenciales.

## Reproducir los resultados

Cada número del informe sale de un script. Con red (DoltHub y Yahoo):

| Paso | Comando | Salida |
|---|---|---|
| Datos 2026 (~20 min) | `python scripts/download_dolthub.py --start 2026-03-02 --end 2026-11-20` | `data/dolthub/` |
| Curación, detectores, backtest y walk-forward | `python scripts/run_dolthub_analysis.py --out reportes_dolthub` | `reportes_dolthub/` |
| Calibración | `python scripts/run_calibration.py --out reportes_dolthub` | `calibracion_*` |
| Sensibilidad de la estrategia | `python scripts/run_rv_sensitivity.py --out reportes_dolthub` | `rv_sensibilidad_*` |
| BS contra binomial | `python scripts/run_model_comparison.py` | `comparacion_*` |
| Coberturas | `python scripts/run_hedging.py` | `cobertura_*` |
| Estrés Monte Carlo (~11 min) | `python scripts/download_history.py` y `python scripts/run_stress.py` | `estres_*` |
| Tabla de decisión | `python scripts/scorecard.py` | `scorecard.json` |
| Historia 2020–2026 (~70 min) | `python scripts/download_dolthub.py --start 2020-01-02 --end 2026-11-20 --data-root data_hist --workers 10` | `data_hist/` |
| Curación de la historia | `python scripts/run_dolthub_analysis.py --data-root data_hist --steps curate --out reportes_replay` | `data_hist/dolthub/curated.parquet` |
| Simulación histórica (~15 min) | `python scripts/run_replay.py --data-root data_hist --out reportes_replay` | `reportes_replay/` |
| Control del informe | `python scripts/check_informe.py` | verifica 117 números del informe contra `reportes_dolthub/` y `reportes_replay/` |
| Gráficos y PDF del resumen | `python resumen_presentacion/hacer_graficos.py` y, en esa carpeta, `latexmk -xelatex -jobname=Resumen_Presentacion_Grupo3 resumen.tex` | `resumen_presentacion/` |

## Paper trading

Las claves de Alpaca **paper** van en `.env` (excluido de git):

```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_PAPER=true
```

```bash
ita-options doctor                                    # conectividad, nivel de opciones y datos
ita-options paper-run --strategy rv_compite           # prueba en seco: no envía órdenes
```

La configuración de producción está en
[`deploy/oracle/paper_daily.sh`](deploy/oracle/paper_daily.sh) (Linux) y
[`scripts/paper_daily.ps1`](scripts/paper_daily.ps1) (Windows): los dos modelos
compiten (`rv_compite`), cobertura delta-gamma, kill-switch de USD 250 al mid,
pérdida diaria de USD 1.000 y órdenes maker publicadas hasta 30 minutos.

Corre sola en una VM gratuita de Oracle Cloud, de lunes a viernes a las 9:20
de Nueva York. Instalación, operación y cómo traer los resultados:
[`deploy/oracle/README.md`](deploy/oracle/README.md). Para cortar todo, crear
un archivo `KILL` en esta carpeta.

## Otros archivos

| Archivo | Qué es |
|---|---|
| `informe_proyecto.md` / `.pdf` | Informe consolidado del semestre. El PDF se arma con pandoc y XeLaTeX (`pandoc informe_proyecto.md -o informe_proyecto.pdf --pdf-engine=xelatex -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono" -V mathfont="Latin Modern Math" -V colorlinks=true`); `generate_report_pdf.py` e `informe_proyecto.css` son una alternativa sin LaTeX, a partir de la versión HTML |
| `ESTADO.md` | Estado por etapa de la consigna y pendientes |
| `resumen_presentacion/` | Resumen de 33 páginas para armar la presentación (LaTeX) |
| `reportes_dolthub/`, `reportes_replay/` | Resultados vigentes (CSV y gráficos) |
| `docs/notas_tecnicas_parte1.md` | Notas técnicas de septiembre: controles de sesgo, IV, filtros, detectores y piloto |
| `validate_sample.py`, `justificacion_generacion_datos.md`, `outputs/` | **Históricos (Parte 1):** validación de la muestra sintética de la demo |
| `reportes/paper_probe_*.json` | **Histórico (Parte 1):** prueba de órdenes del 14/09 |
