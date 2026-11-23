# Estado del proyecto — Grupo 3 (Defensa / ITA)

Última actualización: 23 de noviembre de 2026. Rama `correcciones-parte1`.
Subyacentes: **RTX**, **BA** y **LMT**.

## Cómo correrlo

Requiere Python 3.11 o superior. Desde esta carpeta:

```powershell
pip install -e ".[dev]"
python -m pytest                          # sin red ni credenciales
python scripts/check_informe.py           # números del informe contra reportes_dolthub/
```

Con red (DoltHub y Yahoo):

```powershell
python scripts/download_dolthub.py --start 2026-03-02 --end 2026-11-20   # ~20 min
python scripts/run_dolthub_analysis.py --out reportes_dolthub
python scripts/run_calibration.py --out reportes_dolthub
python scripts/run_rv_sensitivity.py --out reportes_dolthub
```

## Estado por etapa de la consigna

| Etapa | Estado |
|---|---|
| Literatura | Informe, sección 8 |
| Datos | DoltHub: 181 ruedas con bid/ask observado (mar-nov 2026) |
| Limpieza y calidad | Filtros por spread, desfase de spot medido, dividendos point-in-time |
| Calibración binomial | Sonrisa paramétrica; error de repricing dentro y fuera de muestra |
| Detección | Cinco detectores model-free; paridad con dividendo discreto, fondeo y banda de spot |
| Backtest | Contra bid/ask observado, con carry y asignación anticipada: 0 operaciones de arbitraje |
| IS/OOS | Walk-forward de la estrategia de valor relativo (CRR y BSM), julio a noviembre |
| Paper trading | `paper-run --strategy rv_crr`; diagnóstico OK el 22/11; primera rueda: 23/11 |
| Monitoreo y stop-loss | Loop de rueda completa con reconciliación, kill-switch, pérdida diaria y corte manual |
| AI log | `../AI_LOG.md` (cada integrante agrega sus interacciones) |

## Pendiente

- Correr `paper-run` en cada rueda y sumar el P&L acumulado al informe.
- Presentación de la Parte 2 con los números de este informe (`check_informe.py`
  los valida).
- Parte 2: Black-Scholes en producción contra el binomial, greeks y Monte Carlo.

## Paper trading: corrida diaria

Las claves van en `.env` (no se versiona). Arrancar antes de la apertura
(11:30 hora argentina hasta el cambio de horario de EE.UU.); el loop espera la
apertura y termina después del cierre:

```powershell
python -m ita_options.pipeline paper-run --live --strategy rv_crr --contracts 1 --max-open 6 `
    --min-edge 10 --take-profit 0.6 --fill-timeout 180 --kill-switch 250 --daily-loss 1000 `
    --min-dte 21 --max-dte 75 --cycle-seconds 120
```

Corte manual: crear un archivo `KILL` en esta carpeta.
