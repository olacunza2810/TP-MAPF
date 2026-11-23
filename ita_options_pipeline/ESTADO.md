# Estado del proyecto — Grupo 3 (Defensa / ITA)

Última actualización: 23 de noviembre de 2026. Rama `main`.
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
python scripts/run_model_comparison.py                       # Parte 2, paso 3 (~2 min)
python scripts/run_model_comparison.py --spot underlying_price   # robustez con el cierre
python scripts/run_hedging.py                                # Parte 2, paso 4 (~3 min)
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

## Parte 2

| Paso | Estado |
|---|---|
| 1. Competencia CRR vs BSM en paper | `--strategy rv_compite`; vivo = backtest verificado sobre datos reales |
| 2. Griegas de los dos modelos | `ita_options/greeks.py`, dividendo discreto escrowed; Hull 4,44 reproducido |
| 3. BS contra binomial en DoltHub | `run_model_comparison.py`: prima de ejercicio, repricing, Diebold-Mariano |
| 4. Cobertura con griegas | `run_hedging.py` (DoltHub) y `--hedge delta` en paper: un libro de acciones por modelo |
| 5. Monte Carlo de estrés | pendiente |
| 6. Scorecard, informe y presentación | pendiente |

Paso 3, en una línea por resultado (spot implícito, 181 ruedas, 36.515 contratos):

- Diebold-Mariano en medios spreads: el binomial erra menos que BS
  (d medio = 0,037, DM = 13,1); en las OTM del ajuste no hay diferencia
  (p = 0,54); en calls sin fecha ex tampoco (p = 0,53, Merton 1973).
- La prima de ejercicio supera medio spread en 0,3% de los contratos: el modelo
  casi nunca cambia si un precio cae dentro del bid-ask (98,1% CRR, 97,6% BS).
- En el universo de la estrategia (|k| <= 10%, 21-75 días) la brecha CRR-BS
  supera los USD 10 por contrato en 30% de los casos; BS subvalúa las puts ITM
  en USD 19 por contrato en promedio.
- Dividendo discreto contra yield continuo: el escrowed erra menos (DM = 6,1); el
  yield continuo subvalúa calls ITM en USD 16 por contrato. **Aplicado:** la
  estrategia RV (vivo, backtest y calibración) valúa con el árbol escrowed
  (`ita_options/escrowed.py`, columna `dividend_schedule`). Walk-forward,
  sensibilidad y calibración recalculados; el informe se actualizó y las
  conclusiones no cambian (DSR máximo 0,21; las 36 configuraciones pierden).

Paso 4 (1.749 posiciones vendidas, 3.272 días-posición, rebalanceo diario):

- Efectividad $1-\mathrm{Var}(cubierta)/\mathrm{Var}(sin\ cubrir)$, bruta y neta de costos:
  delta 95% y 95%; delta-gamma 96% y 79%; delta-vega 94% y 63%.
- Armar y desarmar la cobertura con opciones cuesta USD 133 (gamma) y 188 (vega)
  por posición, contra USD 3 con acciones. Se elige **delta con acciones**.
- Binomial y BS dan la misma cobertura (diferencia media de delta 0,004;
  Diebold-Mariano sin diferencia significativa).
- En paper: `--hedge delta` cubre cada libro (modelo, subyacente) con acciones,
  banda de 10 acciones; la gamma se informa en cada ciclo pero no se cubre.

## Pendiente

- Correr `paper-run` en cada rueda y sumar el P&L acumulado al informe.
- Pasos 5 y 6 de la Parte 2.

## Paper trading: corrida diaria

Las claves van en `.env` (no se versiona). Arrancar antes de la apertura
(11:30 hora argentina hasta el cambio de horario de EE.UU.); el loop espera la
apertura y termina después del cierre:

```powershell
python -m ita_options.pipeline paper-run --live --strategy rv_compite --cap-per-model 3 --contracts 1 --max-open 6 `
    --min-edge 10 --take-profit 0.6 --fill-timeout 180 --kill-switch 250 --daily-loss 1000 `
    --min-dte 21 --max-dte 75 --cycle-seconds 120 --hedge delta --hedge-band 10
```

Corte manual: crear un archivo `KILL` en esta carpeta.
