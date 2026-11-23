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
python scripts/run_stress.py                                 # Parte 2, paso 5 (~11 min)
python scripts/scorecard.py                                  # Parte 2, paso 6
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
| 4. Cobertura con griegas | `run_hedging.py` (DoltHub) y `--hedge delta_gamma` en paper: un libro por modelo y subyacente |
| 5. Monte Carlo de estrés | `run_stress.py`: cartera en producción, 6 escenarios MC (5.000 trayectorias) y 5 determinísticos |
| 6. Scorecard, informe y presentación | `scorecard.py` (criterios fijados antes del paper), informe consolidado (22 págs., 90 números verificados), AI log, slides de la Parte 2 |

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
  delta 95% y 95%; delta-gamma 96% y 79%; delta-gamma entera (sólo compras, la
  de paper) 95% y 73%; delta-vega 94% y 63%.
- Armar y desarmar la cobertura con opciones cuesta USD 133 (gamma) y 188 (vega)
  por posición, contra USD 3 con acciones. Se elige **delta con acciones**.
- Binomial y BS dan la misma cobertura (diferencia media de delta 0,004;
  Diebold-Mariano sin diferencia significativa).
- En paper: `--hedge delta_gamma` compra la call ATM corta para la gamma de cada
  libro (contratos enteros, sólo compras: nivel 3 no permite vender descubierto)
  y cubre la delta restante con acciones, banda de 10 acciones. `--hedge delta`
  deja sólo las acciones. Por evidencia, la cobertura recomendada es delta; la
  delta-gamma corre en paper porque la consigna pide las dos griegas en producción
  y así se mide su costo real.

Paso 5 (cartera a capacidad del 20/11: 3 estructuras por modelo, 10 ruedas):

- ES 99% al mid sin cobertura / con delta: GBM 1.066 / 589; Heston 1.024 / 594;
  Heston estresado 1.029 / 648; Merton 1.000 / 741; Merton sectorial 902 / 557;
  spreads x3 1.139 / 832 USD. La delta-gamma entera da lo mismo que la delta
  (la gamma redondeada casi nunca llega a 1 contrato), salvo con spreads x3,
  donde comprar gamma la empeora (1.873).
- El riesgo dominante es la liquidez: cerrar todo cruzando el spread cuesta
  USD 708 en condiciones normales y 2.443 con spreads x3, más que cualquier
  escenario de mercado. Las estructuras son de riesgo acotado: ni la caída de
  15% ni la repetición del COVID disparan la pérdida diaria.
- **Corrección en producción:** el kill-switch y la pérdida diaria se medían a
  liquidación; una estructura recién abierta ya "perdía" el spread y el
  kill-switch saltaba en el primer ciclo (probabilidad 100% en el estrés, P&L
  esperado -846 contra -278). Ahora se miden al mid, como en el backtest; el
  take-profit sigue a liquidación.
- Heston de clase: el notebook difunde el precio con `Var[-1]` (la última fila,
  que vale v0), así que la vol estocástica nunca llega al precio. Corregido acá
  (y con truncamiento completo); un test lo verifica.

## Pendiente

- Correr `paper-run` en cada rueda y sumar el P&L acumulado al informe.
- Correr el paper (`scripts/paper_daily.ps1`) y completar con el P&L acumulado: la slide de paper, la sección 12 del informe y el criterio M4 (`scorecard.py --ledger data/paper/ledger.sqlite`).

## Paper trading: corrida diaria

Las claves van en `.env` (no se versiona). Arrancar antes de la apertura
(11:30 hora argentina hasta el cambio de horario de EE.UU.); el loop espera la
apertura y termina después del cierre:

```powershell
python -m ita_options.pipeline paper-run --live --strategy rv_compite --cap-per-model 3 --contracts 1 --max-open 6 `
    --min-edge 10 --take-profit 0.6 --fill-timeout 180 --kill-switch 250 --daily-loss 1000 `
    --min-dte 21 --max-dte 75 --cycle-seconds 120 --hedge delta_gamma --hedge-band 10 --hedge-max-contracts 5
```

Corte manual: crear un archivo `KILL` en esta carpeta.
