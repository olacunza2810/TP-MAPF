# TP MAPF 2026 — Grupo 3, Defensa (ETF de referencia: ITA)

Matemática Aplicada a Problemas Financieros, Universidad de San Andrés.
Agustín Herrera, Octavio Lacunza, Carolina Grunewald, Delfina González y Tiziana Brisighelli.

Sistema de punta a punta para valuar, detectar, operar y cubrir opciones sobre
**RTX**, **BA** y **LMT**, acciones individuales del ITA:

- **Parte 1 (tiempo discreto).** Binomial americano (CRR) calibrado a la
  sonrisa, detectores de arbitraje, backtest sin lookahead contra bid/ask
  observado, walk-forward in-sample / out-of-sample y paper trading con
  monitoreo y kill-switch.
- **Parte 2 (tiempo continuo).** Black-Scholes compitiendo contra el binomial
  en la misma cuenta de paper, dividendos discretos, coberturas con griegas,
  estrés por Monte Carlo (GBM, Heston y Merton correlacionados), una simulación
  histórica del sistema completo de 2020 a 2026 y una decisión final con
  criterios fijados de antemano.

## Qué hay en este repositorio

El código y los resultados reproducibles. El informe consolidado (que incluye el
registro de uso de IA) y las presentaciones se entregan por separado.

| Qué | Dónde |
|---|---|
| Código | [`ita_options_pipeline/`](ita_options_pipeline/) (ver su [README](ita_options_pipeline/README.md)) |
| Resultados de la Parte 2 (CSV y gráficos) | [`ita_options_pipeline/reportes_dolthub/`](ita_options_pipeline/reportes_dolthub/), [`reportes_replay/`](ita_options_pipeline/reportes_replay/) |
| Resultados de la Parte 1 (grilla sobre Polygon, histórico) | [`ita_options_pipeline/reportes_final/`](ita_options_pipeline/reportes_final/) |
| Estado por etapa de la consigna | [`ita_options_pipeline/ESTADO.md`](ita_options_pipeline/ESTADO.md) |

## Resultados principales

| Pregunta | Respuesta | Evidencia |
|---|---|---|
| ¿Hay arbitraje con precios reales? | No. Con los supuestos de la Parte 1 (dividendo continuo, spot del cierre), la cadena observada de 2026 muestra 1.744 violaciones de paridad; con dividendo discreto y spot sincronizado bajan a 3, y ninguna es ejecutable | `reportes_dolthub/arbitraje_*` |
| ¿Qué modelo valúa mejor? | El binomial, con significancia alta ($DM^* = 13{,}1$ sobre 36.515 contratos) y donde la teoría lo predice (puts ITM, fechas ex); pero la diferencia supera medio spread sólo en el 0,3% de los casos | `reportes_dolthub/comparacion_*` |
| ¿Qué cobertura conviene? | Delta con acciones: 95% de efectividad neta por USD 3 por posición; cubrir la gamma con opciones no paga sus costos | `reportes_dolthub/cobertura_*` |
| ¿Qué pasa bajo estrés? | La cobertura delta baja el ES 99% entre 25% y 45%; el riesgo dominante es la liquidez (USD 708 para cerrar todo) | `reportes_dolthub/estres_*` |
| ¿Cómo le habría ido al sistema completo de 2020 a 2026? | Pierde unos USD 100 por estructura todos los años, por selección adversa; sin diferencia entre modelos ($z = -0{,}21$) | `ita_options_pipeline/reportes_replay/` |
| Decisión | Binomial y cobertura delta | `reportes_dolthub/scorecard.json` |

## Correcciones de la entrega parcial

| Devolución | Qué se hizo | Código |
|---|---|---|
| 1. Bid/ask real | Foto diaria de DoltHub con bid/ask observado: 181 ruedas de 2026 y 1.229 desde 2020 | `scripts/download_dolthub.py`, `ita_options/dolthub.py` |
| 2. Paridad con dividendos discretos y fondeo | Cotas americanas con dividendos point-in-time, carry y asignación anticipada; corregido el signo del préstamo de la acción | `ita_options/arbitrage.py`, `ita_options/backtest.py` |
| 3. Paper trading | Valor relativo contra el modelo en verticales y butterflies (pérdida acotada, 1 unidad), con los dos modelos compitiendo | `ita_options/execution/model_rv.py` |
| 4. Walk-forward IS/OOS | Calibración en marzo–junio y evaluación mes a mes de julio a noviembre, con Sharpe deflactado | `ita_options/rv_backtest.py` |
| 5. Ajuste de la calibración | Sonrisa paramétrica por vencimiento; error de repricing dentro y fuera de muestra | `scripts/run_calibration.py` |
| 6. Monitoreo como proceso | Loop de rueda completa: marca, reconcilia con el broker y aplica kill-switch, pérdida diaria y corte manual | `ita_options/execution/live_runner.py` |
| 7. Una sola versión | Informe consolidado reescrito con los números vigentes (se entrega aparte) | — |

## Cómo correrlo

Requiere Python 3.11 o superior. Desde `ita_options_pipeline/`:

```bash
pip install -e ".[dev,analisis]"
python -m pytest                    # 191 tests, sin red ni credenciales
ita-options demo                    # demo offline sobre un mercado sintético
```

La tabla para reproducir cada resultado y la configuración del paper trading
están en el [README del paquete](ita_options_pipeline/README.md).

## Paper trading en producción

Desde el 23 de noviembre de 2026 el sistema opera solo contra la cuenta paper
de Alpaca en una VM gratuita de Oracle Cloud, de lunes a viernes: los dos
modelos compiten, con cobertura delta-gamma, kill-switch de USD 250, pérdida
diaria de USD 1.000 y corte manual con un archivo `KILL`. Instalación y
operación: [`ita_options_pipeline/deploy/oracle/README.md`](ita_options_pipeline/deploy/oracle/README.md).
