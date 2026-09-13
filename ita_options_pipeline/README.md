# Pipeline de datos de opciones — Grupo 3 (Defensa / ITA)

Extracción, alineación y curado de un dataset de opciones sobre **RTX** y **BA**
para alimentar un backtest de arbitraje binomial sin sesgos.

```
ita_options/
├── config.py       Configuración tipada + manifiesto reproducible
├── schemas.py      Contrato de columnas y dtypes
├── volatility.py   BSM + árbol CRR americano + inversión de IV
├── clients.py      Gateway asincrónico sobre alpaca-py (rate limit + retries)
├── ingest.py       Universo, recorder point-in-time, backfill de bars
├── enrich.py       Alineación anti-lookahead, mid/spread, DTE, IV
├── filters.py      Filtros de liquidez con reporte auditable
├── storage.py      Parquet particionado (underlying / trade_date)
├── arbitrage.py    Seis detectores: cinco model-free + uno contra modelo
├── backtest.py     Motor event-driven con barrera anti-lookahead
├── evaluation.py   Métricas, split in-sample/out-of-sample, Sharpe deflactado
├── doctor.py       Diagnóstico de conectividad y datos disponibles
└── pipeline.py     Orquestador + CLI
tests/              25 tests, incluidos tests de contrato contra el SDK real
```

El grafo de dependencias es acíclico y tiene cuatro capas: `volatility`,
`config` y `schemas` no dependen de nada; `arbitrage` depende sólo de
`volatility`; `backtest` de `arbitrage`; `evaluation` de `backtest`. Por eso los
detectores y el árbol se pueden testear sin credenciales ni red.

## Instalación y uso

Requiere Python 3.11 o superior.

```bash
pip install -e .                    # instala el paquete y el comando ita-options
pip install -e ".[dev]"             # con pytest, para correr los tests

# Windows PowerShell
$env:ALPACA_API_KEY="..."; $env:ALPACA_SECRET_KEY="..."
# macOS / Linux
export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...
```

```bash
ita-options doctor                  # EMPEZAR ACÁ: diagnostica en 2 segundos
ita-options universe                # maestro de contratos
ita-options detect --min-edge 5     # arbitrajes sobre la cadena en vivo
ita-options record --interval 300   # grabación point-in-time (dejar corriendo)
ita-options backfill --start 2025-09-01 --end 2026-09-01
ita-options enrich --start 2026-09-01

pytest                              # 25 tests, sin red ni credenciales
```

`doctor` corre los seis chequeos en el orden en que el pipeline los necesita y
se detiene en el primero que falla. Cada chequeo que pasa reporta una medición:
cantidad de contratos, cobertura de open interest, porcentaje con NBBO real y
**spread relativo mediano**, que es el número que determina si la estrategia es
viable.

---

## Limitación estructural de Alpaca (leer antes de diseñar el backtest)

**Alpaca no expone quotes históricos de opciones.** El
`OptionHistoricalDataClient` ofrece bars, trades, *latest* quote, snapshot y
chain. El endpoint de chain devuelve bid/ask, IV y griegas, pero es un snapshot
vivo: no acepta parámetro de fecha y no permite reconstruir la superficie de un
día pasado.

Consecuencias, y cómo las resuelve este pipeline:

| Necesidad | Disponible en Alpaca | Estrategia |
|---|---|---|
| Bid/ask histórico | No | `SnapshotRecorder` graba el chain cada N segundos y construye la serie hacia adelante |
| Historia previa de precios de opción | Bars OHLCV por contrato | `HistoricalBackfill`, con spread estimado y marcado `spread_is_proxy=True` |
| Open interest | Endpoint de contratos, con lag de 1 día (cálculo EOD de OCC) | `attach_lagged_open_interest` anula el OI cuya fecha de publicación no es anterior al quote |
| Universo histórico de contratos | Sólo el vigente | Maestro append-only diario en `contract_master/` |

**La implicancia operativa es que el recorder debería estar corriendo ya.** Cada
día sin grabar es un día que no vas a tener en la Parte 2, cuando haya que
comparar binomial contra Black-Scholes sobre el mismo dataset.

### Feed

Por defecto `indicative`: gratuito, pero con quotes **modificados y demorados 15
minutos**. Sirve para calibrar y para el backtest; no sirve para arbitraje en
vivo. `opra` requiere Algo Trader Plus. La elección debe declararse en la
presentación: un arbitraje detectado sobre un feed demorado 15 minutos no es
explotable, y los profesores lo van a preguntar.

---

## Controles de sesgo implementados

**Lookahead — alineación de barras.** Alpaca etiqueta las barras con el instante
de *apertura* del intervalo. Una barra rotulada 14:30 cubre hasta 14:31, así que
su `close` recién se conoce a las 14:31. `align_underlying` hace `merge_asof`
contra `available_at = timestamp + bar_duration`, con `direction="backward"` y
tolerancia máxima. Sin ese corrimiento se filtra hasta una barra completa de
futuro, lo que en granularidad de 1 minuto alcanza para inventar arbitrajes.

**Lookahead — precios ajustados.** El subyacente se pide con `adjustment="raw"`.
Los precios ajustados por splits y dividendos se recalculan retroactivamente
(información futura), y además los strikes de los contratos están expresados en
términos no ajustados: alinear una opción contra un spot ajustado da un
moneyness incorrecto.

**Lookahead — open interest.** El OI de la rueda del día `t` se publica después
del cierre. Filtrar operaciones del día `t` con ese número es lookahead. El
merge exige `open_interest_date < trade_date`.

**Survivorship.** El endpoint de contratos devuelve el universo vigente; los
contratos vencidos desaparecen. Reconstruir hoy el universo de hace seis meses
devuelve sólo lo que sobrevivió. La única mitigación real es el maestro
append-only. Los backtests sobre períodos anteriores al inicio de la grabación
arrastran el sesgo y deben reportarse como tales — decirlo explícitamente vale
más que esconderlo.

**Data snooping.** El manifiesto (`storage.write_manifest`) versiona todos los
umbrales y supuestos junto al dataset. Cualquier cambio de hiperparámetro queda
registrado, que es la condición mínima para que el split in-sample /
out-of-sample signifique algo.

---

## Volatilidad implícita

Se invierte

$$f(\sigma) = V_{\text{modelo}}(\sigma) - V_{\text{mercado}} = 0$$

por Brent sobre $\sigma\in[\sigma_{\min}, 5]$. Brent y no Newton-Raphson: $f$ es
monótona creciente en $\sigma$, pero el vega

$$\mathcal{V} = S e^{-q\tau}\phi(d_1)\sqrt{\tau}$$

se anula en las colas, y Newton diverge justamente en los contratos deep OTM que
más interesan para detectar dislocaciones.

Se calculan **dos** IVs por contrato:

- `iv_american` — árbol CRR con ejercicio anticipado. Es la correcta: RTX y BA
  son opciones americanas sobre acciones que pagan (o pagaron) dividendos.
- `iv_european` — BSM cerrado, como control.

La diferencia entre ambas cuantifica la prima de ejercicio anticipado. Reportar
la IV europea sobre una put americana ITM y llamar "arbitraje" a la brecha
resultante es el error clásico de este ejercicio.

$\sigma_{\min}$ es adaptativo: el árbol CRR sólo mantiene $p\in(0,1)$ si
$\sigma > |r-q|\sqrt{\Delta t}$. Abrir la búsqueda por debajo de esa cota hace
fallar el solver sobre contratos válidos.

**Cotas de no arbitraje.** Un mid fuera de
$\max(Se^{-q\tau}-Ke^{-r\tau},\,S-K,\,0)\le C\le S$ es un error de datos o un
arbitraje estático, nunca una IV. El solver devuelve `NaN` y marca
`violates_bounds=True` en vez de forzar una raíz inexistente. Esas filas son
material de presentación, no basura.

---

## Filtros de liquidez

Los filtros **no borran filas en silencio**: marcan cada causa en una columna
`excl_*` y devuelven un `FilterReport` con el conteo por criterio. Esa tabla es
la diferencia entre "aplicamos filtros de liquidez" y una justificación
auditable.

| Criterio | Umbral | Razón |
|---|---|---|
| `volume > 0` | 0 | Sin volumen no hay evidencia de operabilidad |
| `open_interest > 50` | 50 | Sobre OI rezagado |
| `bid > 0` | 0 | La exclusión más importante: con bid cero el mid colapsa a medio ask y genera falsos arbitrajes |
| `(ask-bid)/mid < 0.15` | 0.15 | El spread es el costo de cruzar |
| `spread_abs > 0` | — | Descarta quotes cruzados o bloqueados |
| `mid ≥ 0.05` | 0.05 | Bajo eso el tick de 1 centavo domina la señal |
| `7 ≤ dte ≤ 180` | — | Pin risk abajo, quotes poco confiables arriba |

---

## Sobre la asincronía

Los clientes REST de `alpaca-py` son **sincrónicos**; sólo el websocket
(`OptionDataStream`) es nativo async. Envolverlos en `async def` sin más daría
corrutinas que igual bloquean el event loop — asincronía cosmética. `clients.py`
despacha cada llamada a un thread pool vía `asyncio.to_thread`, coordinada por
un semáforo y un token bucket dimensionado en requests por minuto (190 por
defecto, bajo el límite de 200 del plan gratuito), con reintentos por backoff
exponencial y jitter.

---

## Validación

Los módulos de valuación se verifican contra invariantes conocidos: convergencia
del CRR al europeo sin dividendos ($\Delta < 2.3\times10^{-3}$ con $N=1024$),
paridad put-call europea al nivel de $10^{-8}$, prima de ejercicio anticipado no
negativa, round-trip de IV con error $< 10^{-9}$, y una prueba de alineación con
una barra "envenenada" que verifica que el `merge_asof` no la consuma.

## Pendiente

- Curva de tasas por tenor en vez de $r$ constante (el hook está en
  `PricingAssumptions.risk_free_curve_path`).
- Dividendos discretos en lugar de yield continuo — relevante para RTX cerca de
  la ex-date, donde se concentra el ejercicio anticipado de calls.
- Vencimiento anclado a 20:00 UTC; el huso correcto (16:00 ET) importa sólo en
  contratos de 0-1 DTE, que igual quedan fuera por el filtro de `min_dte`.
- Paralelización por proceso de la inversión de IV si el dataset crece.


---

## Detección de arbitrajes

Seis detectores en `arbitrage.py`. Los cinco primeros son **model-free**: se
cumplen por no arbitraje puro y valen aunque el binomial esté mal calibrado.

| Detector | Relación testeada |
|---|---|
| `monotonicity` | $K_1 < K_2 \Rightarrow C(K_1) \ge C(K_2)$ |
| `vertical_bound` | $C(K_1) - C(K_2) \le K_2 - K_1$ |
| `butterfly` | $C(K_2) \le w\,C(K_1) + (1-w)\,C(K_3)$ |
| `put_call_parity` | $Se^{-q\tau} - K \le C_A - P_A \le S - Ke^{-r\tau}$ (banda americana) |
| `calendar` | $T_2 > T_1 \Rightarrow C(T_2,K) \ge C(T_1,K)$ |
| `model_dislocation` | Precio CRR contra el lado de mercado que se cruzaría |

Todos usan bid y ask, nunca el mid ni el `lastPrice`: se compra al ask y se
vende al bid, así el spread queda internalizado en el planteo. Cada oportunidad
reporta el **edge neto en USD** después de comisiones, y la liquidez de su peor
pata.

Los detectores exigen la columna `is_tradable`. Si falta, levantan `KeyError` en
vez de saltear el filtro: un filtro silencioso es peor que un filtro ausente.

## Backtest

`backtest.py` separa el instante de detección $t_d$ del de ejecución
$t_e = t_d + \ell$, y **revalúa las patas con las cotizaciones de $t_e$**. Si el
crédito se evaporó, la operación no se hace. La sensibilidad a $\ell$ es el
resultado más informativo: sobre los datos de prueba, pasar de 5 a 20 minutos de
latencia da vuelta el P&L de +756 a −143 USD.

`PointInTimeView` es la única puerta a los datos durante el loop y levanta
`LookaheadError` ante cualquier consulta hacia adelante. El control no es una
convención sino una barrera.

`evaluation.py` provee el split in-sample / out-of-sample sin solapamiento y el
Sharpe deflactado por cantidad de configuraciones probadas. A igual Sharpe
anualizado de 1.80, el DSR cae de 0.96 con una configuración a 0.007 con veinte.

## Tests

```bash
pytest
```

25 tests, sin red ni credenciales. La estrategia es generar los precios con el
mismo modelo que después detecta: si un detector encuentra algo sobre una cadena
generada por el modelo, el falso positivo es del detector.

`test_sdk_contract.py` construye los objetos que devuelve Alpaca y verifica que
los adaptadores lean los campos correctos. Es la clase de test que detecta un
nombre de campo inventado, cosa que ningún test con datos propios logra.
