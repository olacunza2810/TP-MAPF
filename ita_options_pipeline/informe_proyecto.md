---
title: "Pipeline de detección y backtesting de arbitrajes de opciones"
subtitle: "Proyecto ITA - RTX, BA y LMT"
author: "Informe técnico para la presentación"
date: "14 de septiembre de 2026"
geometry: margin=2.2cm
fontsize: 11pt
---

# Resumen ejecutivo

El proyecto construye un proceso reproducible para estudiar posibles arbitrajes de opciones sobre las acciones RTX, BA y LMT, constituyentes individuales del ETF ITA. El flujo parte de contratos y cotizaciones, limpia y alinea la información con controles contra lookahead bias, calcula volatilidad implícita mediante un árbol binomial americano, filtra contratos que no parecen operables, detecta inconsistencias de precios y finalmente simula la ejecución con costos, latencia y reglas de riesgo.

Las etapas de datos, limpieza, valuación, detección, backtesting y evaluación están implementadas y cuentan con 59 tests. La evidencia empírica proviene de velas diarias reales descargadas de Polygon (Massive) con el plan gratuito: 365 contratos de RTX, BA y LMT, 7.323 velas entre el 2 de marzo y el 11 de septiembre de 2026.

Sobre esos datos se corrió una grilla factorial de 16 configuraciones: fuente de precio, spread supuesto, latencia de ejecución y umbral del kill-switch. La señal se detecta al cierre y la orden se ejecuta a la apertura de la rueda siguiente, sólo si el edge sigue existiendo. En la configuración base (cierre, spread 2%, apertura de t+1, kill-switch de USD 2.000):

- 92 operaciones con P&L simulado de USD 19.055 y hit rate de 97,8%;
- las 81 operaciones que llegaron al vencimiento cerraron con ganancia;
- la paridad put-call explica el 87% del P&L;
- el Sharpe deflactado es 0,75 con N = 16 y 0,71 con N = 35.

La distinción central para la presentación es la siguiente:

- Las 16 configuraciones dan P&L simulado positivo, pero **ninguna alcanza un Sharpe deflactado de 0,95**: el resultado no se distingue estadísticamente de lo que produciría una búsqueda sobre ruido.
- El bid y el ask son sintéticos, los cierres no son sincrónicos y el backtest no cobra el fondeo ni los dividendos de la paridad. El P&L ajustado estimado de la configuración base baja a USD 17.243.
- La revisión de literatura, el paper trading, el monitoreo en vivo y el P&L real acumulado siguen pendientes.

# 1. Objetivo y pregunta financiera

La pregunta del proyecto es si una cadena de opciones contiene precios incompatibles con relaciones de no arbitraje, y si esas diferencias son suficientemente grandes y líquidas como para intentar una estrategia multi-pata.

No alcanza con comparar un precio observado contra un precio teórico. Una oportunidad debe cumplir simultáneamente varias condiciones:

1. La relación financiera debe ser correcta para opciones americanas.
2. Las cotizaciones deben haber estado disponibles en ese instante.
3. Debe poder comprarse al ask y venderse al bid.
4. La señal debe sobrevivir comisiones, spread, borrow y latencia.
5. Todas las patas deben tener liquidez suficiente.
6. El resultado debe mantenerse fuera de muestra.

Por eso el proyecto separa detectores model-free, basados en relaciones de no arbitraje, de un detector model-based, basado en el precio del árbol binomial.

# 2. Arquitectura del proyecto

El flujo completo es:

```text
Alpaca
  -> universo de contratos
  -> snapshots y barras históricas
  -> esquema y limpieza
  -> alineación temporal con el subyacente
  -> DTE, mid, spread e IV
  -> filtros de liquidez
  -> persistencia Parquet
  -> detectores de arbitraje
  -> backtest event-driven
  -> evaluación in-sample / out-of-sample
```

## Módulos principales

| Módulo | Función |
|---|---|
| `config.py` | Define subyacentes, tasas, dividendos, feed, límites y credenciales. |
| `clients.py` | Envuelve el SDK sincrónico de Alpaca con `asyncio`, rate limit, concurrencia y reintentos. |
| `ingest.py` | Descarga contratos, graba snapshots y hace backfill de barras. |
| `schemas.py` | Fija las columnas y tipos del dataset. |
| `enrich.py` | Calcula mid, spread, tiempo a vencimiento, alineación y volatilidad. |
| `volatility.py` | Implementa BSM, árbol CRR, cotas e inversión de IV. |
| `calibration.py` | Define el perfil calibrado de cada activo y simula su precio con difusión con saltos de Merton. |
| `demo.py` | Genera el mercado sintético y corre el pipeline completo offline. |
| `doctor.py` | Diagnostica credenciales, feed, universo, NBBO y viabilidad antes de operar. |
| `filters.py` | Marca contratos no explotables y produce un reporte auditable. |
| `arbitrage.py` | Implementa seis detectores y calcula el edge neto. |
| `storage.py` | Escribe y lee Parquet particionado y manifiestos. |
| `backtest.py` | Simula señales, ejecuciones (al quote o a la apertura), control de edge residual, cierres, costos y kill-switch. |
| `evaluation.py` | Hace split temporal, grid search, métricas, Sharpe deflactado, embudo de señales y desgloses de P&L. |
| `daily.py` | Modo diario: sella velas al cierre, calcula el volumen de la rueda anterior, cura y corre el backtest EOD. |
| `pipeline.py` | Expone la CLI: `demo`, `doctor`, `record`, `backfill`, `enrich`, `detect`, `enrich-daily` y `backtest-daily`. |
| `scripts/download_polygon.py` | Descarga velas diarias de Polygon (Massive) respetando el plan gratuito. |
| `scripts/run_daily_grid.py` | Corre la grilla factorial de 16 configuraciones y calcula el DSR. |

# 3. Revisión de literatura y aplicación

Esta parte todavía debe documentarse formalmente en el proyecto. La revisión puede organizarse en cuatro grupos.

## Modelo binomial

Cox, Ross y Rubinstein introducen el árbol binomial para valorar opciones. El precio evoluciona en pasos discretos hacia arriba o hacia abajo y se descuenta el valor esperado bajo una probabilidad riesgo-neutral.

La aplicación al caso es directa: RTX, BA y LMT tienen opciones americanas, por lo que el árbol permite comparar en cada nodo el valor de continuar con el valor de ejercer inmediatamente.

## Opciones americanas

La literatura sobre opciones americanas muestra que Black-Scholes europeo no captura correctamente el ejercicio anticipado. La aproximación de Barone-Adesi y Whaley y los trabajos de Broadie y Detemple sirven como referencias para justificar el tratamiento específico de opciones americanas.

En el proyecto, Black-Scholes queda como control europeo, mientras que el árbol CRR es el modelo principal para la IV americana.

## Superficie de volatilidad

La volatilidad no es constante entre strikes y vencimientos. En acciones suele observarse skew: las puts OTM pueden tener una IV mayor por la demanda de protección contra caídas.

La demo genera una superficie con skew y smile. El código también advierte que una única sigma global puede producir supuestos arbitrajes en las alas cuando en realidad el problema es la mala especificación del modelo.

## Arbitraje y ejecución

Las relaciones de monotonicidad, convexidad, paridad put-call y calendarios permiten construir tests model-free. La literatura de microestructura y ejecución agrega el punto esencial: una anomalía matemática no es operable si el spread, la latencia o la falta de profundidad consumen el edge.

## Mensaje de la literatura

La aplicación al proyecto es:

> Primero se deben comprobar relaciones model-free con precios ejecutables. Después se puede usar el modelo binomial para detectar dislocaciones adicionales, pero esas señales tienen que interpretarse junto con el smile, la liquidez y la incertidumbre de los parámetros.

# 4. Obtención de datos

## Datos disponibles en Alpaca

El gateway utiliza:

- contratos de opciones y su open interest;
- snapshots con bid, ask, tamaños, último trade, IV y griegas;
- barras históricas de opciones;
- barras del subyacente.

El subyacente se pide con `adjustment="raw"`. Las series ajustadas retroactivamente por dividendos o splits podrían incorporar información futura y además no estar expresadas en la misma convención que los strikes de las opciones.

## Limitación del histórico

Alpaca no ofrece un histórico completo de NBBO de opciones. Por eso hay dos modos:

1. `record`: graba snapshots desde el momento en que se prende el recorder. Es la fuente preferida para tener bid/ask real.
2. `backfill`: recupera barras pasadas y estima el spread. Sirve para extender la historia, pero queda marcado como proxy.

El feed `indicative` es gratuito y demorado. El feed `opra` es más apropiado para tiempo real, pero requiere una suscripción adicional.

## Persistencia

Los datos se guardan en Parquet particionado por subyacente y fecha. Junto al dataset se puede guardar un manifiesto con los supuestos usados: tasas, dividendos, umbrales, feed y parámetros del modelo.

## Modo diario con Polygon (Massive) gratuito

La evidencia empírica del informe proviene de Polygon (rebautizado Massive), porque Alpaca no permite reconstruir cotizaciones pasadas. El plan gratuito impone dos restricciones que definen la arquitectura:

1. no incluye cotizaciones históricas de opciones (`/v3/quotes`) ni snapshots: sólo velas diarias OHLCV con VWAP por contrato;
2. admite 5 requests por minuto.

### Descarga

`scripts/download_polygon.py` descarga, en un solo request por contrato, la serie diaria completa de su ventana. El universo se acota antes de pedir datos:

- **Vencimientos:** los dos vencimientos mensuales más próximos en cada rueda. Cada vencimiento se descarga sólo mientras pertenece a ese par.
- **Strikes:** dentro de ±15% del spot y, de esos, los cuatro más cercanos por encima y por debajo. El spot de referencia es el cierre de la rueda anterior al inicio de la ventana, para no seleccionar contratos con precios futuros.
- **Tasa:** 12 segundos entre requests y backoff exponencial ante HTTP 429. Cada respuesta queda en caché, de modo que la descarga es reanudable.

| Parámetro | Valor |
|---|---|
| Período | 2026-03-01 a 2026-09-12 (ruedas del 2 de marzo al 11 de septiembre) |
| Subyacentes | RTX, BA y LMT |
| Contratos | 365 |
| Velas diarias de opciones | 7.323 (BA 3.121, RTX 2.425, LMT 1.777) |
| Requests totales | 415, en unos 83 minutos |

Además se descargan las velas diarias sin ajustar de las tres acciones, los dividendos y los rendimientos del Tesoro.

### Curación

`enrich-daily` adapta el pipeline a velas diarias sin tocar los detectores:

- **Sellado temporal.** Cada vela se sella a las 16:00 de Nueva York de su rueda, cuando su cierre ya se conoce. El spot del subyacente se sella igual y se aparea con la misma rueda.
- **Bid y ask sintéticos.** No hay cotizaciones, así que se construyen alrededor del precio de referencia (`close` o `vwap`) con un spread relativo supuesto: $h=\max(p\,s/2,\ 0{,}01)$, $bid=p-h$, $ask=p+h$. Con la misma regla sobre el `open` se construyen los lados de la apertura, contra los que se ejecuta. Todas esas filas quedan marcadas con `spread_is_proxy=True`.
- **Liquidez.** Sin open interest histórico ni spread observado, el filtro usa el volumen de la rueda anterior (`volume_prev_day > 0`), el precio mínimo de USD 0,05, el rango de 7 a 180 días al vencimiento y el spot alineado.

## Muestra sintética calibrada

Como no hay NBBO histórico, la demo y la validación de los detectores usan un mercado generado. Para que sea un banco de pruebas válido, la muestra tiene que reproducir la distribución de cada activo real. La metodología completa está en `justificacion_generacion_datos.md`; el resumen es:

- **Elección de activos.** RTX y BA, más LMT como tercer constituyente del ITA. Se descartó GE Aerospace porque su escisión de 2024 hace que la serie histórica no corresponda a la entidad actual. Los tres activos cubren regímenes distintos: BA con vol alta y colas muy gordas, RTX con vol media y distribución más simétrica, LMT con vol de base baja y saltos raros pero severos.
- **Modelo.** Difusión con saltos de Merton bajo la medida riesgo-neutral. La vol total del proceso se fija igual a la IV at-the-money con la que se valúan las opciones, de modo que la vol realizada coincide con la implícita por construcción.
- **Parámetros por activo** (fuentes públicas, septiembre de 2026):

| Parámetro | RTX | BA | LMT |
|---|---:|---:|---:|
| Spot de referencia | 174 | 210 | 524 |
| Dividendo continuo | 1,6% | 0% | 2,6% |
| Vol total anual (= IV atm) | 28% | 36% | 26% |
| Saltos por año | 6 | 8 | 4 |
| Media / desvío del salto | −2,5% / 3,5% | −3,5% / 5,5% | −3,0% / 4,5% |
| Skew / curvatura del smile | 0,35 / 0,90 | 0,55 / 1,30 | 0,35 / 0,85 |

- **Validación.** Sobre 24 réplicas de 2.520 días, la vol simulada iguala al objetivo (28,0%, 36,0% y 26,0%), el skew es negativo en los tres (−0,45, −0,97 y −0,69) y el exceso de curtosis es positivo (+2,2, +6,3 y +4,6). `validate_sample.py` reproduce la tabla y la figura `outputs/validacion_distribucion.png`, y `tests/test_calibration.py` falla si algún cambio rompe ese parecido.

El proceso tiene vol constante por activo: no modela clustering de volatilidad. Es una decisión consciente para mantener la coherencia con el árbol CRR de una sola sigma por contrato.

# 5. Limpieza y controles de calidad

## Mid y spread

Para cada quote se calcula:

$$m = \frac{bid + ask}{2}, \qquad spread = ask - bid, \qquad spread\_rel = \frac{ask-bid}{m}.$$

El mid sirve para calibrar, pero no representa una ejecución garantizada. La ejecución compra al ask y vende al bid.

## Alineación anti-lookahead

Las barras de un minuto se etiquetan por su inicio. El cierre de la barra 14:30 sólo se conoce después de las 14:31. `align_underlying()` desplaza la disponibilidad de la barra y usa un `merge_asof` hacia atrás.

La función nunca toma una barra futura, aplica una tolerancia máxima y registra el retraso de alineación.

## Open interest

El OI publicado para un día puede conocerse recién después del cierre. Por eso se exige:

```text
open_interest_date < trade_date
```

Si el dato no estaba disponible todavía, se anula para ese quote.

## Survivorship bias

Los contratos vencidos desaparecen del universo actual. El maestro append-only diario permite reconstruir qué contratos existían en cada fecha, pero no puede corregir períodos anteriores al comienzo de la grabación.

## Validación técnica

La suite cubre, entre otros casos:

- convergencia del CRR al europeo sin dividendos;
- paridad put-call europea;
- prima no negativa por ejercicio anticipado;
- round-trip de IV;
- alineación contra una barra envenenada para verificar que no se consuma futuro;
- contratos del SDK de Alpaca;
- detectores de arbitraje y backtest.

# 6. Modelo binomial y calibración

El árbol Cox-Ross-Rubinstein usa:

$$u=e^{\sigma\sqrt{\Delta t}},\qquad d=\frac{1}{u},\qquad p=\frac{e^{(r-q)\Delta t}-d}{u-d}.$$

En cada nodo se calcula el máximo entre ejercer y continuar:

$$V_i^n = \max\left(\text{payoff}, e^{-r\Delta t}\left[pV_{i+1}^{n+1}+(1-p)V_i^{n+1}\right]\right).$$

La IV se obtiene resolviendo:

$$V_{CRR}(\sigma)-V_{mercado}=0.$$

Se usa Brent en un intervalo acotado. Antes se controlan las cotas de no arbitraje. Un precio fuera de esas cotas se marca como dato inválido o posible arbitraje estático, pero no se fuerza una IV.

## Resultados de la demo offline

La corrida realizada produjo:

- 1.872 cotizaciones;
- 12 cortes temporales;
- 296 contratos;
- 3 subyacentes (RTX, BA y LMT);
- 100% de las cotizaciones con spot alineado;
- 1.872 de 1.872 IV americanas invertidas;
- IV mediana cercana a 27,8%;
- rango aproximado entre 24,7% y 73,9%;
- cero filas fuera de las cotas de no arbitraje.

Estos números demuestran que la implementación funciona sobre el mercado sintético generado por el proyecto. No son resultados de mercado real.

## Qué falta para afirmar ajuste a la superficie observada

Hay que ejecutar el proceso sobre snapshots o datos históricos reales y presentar:

- gráfico de IV contra moneyness;
- gráfico de IV contra vencimiento;
- precio observado contra precio CRR;
- RMSE o MAE ponderado por vega;
- comparación entre IV americana, IV europea e IV del proveedor;
- análisis separado por RTX, BA y LMT.

El código ya contiene `calibrate_chain_sigma()` para una sigma global, pero todavía no hay en el repositorio una tabla real con esos resultados.

Sobre las velas diarias reales de Polygon, la IV americana se invirtió en 6.785 de las 7.323 filas con precio de cierre (mediana 33,4%) y en 6.697 con VWAP (mediana 33,5%). Esas IV se calculan sobre precios de último trade con tasa plana y dividendos continuos, así que sirven como control de consistencia de los datos y no como calibración de la superficie.

# 7. Detección de arbitrajes

## Detectores model-free

`arbitrage.py` implementa:

1. Monotonicidad vertical: el call debe bajar al subir el strike y la put debe subir.
2. Cota del spread vertical: el valor del spread no puede superar su ancho.
3. Convexidad o butterfly: la curva de precios debe ser convexa.
4. Paridad put-call americana: se usan desigualdades, no la igualdad europea.
5. Calendarios: una opción con mayor vencimiento no debería valer menos.

Todos compran al ask y venden al bid. Se descartan quotes con bid o ask cero y se exige `is_tradable`.

## Detector model-based

El sexto detector compara el precio CRR contra bid y ask. Es una señal más débil porque depende de la sigma y de los supuestos de tasa, dividendos y ejercicio.

## Correcciones relevantes

El proyecto corrige falsos positivos del notebook original:

- una butterfly con el signo invertido puede dar crédito hoy a cambio de payoff negativo;
- un calendario con las patas invertidas detecta el spread normal del mercado;
- un ask o bid cero no es una opción gratis, sino una cotización ausente;
- usar paridad europea sobre opciones americanas crea falsos positivos en puts ITM.

# 8. Filtros de liquidez

Los umbrales por defecto son:

| Criterio | Umbral |
|---|---:|
| Volumen | mayor que 0 |
| Open interest | mayor que 50 |
| Bid | mayor que 0 |
| Spread relativo | menor que 15% |
| Mid | al menos USD 0,05 |
| DTE mínimo | 7 días |
| DTE máximo | 180 días |

En la demo se conservaron 1.855 de 1.872 filas, un 99,1%. Dieciséis fueron excluidas por open interest insuficiente y una por volumen insuficiente. Nuevamente, es un resultado sintético.

Con las velas diarias reales se usan los umbrales del modo diario (sección 4). Se conservaron 5.602 de 7.323 filas, un 76,5%. Las causas de exclusión, no excluyentes entre sí, fueron:

| Criterio | Filas excluidas |
|---|---:|
| Días al vencimiento fuera de 7 a 180 | 1.008 |
| Sin volumen en la rueda anterior | 788 |
| Precio menor a USD 0,05 | 230 con cierre, 182 con VWAP |
| Volumen anterior desconocido | 0 |
| Spot no alineado | 0 |

La oportunidad se reporta con la liquidez de la peor pata: volumen mínimo, OI mínimo y máximo spread relativo. Esto evita mostrar un arbitraje de tres patas como líquido sólo porque dos patas lo son.

# 9. Backtest y costos

El motor es event-driven. La señal ocurre en un timestamp y la ejecución ocurre después de una latencia configurable. En la ejecución se vuelven a cotizar todas las patas; si una dejó de cotizar, se rechaza la operación completa.

Los supuestos principales son:

- comisión de USD 0,65 por contrato;
- multiplicador de 100 acciones;
- edge mínimo de USD 5;
- borrow anual de 0,50% para short del subyacente;
- spread aproximado de USD 0,01 en la acción;
- cierre al bid o ask según el sentido de la pata;
- límite de posiciones simultáneas;
- tamaño máximo por señal;
- kill-switch por posición;
- control de edge residual: al ejecutar se recalcula el edge con los precios de ese momento, con la misma fórmula y comisión del detector, y la operación se descarta si quedó bajo el mínimo.

La barrera `PointInTimeView` no permite pedir datos posteriores al reloj. Esto controla lookahead por construcción, no sólo por convención.

## Resultados con datos reales: modo diario

### Diseño de la prueba

- **Señal.** Al cierre de la rueda t (16:00 ET), sobre el bid y el ask sintéticos construidos alrededor del precio de referencia.
- **Ejecución.** A la apertura (09:30 ET) de t+1 o t+2, contra el bid y el ask sintéticos del `open`. Sólo se ejecuta si el contrato operó esa rueda y el edge recalculado con esos precios sigue siendo al menos USD 5. La orden se procesa antes que los cierres de la rueda, así que la capacidad disponible es la que había a las 09:30.
- **Costos y límites.** Comisión de USD 0,65 por contrato, edge mínimo de USD 5, hasta 20 posiciones simultáneas y hasta 3 señales por rueda.
- **Salidas.** Liquidación al vencimiento por valor intrínseco, kill-switch por pérdida no realizada, o cierre al final de los datos marcando al bid o al ask.
- **Grilla.** Factorial de 16 configuraciones: fuente de precio (`close` o `vwap`) × spread supuesto (2% o 5%) × ejecución (apertura de t+1 o de t+2) × kill-switch (USD 1.500 o 2.000).
- **Configuración base.** Cierre, 2%, t+1 y USD 2.000, fijada antes de correr la grilla.

`scripts/run_daily_grid.py` reproduce todos los números de esta sección.

### Resultados de las 16 configuraciones

| Configuración | Ops | P&L | Hit rate | Edge capt. | Sharpe | Máx. drawdown | DSR N=16 | DSR N=35 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| close · 2% · t+1 · 1.500 | 92 | USD 16.311 | 96,7% | 0,96 | 1,56 | USD 7.723 | 0,71 | 0,67 |
| **close · 2% · t+1 · 2.000 (base)** | 92 | USD 19.055 | 97,8% | 1,10 | 1,69 | USD 7.721 | 0,75 | 0,71 |
| close · 2% · t+2 · 1.500 | 88 | USD 20.030 | 97,7% | 1,22 | 1,79 | USD 5.741 | 0,77 | 0,74 |
| close · 2% · t+2 · 2.000 | 88 | USD 20.030 | 97,7% | 1,22 | 1,79 | USD 5.741 | 0,77 | 0,74 |
| close · 5% · t+1 · 1.500 | 76 | USD 15.342 | 97,4% | 1,10 | 1,28 | USD 8.834 | 0,64 | 0,60 |
| close · 5% · t+1 · 2.000 | 76 | USD 19.005 | 98,7% | 1,37 | 1,50 | USD 8.832 | 0,70 | 0,66 |
| close · 5% · t+2 · 1.500 | 52 | USD 13.095 | 98,1% | 1,62 | 1,48 | USD 5.213 | 0,70 | 0,66 |
| close · 5% · t+2 · 2.000 | 52 | USD 13.095 | 98,1% | 1,62 | 1,48 | USD 5.213 | 0,70 | 0,66 |
| vwap · 2% · t+1 · 1.500 | 102 | USD 11.667 | 93,1% | 0,46 | 0,68 | USD 14.699 | 0,47 | 0,43 |
| vwap · 2% · t+1 · 2.000 | 103 | USD 18.186 | 96,1% | 0,68 | 0,96 | USD 14.619 | 0,55 | 0,51 |
| vwap · 2% · t+2 · 1.500 | 89 | USD 16.048 | 98,9% | 0,75 | 1,21 | USD 7.183 | 0,62 | 0,58 |
| vwap · 2% · t+2 · 2.000 | 89 | USD 16.048 | 98,9% | 0,75 | 1,21 | USD 7.183 | 0,62 | 0,58 |
| vwap · 5% · t+1 · 1.500 | 84 | USD 5.450 | 92,9% | 0,26 | 0,37 | USD 15.525 | 0,39 | 0,35 |
| vwap · 5% · t+1 · 2.000 | 84 | USD 10.522 | 94,0% | 0,50 | 0,69 | USD 15.095 | 0,47 | 0,43 |
| vwap · 5% · t+2 · 1.500 | 82 | USD 10.745 | 96,3% | 0,69 | 0,89 | USD 7.967 | 0,53 | 0,49 |
| vwap · 5% · t+2 · 2.000 | 82 | USD 10.745 | 96,3% | 0,69 | 0,89 | USD 7.967 | 0,53 | 0,49 |

El Sharpe está anualizado sobre 134 ruedas. El DSR se explica en la sección 10.

### Embudo de señales y tipo de salida

| Configuración | Detectadas | Encoladas | Ejecutadas | Edge desaparecido | Sin precio | Sin capacidad | Vencimiento | Kill-switch | Fin de datos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| close · 2% · t+1 · 1.500 | 2.537 | 397 | 92 | 179 | 82 | 41 | 80 | 2 | 10 |
| **close · 2% · t+1 · 2.000 (base)** | 2.537 | 397 | 92 | 179 | 82 | 41 | 81 | 1 | 10 |
| close · 2% · t+2 · 1.500 | 2.537 | 397 | 88 | 192 | 106 | 5 | 77 | 0 | 11 |
| close · 2% · t+2 · 2.000 | 2.537 | 397 | 88 | 192 | 106 | 5 | 77 | 0 | 11 |
| close · 5% · t+1 · 1.500 | 1.512 | 348 | 76 | 183 | 82 | 4 | 72 | 2 | 2 |
| close · 5% · t+1 · 2.000 | 1.512 | 348 | 76 | 183 | 82 | 4 | 73 | 1 | 2 |
| close · 5% · t+2 · 1.500 | 1.512 | 348 | 52 | 191 | 99 | 0 | 46 | 0 | 6 |
| close · 5% · t+2 · 2.000 | 1.512 | 348 | 52 | 191 | 99 | 0 | 46 | 0 | 6 |
| vwap · 2% · t+1 · 1.500 | 3.177 | 403 | 102 | 162 | 69 | 67 | 85 | 5 | 12 |
| vwap · 2% · t+1 · 2.000 | 3.177 | 403 | 103 | 166 | 70 | 61 | 88 | 2 | 13 |
| vwap · 2% · t+2 · 1.500 | 3.177 | 403 | 89 | 214 | 82 | 12 | 80 | 1 | 8 |
| vwap · 2% · t+2 · 2.000 | 3.177 | 403 | 89 | 214 | 82 | 12 | 80 | 1 | 8 |
| vwap · 5% · t+1 · 1.500 | 2.116 | 387 | 84 | 215 | 63 | 22 | 72 | 5 | 7 |
| vwap · 5% · t+1 · 2.000 | 2.116 | 387 | 84 | 215 | 63 | 22 | 73 | 3 | 8 |
| vwap · 5% · t+2 · 1.500 | 2.116 | 387 | 82 | 223 | 71 | 5 | 75 | 2 | 5 |
| vwap · 5% · t+2 · 2.000 | 2.116 | 387 | 82 | 223 | 71 | 5 | 75 | 2 | 5 |

- **Detectadas** son todas las violaciones por encima del umbral. **Encoladas** son las que entran por el tope de tres por rueda.
- Cada encolada termina en un solo destino: ejecutada, edge desaparecido, sin precio (el contrato no operó en la rueda de ejecución), sin capacidad o pendiente al final de los datos.
- Entre 162 y 223 señales por configuración (40% a 58% de las encoladas) se descartaron porque la violación ya no existía en la apertura.

### Desglose de la configuración base

Embudo por detector:

| Detector | Detectadas | Encoladas | Ejecutadas | Edge desaparecido | Sin precio | Sin capacidad | Pendientes |
|---|---:|---:|---:|---:|---:|---:|---:|
| `butterfly` | 1.235 | 112 | 20 | 58 | 25 | 8 | 1 |
| `calendar` | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| `monotonicity` | 103 | 11 | 1 | 7 | 2 | 1 | 0 |
| `put_call_parity` | 981 | 245 | 67 | 102 | 44 | 30 | 2 |
| `vertical_bound` | 210 | 29 | 4 | 12 | 11 | 2 | 0 |
| **TOTAL** | 2.537 | 397 | 92 | 179 | 82 | 41 | 3 |

P&L por detector:

| Detector | Ops | P&L total | P&L medio | Hit rate | Edge anunciado | Edge al ejecutar | Edge capt. |
|---|---:|---:|---:|---:|---:|---:|---:|
| `put_call_parity` | 67 | USD 16.534 | USD 247 | 99% | USD 13.739 | USD 15.247 | 1,20 |
| `butterfly` | 20 | USD 1.740 | USD 87 | 95% | USD 3.021 | USD 1.080 | 0,58 |
| `vertical_bound` | 4 | USD 749 | USD 187 | 100% | USD 314 | USD 477 | 2,39 |
| `monotonicity` | 1 | USD 31 | USD 31 | 100% | USD 206 | USD 31 | 0,15 |

P&L por tipo de salida:

| Salida | Ops | P&L total | P&L medio | Hit rate | Edge anunciado | Edge al ejecutar | Edge capt. |
|---|---:|---:|---:|---:|---:|---:|---:|
| `expiry` | 81 | USD 20.779 | USD 257 | 100% | USD 15.749 | USD 16.157 | 1,32 |
| `end_of_data` | 10 | USD 851 | USD 85 | 90% | USD 785 | USD 382 | 1,09 |
| `stop_loss` | 1 | -USD 2.575 | -USD 2.575 | 0% | USD 746 | USD 296 | -3,45 |

### Kill-switch frente a salidas por vencimiento

Las dos salidas cumplen roles opuestos:

- **Vencimiento.** Es la salida natural de una estructura de no arbitraje: se liquida por valor intrínseco y el payoff queda acotado por construcción. En las 16 configuraciones, **el 100% de las operaciones liquidadas al vencimiento cerró con ganancia**. Esa es la firma empírica de que las posiciones ejecutadas con edge residual eran estructuras de arbitraje bien armadas.
- **Kill-switch.** Es una red de contención extrema y no una regla de toma de ganancias. Se activó entre 0 y 5 veces por configuración, siempre con pérdida. En la base se activó una sola vez, en una paridad sobre LMT abierta el 1 de mayo y cerrada el 11 de junio (-USD 2.575). Esa tenencia atravesó la fecha ex-dividendo de LMT del 1 de junio (USD 3,45 por acción), justamente el caso donde el supuesto de dividendo continuo desplaza la cota de la paridad.
- **1.500 frente a 2.000.** Con ejecución en t+2 los dos umbrales dan resultados idénticos. Con t+1, el umbral de USD 1.500 cierra entre una y tres posiciones más y reduce el P&L entre USD 2.700 y 6.500. El backtest no permite saber si esas posiciones habrían convergido al vencimiento, así que la comparación no justifica elegir un umbral por su P&L.

### Fondeo y dividendos no contabilizados

El P&L simulado de la paridad supera el edge al ejecutar (captura de 1,20 en la base). Para una conversión, cuyo payoff al vencimiento está fijado, eso delata costos que el backtest no cobra: los intereses sobre el nocional de la acción y los dividendos durante la tenencia. La tabla estima ese ajuste con tasa plana de 4,25% y los dividendos descargados de Polygon.

| Configuración | P&L simulado | Fondeo paridad | Dividendos paridad | P&L ajustado estimado |
|---|---:|---:|---:|---:|
| close · 2% · t+1 · 1.500 | USD 16.311 | -USD 1.137 | -USD 544 | USD 14.630 |
| **close · 2% · t+1 · 2.000 (base)** | USD 19.055 | -USD 922 | -USD 889 | USD 17.243 |
| close · 2% · t+2 · 1.500 | USD 20.030 | -USD 1.780 | -USD 345 | USD 17.905 |
| close · 2% · t+2 · 2.000 | USD 20.030 | -USD 1.780 | -USD 345 | USD 17.905 |
| close · 5% · t+1 · 1.500 | USD 15.342 | -USD 450 | -USD 418 | USD 14.474 |
| close · 5% · t+1 · 2.000 | USD 19.005 | -USD 456 | -USD 418 | USD 18.131 |
| close · 5% · t+2 · 1.500 | USD 13.095 | -USD 1.362 | USD 0 | USD 11.732 |
| close · 5% · t+2 · 2.000 | USD 13.095 | -USD 1.362 | USD 0 | USD 11.732 |
| vwap · 2% · t+1 · 1.500 | USD 11.667 | -USD 2.168 | -USD 418 | USD 9.081 |
| vwap · 2% · t+1 · 2.000 | USD 18.186 | -USD 2.024 | -USD 763 | USD 15.399 |
| vwap · 2% · t+2 · 1.500 | USD 16.048 | -USD 2.195 | -USD 564 | USD 13.290 |
| vwap · 2% · t+2 · 2.000 | USD 16.048 | -USD 2.195 | -USD 564 | USD 13.290 |
| vwap · 5% · t+1 · 1.500 | USD 5.450 | -USD 687 | -USD 418 | USD 4.345 |
| vwap · 5% · t+1 · 2.000 | USD 10.522 | -USD 693 | -USD 418 | USD 9.412 |
| vwap · 5% · t+2 · 1.500 | USD 10.745 | -USD 2.128 | -USD 345 | USD 8.272 |
| vwap · 5% · t+2 · 2.000 | USD 10.745 | -USD 2.128 | -USD 345 | USD 8.272 |

La estimación infiere la dirección de cada paridad por el signo de la caja de entrada y no incluye el costo de tomar prestada la acción. En butterflies y spreads verticales una captura mayor a 1 sí es legítima: el edge mide el crédito en el peor escenario, y el payoff al vencimiento puede sumar valor intrínseco a favor.

### Lectura

1. **Signo positivo en todas las configuraciones.** El P&L simulado va de USD 5.450 a 20.030 y el ajustado estimado de USD 4.345 a 18.131.
2. **La paridad concentra el resultado.** Es la estrategia con más supuestos no contabilizados (fondeo, dividendos discretos, préstamo de la acción), de modo que es también la parte más frágil del P&L.
3. **VWAP no mejora.** Con VWAP el Sharpe cae (0,37 a 1,21 contra 1,28 a 1,79 con cierre), el drawdown máximo casi se duplica (hasta USD 15.525, contra USD 8.834 con cierre), el kill-switch se activa más y se detectan más señales (3.177 contra 2.537 con spread 2%). Es coherente con la incoherencia de comparar el VWAP de la opción con el cierre de la acción, y no debe presentarse como mejora.
4. **Más spread supuesto, menos oportunidades.** Pasar del 2% al 5% reduce las detecciones (de 2.537 a 1.512 con cierre) y las operaciones.
5. **La latencia no tiene un efecto monótono.** Con cierre y 2%, t+2 mejora levemente a t+1. Con 5%, t+2 opera un 32% menos. La composición cambia también por el tope de posiciones (41 rechazos por capacidad en t+1 contra 5 en t+2).
6. **Ninguna configuración supera un DSR de 0,95** (sección 10).

## Resultado de la demo

La demo sintética cubre una hora y tiene vencimientos varios meses después. Por eso las posiciones se cierran por fin de datos y no por vencimiento:

| Latencia | Operaciones | P&L aproximado |
|---:|---:|---:|
| 1 minuto | 3 | -USD 481 |
| 5 minutos | 3 | -USD 481 |
| 15 minutos | 3 | -USD 674 |

Con el control de edge residual activo, sólo se ejecutan las tres señales cuyo edge sobrevive al re-cotizar. Las tres posiciones cierran por fin de datos. El resultado negativo es coherente con pagar el spread de entrada y salida sin llegar a la liquidación final, y empeora con más latencia porque la ejecución se hace contra quotes más alejados de la señal. No es una medida de rentabilidad de la estrategia.

# 10. In-sample y out-of-sample

`evaluation.py` divide cronológicamente el dataset. Por defecto se puede usar 60% in-sample y 40% out-of-sample.

La búsqueda de hiperparámetros puede variar:

- edge mínimo;
- spread máximo;
- open interest y volumen mínimos;
- DTE máximo;
- cantidad de posiciones;
- detectores activos;
- latencia;
- stop-loss;
- tamaño de la posición.

La grilla registra todas las configuraciones probadas y no sólo la ganadora. Luego se calcula el Sharpe deflactado por la cantidad de pruebas, para reducir el riesgo de presentar como descubrimiento un máximo producido por data snooping.

## Sharpe deflactado sobre la grilla diaria

### Espacio de búsqueda

La grilla de la sección 9 es un diseño factorial de cuatro factores con dos niveles cada uno: fuente de precio (cierre o VWAP), spread supuesto (2% o 5%), latencia (apertura de t+1 o de t+2) y tolerancia del kill-switch (USD 1.500 o 2.000). Son N = 16 configuraciones. Se evaluaron y reportaron todas, no sólo la mejor, y la configuración base se fijó antes de correrla.

El Sharpe deflactado (Bailey y López de Prado, 2014) compara el Sharpe observado contra el máximo que se esperaría obtener por azar al probar N configuraciones sin poder predictivo:

$$\mathbb{E}[\max \widehat{SR}] \approx \sqrt{V}\left[(1-\gamma)\,\Phi^{-1}\!\left(1-\tfrac{1}{N}\right)+\gamma\,\Phi^{-1}\!\left(1-\tfrac{1}{Ne}\right)\right],$$

donde $V$ es la varianza de los Sharpe entre configuraciones (0,00074 en Sharpe por rueda) y $\gamma$ la constante de Euler-Mascheroni. Luego se calcula la probabilidad de que el Sharpe verdadero supere ese umbral, corrigiendo por asimetría y curtosis de los retornos diarios. Un DSR menor a 0,95 indica que el resultado no se distingue de lo que produciría la búsqueda sobre ruido.

### Declaración de la búsqueda previa

El espacio factorial no fue el único explorado. Antes de la grilla definitiva se evaluaron:

- tres corridas sin control de edge residual;
- una grilla de 16 configuraciones con ejecución al cierre y stop-loss de USD 500 o desactivado.

La ejecución a la apertura, los niveles del kill-switch y la decisión de ampliar el stop se tomaron después de observar esos resultados. Por eso el DSR se reporta con dos valores de N:

- **N = 16:** la grilla factorial definitiva.
- **N = 35:** todas las configuraciones evaluadas en el proyecto (3 + 16 + 16). Es el valor más honesto para juzgar el resultado.

### Resultados

| | DSR N=16 | DSR N=35 |
|---|---:|---:|
| Configuración base (cierre · 2% · t+1 · USD 2.000) | 0,75 | 0,71 |
| Mejor configuración (cierre · 2% · t+2) | 0,77 | 0,74 |
| Peor configuración (VWAP · 5% · t+1 · USD 1.500) | 0,39 | 0,35 |

Ninguna configuración alcanza 0,95, ni siquiera con N = 16. Con 134 ruedas, un Sharpe anualizado de 1,69 no alcanza para distinguir la estrategia de un máximo producido por la búsqueda. El P&L positivo y el 100% de acierto al vencimiento son consistentes con arbitrajes reales, pero la evidencia estadística es insuficiente para afirmarlo.

### Pendiente: split temporal

Todavía falta el split in-sample / out-of-sample con datos reales. `split_in_sample()` y `grid_search()` ya lo soportan, pero con seis meses y vencimientos mensuales, un corte 60/40 deja muy pocos vencimientos en cada tramo. Conviene hacerlo cuando haya al menos un año de historia descargada.

# 11. Paper trading

Esta parte no está implementada. El proyecto puede consultar contratos mediante `TradingClient`, pero no envía órdenes.

Por ese motivo todavía no existen:

- capturas de órdenes;
- IDs de órdenes;
- fills parciales o completos;
- posiciones reales en paper;
- reconciliación de cuenta;
- P&L acumulado real.

Para cumplir esta parte hay que implementar una capa que reciba `leg_spec`, cree órdenes para todas las patas, espere fills, cancele órdenes incompletas y cierre la estrategia si falla una pata.

La operación debe realizarse únicamente con `ALPACA_PAPER=true` y con tamaños mínimos durante la etapa de validación.

# 12. Monitoreo y stop-loss

El stop-loss actual sólo existe dentro del backtest, donde funciona como **kill-switch de cola**: cierra una posición cuando su pérdida no realizada supera un umbral. No está conectado a una cuenta ni a un proceso en vivo.

La calibración surge del propio backtest. Un umbral estrecho (USD 500) resultó incompatible con estrategias de convergencia al vencimiento marcadas con cierres diarios asincrónicos: el ruido del marcado disparaba el stop y cristalizaba pérdidas en posiciones que al vencimiento habrían convergido. Por eso el kill-switch se fija como red de contención extrema, en USD 1.500 y 2.000 por posición, aproximadamente el 4% y el 5% del nocional medio de una conversión. El nivel se deriva del tamaño de la exposición y no de maximizar el P&L, pero la decisión de ampliarlo se tomó después de observar el resultado con USD 500, y así se declara en la sección 10.

El sistema de producción debería tener estos componentes:

```text
Snapshot de mercado
  -> detector
  -> paper trader
  -> registro de órdenes y fills
  -> monitor de posiciones
  -> risk manager
  -> cierre automático
```

El monitor debe observar:

- P&L realizado y no realizado;
- P&L acumulado diario;
- drawdown;
- posiciones abiertas;
- exposición por subyacente y vencimiento;
- edge anunciado y edge capturado;
- latencia de ejecución;
- quotes stale;
- fills parciales;
- errores de API;
- pérdida por posición y pérdida diaria.

El risk manager debería tener al menos:

- stop-loss por posición;
- stop-loss diario;
- límite de posiciones simultáneas;
- límite de exposición;
- cierre si falta una pata;
- cierre si el quote se vuelve stale;
- cierre si se pierde conexión;
- botón de emergencia.

# 13. Qué está terminado y qué falta

| Etapa de la consigna | Estado actual |
|---|---|
| Literatura | Falta documentar la revisión y sus referencias. |
| Obtención de datos | Código para Alpaca sin probar contra la API real. Descarga diaria de Polygon ejecutada con datos reales (365 contratos, 7.323 velas). |
| Limpieza y calidad | Implementada y testeada; modo diario aplicado sobre datos reales. |
| Calibración binomial | Implementada y testeada; IV americana invertida sobre velas reales. |
| Detección de arbitrajes | Implementada y testeada; aplicada sobre datos reales con bid/ask sintético. |
| Backtest realista | Implementado con costos, ejecución a la apertura, control de edge residual, kill-switch y barrera anti-lookahead. Grilla de 16 configuraciones ejecutada sobre datos reales. |
| In-sample / out-of-sample | Sharpe deflactado calculado sobre la grilla; falta el split temporal con datos reales. |
| Paper trading | Falta implementar envío y seguimiento de órdenes. |
| Monitoreo productivo | Falta implementar. |
| Stop-loss productivo | Kill-switch calibrado sólo dentro del backtest. |
| P&L acumulado real | No disponible: el P&L del backtest es simulado. |

# 14. Limitaciones técnicas

Las limitaciones deben declararse en la presentación.

## Limitaciones del backtest diario

- **Costo de fondeo en conversiones.** El backtest no cobra intereses sobre el nocional de la acción comprada en una conversión ni acredita intereses sobre la venta en corto de una conversión reversa. Tampoco acredita o debita los dividendos con fecha ex durante la tenencia, ni el costo de tomar prestada la acción. El detector sí descuenta el strike ($Ke^{-r\tau}$) al evaluar la oportunidad, así que el P&L simulado de la paridad supera al edge que la originó. La sección 9 cuantifica el ajuste estimado.
- **Dividendos continuos.** La paridad usa un yield continuo. RTX y LMT pagan dividendos en fechas discretas (RTX: 22 de mayo y 14 de agosto; LMT: 2 de marzo, 1 de junio y 1 de septiembre de 2026), lo que desplaza la cota inferior cerca de esas fechas y puede generar conversiones reversas espurias.
- **Asincronía de cierres.** El cierre y la apertura de cada contrato son el precio de su primer o último trade, que ocurre a horas distintas en contratos distintos y en la acción. Comparar esos precios genera violaciones aparentes de no arbitraje. El control de edge residual exige que la violación persista en la apertura siguiente, lo que mitiga pero no elimina el sesgo.
- **VWAP no coherente con el spot.** Con `vwap`, la opción usa el precio medio de la rueda, pero el spot sigue siendo el cierre de la acción. En la paridad eso mezcla precios de momentos distintos.
- **Spreads sintéticos.** El bid y el ask no se observan: se suponen al 2% o al 5% del precio. El edge y el P&L dependen de ese supuesto, y por eso la grilla lo varía.
- **Sin profundidad ni open interest.** No se limita el tamaño por la profundidad del libro y Polygon no publica open interest histórico.
- **Tope de posiciones y señales repetidas.** El límite de 20 posiciones simultáneas rechaza señales y condiciona el P&L; la misma estructura puede reabrirse en ruedas sucesivas.
- **Asignación anticipada.** No se modela el ejercicio anticipado de las patas vendidas, que en opciones americanas con dividendos es el riesgo central de estas estructuras.
- **Selección del universo.** Los strikes listados después del inicio de cada ventana no se descargan.

## Limitaciones generales

- no hay bid/ask histórico completo de opciones en Alpaca;
- el feed gratuito está demorado;
- las barras históricas usan un spread proxy;
- no se limita aún el tamaño por profundidad real del libro;
- no se modela asignación anticipada;
- se usan dividendos continuos en lugar de dividendos discretos;
- se usa una tasa plana por defecto;
- una sigma global no representa todo el smile;
- una misma dislocación puede producir muchas oportunidades duplicadas;
- la curva de tasas por plazo tiene un campo de configuración, pero todavía no se usa;
- el recorder arma el universo una sola vez al arrancar y no refresca el maestro diario si queda corriendo varios días;
- `enrich` toma el open interest del último maestro disponible, lo que anula el OI al procesar fechas pasadas;
- los subcomandos de red de la CLI usan por defecto sólo RTX y BA; LMT hay que pasarlo con `--tickers`;
- la historia previa al recorder conserva survivorship bias;
- el demo no es evidencia de rentabilidad;
- no existe todavía ejecución paper ni monitoreo en vivo.

# 15. Conclusiones

El proyecto construye una base rigurosa para investigar arbitraje de opciones. Su aporte principal no es afirmar que ya encontró una estrategia rentable, sino mostrar cómo debe validarse una señal antes de considerarla operable.

Las conclusiones técnicas son:

1. El modelo binomial americano es más apropiado que Black-Scholes europeo para las opciones analizadas.
2. Las relaciones model-free son la evidencia más fuerte porque no dependen de una sigma correcta.
3. El bid-ask, la latencia y la liquidez pueden eliminar completamente el edge teórico. Con datos reales, entre el 40% y el 58% de las señales encoladas ya no existían en la apertura siguiente.
4. El control temporal de los datos es tan importante como la fórmula de valuación: ejecutar sin verificar el edge residual convertía el backtest en una medición de señales vencidas.
5. La ausencia de NBBO histórico limita fuertemente cualquier backtest retrospectivo: con el plan gratuito de Polygon el bid y el ask deben suponerse.
6. Con velas diarias reales y edge residual verificado, las 16 configuraciones dan P&L simulado positivo y el 100% de las posiciones liquidadas al vencimiento gana. Es la firma de estructuras de no arbitraje bien ejecutadas.
7. Un stop-loss estrecho es incompatible con estrategias de convergencia marcadas con precios asincrónicos. El kill-switch debe calibrarse como red de contención de cola, y aun así se activa en las posiciones más expuestas a los supuestos del modelo, como dividendos discretos.
8. La paridad put-call aporta cerca del 87% del P&L y es también la estrategia más afectada por costos no contabilizados. El ajuste estimado por fondeo y dividendos reduce el P&L de la base de USD 19.055 a 17.243.
9. Ninguna configuración alcanza un Sharpe deflactado de 0,95 (0,75 en la base con N = 16, 0,71 con N = 35): la evidencia no alcanza para afirmar que la estrategia es rentable fuera de muestra.
10. La demo verifica la cañería, pero no prueba rentabilidad.
11. Antes de presentar P&L real hay que contabilizar fondeo y dividendos, grabar cotizaciones propias, ejecutar paper trading y monitorear fills.

La conclusión general para la exposición puede formularse así:

> Una inconsistencia matemática no es automáticamente una oportunidad de trading. Para que sea explotable debe existir una cotización ejecutable, liquidez en todas las patas, una relación financiera correcta para opciones americanas, costos y latencia compatibles, y un resultado que sobreviva fuera de muestra.

# 16. Guion sugerido para la presentación

1. Presentar el objetivo y el flujo general del pipeline.
2. Explicar la limitación de datos: Alpaca no tiene NBBO histórico y el plan gratuito de Polygon sólo da velas diarias a 5 requests por minuto.
3. Mostrar la arquitectura del modo diario: descarga acotada, sellado al cierre, bid/ask sintético y filtro por volumen de la rueda anterior (365 contratos, 7.323 velas, 76,5% operables).
4. Mostrar el control anti-lookahead y el esquema de calidad.
5. Introducir el árbol CRR y comparar IV americana contra europea.
6. Explicar los cinco detectores model-free, el detector contra modelo y las correcciones al notebook de cátedra.
7. Explicar el backtest: señal al cierre, ejecución a la apertura, control de edge residual y barrera `PointInTimeView`.
8. Mostrar la tabla de las 16 configuraciones y el embudo de señales: cuántas violaciones desaparecen antes de poder ejecutarse.
9. Mostrar el desglose por tipo de salida: 100% de acierto al vencimiento frente al rol del kill-switch.
10. Mostrar el desglose por detector y el ajuste por fondeo y dividendos de la paridad.
11. Presentar el Sharpe deflactado con N = 16 y N = 35, declarando la búsqueda previa.
12. Cerrar con limitaciones (spreads sintéticos, asincronía, fondeo, dividendos discretos) y próximos pasos: contabilidad del fondeo, cotizaciones propias, paper trading y monitoreo.

## Estado reproducible al preparar este informe

La suite ejecutada con Python 3.14.3 el 14 de septiembre de 2026 produjo 59 tests exitosos. El demo offline recorrió generación sintética calibrada, alineación, IV, filtros, Parquet, detección y backtest. La capa de Alpaca no se ejecutó porque no estaban configuradas sus credenciales.

Los resultados empíricos de las secciones 9 y 10 se reproducen con:

```text
python scripts/download_polygon.py --start 2026-03-01 --end 2026-09-12 --max-strikes-per-side 4
python scripts/run_daily_grid.py --out reportes_final
```

`reportes_final/` contiene `summary.csv` con las 16 configuraciones y el DSR, `data_summary.json` con la descripción del dataset, y por cada configuración las operaciones, la curva de equity, el embudo de señales y los desgloses por detector y por motivo de salida.
