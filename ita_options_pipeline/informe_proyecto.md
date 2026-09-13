---
title: "Pipeline de detección y backtesting de arbitrajes de opciones"
subtitle: "Proyecto ITA - RTX y BA"
author: "Informe técnico para la presentación"
date: "5 de septiembre de 2026"
geometry: margin=2.2cm
fontsize: 11pt
---

# Resumen ejecutivo

El proyecto construye un proceso reproducible para estudiar posibles arbitrajes de opciones sobre las acciones RTX y BA. El flujo parte de contratos y cotizaciones, limpia y alinea la información con controles contra lookahead bias, calcula volatilidad implícita mediante un árbol binomial americano, filtra contratos que no parecen operables, detecta inconsistencias de precios y finalmente simula la ejecución con costos, latencia y reglas de riesgo.

La implementación técnica de las etapas de datos, limpieza, valuación, detección, backtesting y evaluación está avanzada y cuenta con tests. Sin embargo, el proyecto todavía no es un sistema completo en producción: la revisión de literatura no está documentada en el repositorio, no existe envío de órdenes de paper trading, no hay monitoreo conectado a una cuenta y no existe un P&L real acumulado.

La distinción central para la presentación es la siguiente:

- La demo offline demuestra que la cañería funciona.
- Los datos de la demo son sintéticos y no son evidencia de rentabilidad.
- La conexión real con Alpaca y la ejecución paper todavía deben completarse.

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
| `filters.py` | Marca contratos no explotables y produce un reporte auditable. |
| `arbitrage.py` | Implementa seis detectores y calcula el edge neto. |
| `storage.py` | Escribe y lee Parquet particionado y manifiestos. |
| `backtest.py` | Simula señales, ejecuciones, cierres, costos y stop-loss. |
| `evaluation.py` | Hace split temporal, grid search, métricas y Sharpe deflactado. |
| `pipeline.py` | Expone la CLI: `demo`, `doctor`, `record`, `backfill`, `enrich` y `detect`. |

# 3. Revisión de literatura y aplicación

Esta parte todavía debe documentarse formalmente en el proyecto. La revisión puede organizarse en cuatro grupos.

## Modelo binomial

Cox, Ross y Rubinstein introducen el árbol binomial para valorar opciones. El precio evoluciona en pasos discretos hacia arriba o hacia abajo y se descuenta el valor esperado bajo una probabilidad riesgo-neutral.

La aplicación al caso es directa: RTX y BA tienen opciones americanas, por lo que el árbol permite comparar en cada nodo el valor de continuar con el valor de ejercer inmediatamente.

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

- 1.248 cotizaciones;
- 12 cortes temporales;
- 136 contratos;
- 2 subyacentes;
- 100% de las cotizaciones con spot alineado;
- 1.248 de 1.248 IV americanas invertidas;
- IV mediana cercana a 32,4%;
- rango aproximado entre 22,8% y 59,7%;
- cero filas fuera de las cotas de no arbitraje.

Estos números demuestran que la implementación funciona sobre el mercado sintético generado por el proyecto. No son resultados de mercado real.

## Qué falta para afirmar ajuste a la superficie observada

Hay que ejecutar el proceso sobre snapshots o datos históricos reales y presentar:

- gráfico de IV contra moneyness;
- gráfico de IV contra vencimiento;
- precio observado contra precio CRR;
- RMSE o MAE ponderado por vega;
- comparación entre IV americana, IV europea e IV del proveedor;
- análisis separado por RTX y BA.

El código ya contiene `calibrate_chain_sigma()` para una sigma global, pero todavía no hay en el repositorio una tabla real con esos resultados.

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

En la demo se conservaron 1.239 de 1.248 filas, un 99,3%. Nueve fueron excluidas por open interest insuficiente. Nuevamente, es un resultado sintético.

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
- stop-loss configurable.

La barrera `PointInTimeView` no permite pedir datos posteriores al reloj. Esto controla lookahead por construcción, no sólo por convención.

## Resultado de la demo

La demo sintética cubre una hora y tiene vencimientos varios meses después. Por eso las posiciones se cierran por fin de datos y no por vencimiento:

| Latencia | Operaciones | P&L aproximado |
|---:|---:|---:|
| 1 minuto | 9 | -USD 1.019 |
| 5 minutos | 9 | -USD 1.019 |
| 15 minutos | 6 | -USD 864 |

El resultado negativo es coherente con pagar el spread de entrada y salida sin llegar a la liquidación final. No es una medida de rentabilidad de la estrategia.

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

## Pendiente de resultados

Todavía falta ejecutar y documentar una búsqueda real con:

- grilla concreta;
- mejor configuración in-sample;
- desempeño out-of-sample;
- curva de equity;
- drawdown;
- hit rate;
- edge capture;
- Sharpe y DSR.

La infraestructura existe, pero la evidencia empírica todavía no está generada.

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

El stop-loss actual sólo existe dentro del backtest. Por defecto se configura como una pérdida no realizada de USD 500 por posición, pero no está conectado a una cuenta ni a un proceso en vivo.

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
| Obtención de datos | Código para Alpaca, pero conexión real no probada. |
| Limpieza y calidad | Implementada y testeada. |
| Calibración binomial | Implementada y testeada sobre datos sintéticos. |
| Detección de arbitrajes | Implementada y testeada. |
| Backtest realista | Implementado con costos, latencia y barrera anti-lookahead. |
| In-sample / out-of-sample | Infraestructura implementada; faltan resultados reales documentados. |
| Paper trading | Falta implementar envío y seguimiento de órdenes. |
| Monitoreo productivo | Falta implementar. |
| Stop-loss productivo | Sólo existe en el backtest. |
| P&L acumulado real | No disponible. |

# 14. Limitaciones técnicas

Las limitaciones deben declararse en la presentación:

- no hay bid/ask histórico completo de opciones en Alpaca;
- el feed gratuito está demorado;
- las barras históricas usan un spread proxy;
- no se limita aún el tamaño por profundidad real del libro;
- no se modela asignación anticipada;
- se usan dividendos continuos en lugar de dividendos discretos;
- se usa una tasa plana por defecto;
- una sigma global no representa todo el smile;
- una misma dislocación puede producir muchas oportunidades duplicadas;
- la historia previa al recorder conserva survivorship bias;
- el demo no es evidencia de rentabilidad;
- no existe todavía ejecución paper ni monitoreo en vivo.

# 15. Conclusiones

El proyecto construye una base rigurosa para investigar arbitraje de opciones. Su aporte principal no es afirmar que ya encontró una estrategia rentable, sino mostrar cómo debe validarse una señal antes de considerarla operable.

Las conclusiones técnicas son:

1. El modelo binomial americano es más apropiado que Black-Scholes europeo para las opciones analizadas.
2. Las relaciones model-free son la evidencia más fuerte porque no dependen de una sigma correcta.
3. El bid-ask, la latencia y la liquidez pueden eliminar completamente el edge teórico.
4. El control temporal de los datos es tan importante como la fórmula de valuación.
5. La ausencia de NBBO histórico limita fuertemente cualquier backtest retrospectivo.
6. La demo verifica la cañería, pero no prueba rentabilidad.
7. Antes de presentar P&L real hay que grabar datos propios, ejecutar paper trading y monitorear fills.

La conclusión general para la exposición puede formularse así:

> Una inconsistencia matemática no es automáticamente una oportunidad de trading. Para que sea explotable debe existir una cotización ejecutable, liquidez en todas las patas, una relación financiera correcta para opciones americanas, costos y latencia compatibles, y un resultado que sobreviva fuera de muestra.

# 16. Guion sugerido para la presentación

1. Presentar el objetivo y el flujo general del pipeline.
2. Explicar la limitación de datos de Alpaca y la diferencia entre snapshot real y barra con spread proxy.
3. Mostrar el control anti-lookahead y el esquema de calidad.
4. Introducir el árbol CRR y comparar IV americana contra europea.
5. Mostrar la superficie de volatilidad y el RMSE real cuando se genere con datos de mercado.
6. Explicar los cinco detectores model-free y el detector contra modelo.
7. Mostrar el reporte de filtros de liquidez.
8. Explicar la latencia, costos y barrera `PointInTimeView` del backtest.
9. Mostrar la tabla in-sample / out-of-sample y el DSR cuando se ejecute sobre datos reales.
10. Presentar órdenes paper, fills y P&L sólo después de implementar esa capa.
11. Cerrar con el diseño del monitor, stop-loss y limitaciones.

## Estado reproducible al preparar este informe

La suite ejecutada con Python 3.14 produjo 26 tests exitosos. El demo offline recorrió generación sintética, alineación, IV, filtros, Parquet, detección y backtest. La capa de red no se ejecutó porque no estaban configuradas las credenciales de Alpaca.
