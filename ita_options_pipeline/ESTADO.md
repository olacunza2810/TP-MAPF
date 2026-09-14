# Estado del proyecto — Grupo 3 (Defensa / ITA)

Documento de traspaso. Última actualización: 13 de septiembre de 2026.
Subyacentes: **RTX**, **BA** y **LMT**. Presentación Parte 1: **7 de septiembre**.

---

## Cómo correrlo en cinco minutos

Requiere Python 3.11 o superior. Verificado en Python 3.14.3.

```powershell
cd "ruta\a\ita_options_pipeline"
pip install -e ".[dev]"

python -m pytest                        # 58 tests, ~17 segundos
python -m ita_options.pipeline demo     # pipeline completo offline, ~50 segundos
```

Ninguno de los dos necesita credenciales de Alpaca. Si `pytest` dice
`58 passed`, todo lo que no depende de la red está funcionando.

Cuando haya claves:

```powershell
$env:ALPACA_API_KEY="..."
$env:ALPACA_SECRET_KEY="..."
python -m ita_options.pipeline doctor   # diagnóstico, 2 segundos
```

Si `python --version` no dice 3.14, usar `py -3.14 -m ...` en todos los comandos.

---

## Qué está terminado y verificado

| Etapa de la consigna | Estado |
|---|---|
| 2. Obtención de datos | Código completo, **sin probar contra la API real** |
| 3. Limpieza y control de calidad | Completo |
| 4. Calibración del binomial | Completo y testeado |
| 5. Detección de arbitrajes | Completo y testeado |
| 6. Backtest realista | Completo y testeado |
| 7. Split in-sample / out-of-sample | Completo y testeado |

## Qué falta

| | |
|---|---|
| 1. Revisión de literatura | **Sin empezar.** No depende de nada técnico. |
| 8. Ejecución en paper trading | No existe código de órdenes. |
| 9. Monitoreo y stop-loss en producción | El stop-loss existe dentro del backtest, no en vivo. |
| AI log | Sin armar. La consigna lo pide como entregable obligatorio. |

**Urgente y no técnico:** prender el recorder (`pipeline record`) apenas haya
credenciales. Construye la serie de datos hacia adelante y cada día sin grabar es
un día que no vamos a tener en la Parte 2.

---

## Los tres hallazgos que sirven para la presentación

### 1. Tres de los cuatro detectores del notebook de cátedra dan falsos positivos

Corrimos las pruebas de las celdas 39-42 sobre una cadena generada por el propio
modelo binomial, es decir **libre de arbitraje por construcción**:

```
mariposas (celda 41):    8 falsos positivos de 8 tríos
calendarios (celda 42):  10 de 10 strikes
monotonicidad (celda 40): correcta, pero sin filtro de cotizaciones nulas
```

La mariposa tiene el signo invertido: compra el cuerpo y vende las alas, un
portafolio de payoff no positivo. El calendario tiene las patas al revés y
termina detectando el bid-ask spread. Y los "mejores" arbitrajes del output
guardado del profesor tienen `ask = 0.00`, que es una cotización ausente, no un
call gratis.

Está todo corregido en `arbitrage.py` y documentado en los docstrings.

### 2. El feed gratuito viene demorado 15 minutos, y eso cambia todo

El backtest separa el instante de detección del de ejecución y revalúa las patas
con las cotizaciones del segundo. Sobre los mismos datos:

```
latencia  5 min:  P&L = +756 USD
latencia 20 min:  P&L = -143 USD
```

Con el feed `indicative` de Alpaca la latencia realista es de **900 segundos**.
Hay que correr el backtest con ese valor y mostrar el número.

### 3. Alpaca no tiene bid/ask histórico de opciones

Verificado contra el SDK: sólo existe `get_option_latest_quote`, no hay endpoint
con rango de fechas. El histórico de precios (`backfill`) sólo trae barras
OHLCV, sin NBBO, así que el spread hay que estimarlo.

Simulamos ese proxy sobre cadenas sin arbitraje. En contratos cerca del dinero
se porta bien. En calls OTM con strikes cada 1 dólar y spread del 10%:

```
62% de los días simulados muestran arbitrajes fantasma (media 3.5, máximo 20)
```

Y ese es justo el régimen donde un screen ingenuo cree encontrar oro.

---

## Limitaciones conocidas (declararlas en la presentación)

**El detector de mariposas no escala.** Evalúa todos los tríos de strikes, que
crece con el cubo. Sobre 945 contratos —el tamaño real de la cadena de RTX—
tarda 130 segundos de los 145 totales. El parámetro `consecutive_only=True` lo
baja a 0.2 segundos, pero pierde violaciones entre strikes salteados. **Si van a
hacer una demo en vivo con una cadena real, usen ese parámetro o se va a
colgar.**

**El backtest no limita el tamaño por profundidad del libro.** No usa
`bid_size` / `ask_size`, así que asume que se puede operar todo al precio
publicado. Infla el P&L, y justo en las oportunidades más atractivas.

**Una dislocación se reporta muchas veces.** Tres contratos mal cotizados
generaron 45 oportunidades: es la misma causa vista desde 45 ángulos. No hay
agrupación por contrato causante.

**No se modela la asignación anticipada**, que es el riesgo central de estas
estrategias: si ejercen la pata corta de una mariposa antes del vencimiento, la
cobertura se rompe.

**Una sola sigma contra un smile.** El detector contra modelo va a marcar
sistemáticamente las alas. Por eso está apagado por defecto.

**Dividendos continuos y tasa plana.** RTX paga dividendos discretos en fechas
concretas. Afecta sobre todo la prueba de paridad put-call.

---

## Sugerencia para armar la presentación

Son 25 minutos y la consigna dice explícitamente que **no** se evalúa el retorno,
sino la rigurosidad analítica y la claridad.

La diapositiva más fuerte que tenemos es la de validación: *"corrimos los
detectores sobre una cadena sin arbitraje generada por el propio modelo; tres de
los cuatro del material de cátedra disparaban; los corregimos, y el nuestro da
cero"*. Eso es control de calidad demostrable en una imagen.

La segunda es la de latencia: el mismo backtest, dos valores de demora, resultado
opuesto.

La tercera es la de datos: qué hay disponible, qué no, y en qué régimen el proxy
falla el 62% de las veces.

Las limitaciones de arriba conviene listarlas nosotros. Que un profesor las
encuentre es peor que decirlas.
