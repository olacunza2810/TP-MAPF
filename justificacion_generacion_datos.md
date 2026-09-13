# Generación de la muestra sintética: metodología y justificación

> Sección para incorporar al informe (reemplaza / amplía lo que hoy dice
> "generación sintética"). Responde la observación del profesor: la distribución
> de precios de la muestra debe parecerse lo más posible a la de los tres activos
> reales, y cada supuesto debe estar justificado.

## 1. Por qué generamos datos y qué debe cumplir la muestra

Alpaca **no expone bid/ask histórico de opciones**: sólo hay *latest quote* y
barras OHLCV, sin NBBO con rango de fechas. Sin un recorder corriendo desde hace
meses no existe una serie histórica de cotizaciones sobre la cual backtestear.
Por eso la muestra se genera sintéticamente. Pero una muestra sintética sólo es
un banco de pruebas válido si la **distribución de precios del subyacente
reproduce la del activo real**; de lo contrario, cualquier conclusión sobre
detección de arbitrajes y sobre el comportamiento del backtest no es
extrapolable. Fijamos tres requisitos:

1. **Coherencia interna.** La volatilidad realizada del camino simulado debe
   igualar a la volatilidad implícita con la que se valúan sus opciones. Si el
   subyacente "se mueve" a una vol y sus opciones se valúan a otra, el mercado es
   incoherente y se inventan (o se ocultan) señales.
2. **Parecido con el activo real.** No sólo el nivel de volatilidad: también la
   **asimetría** (skew) y las **colas** (curtosis) de los retornos, el
   **dividendo** y el **skew de la superficie de IV**, específicos de cada
   nombre.
3. **Ausencia de arbitraje por construcción.** La cadena se valúa con un único
   modelo consistente (árbol CRR sobre el smile), de modo que las únicas
   violaciones sean las que inyectamos a propósito para probar los detectores.

## 2. Diagnóstico del generador anterior

El generador original construía el subyacente con un GBM gaussiano de paso fijo
`N(0, 0.0009)` cada 5 minutos, con parámetros de RTX y BA puestos a mano. Medido
sobre su propia salida:

| Activo | Vol realizada del camino | IV con la que se valuaban sus opciones | Exceso de curtosis |
|---|---:|---:|---:|
| RTX | **12,5 %** | 24 % | −1,06 |
| BA  | **9,2 %**  | 34 % | −1,74 |

Tres problemas, todos los que marcó la corrección:

- **Incoherencia vol realizada / implícita.** El camino se movía a la mitad
  (RTX) o a un tercio (BA) de la vol con la que se valuaban sus opciones.
- **Orden invertido.** BA —el activo más volátil del par— simulaba *menos* vol
  que RTX. La muestra contradecía el ranking de volatilidad real.
- **Colas finas.** El exceso de curtosis era **negativo**, cuando los retornos
  de acciones tienen colas **gordas** (curtosis en exceso positiva). Además no
  había dividendo diferenciado calibrado ni skew específico por activo, ni un
  tercer activo.

## 3. Elección de los tres activos

El universo del trabajo son constituyentes individuales del ETF **ITA** (iShares
U.S. Aerospace & Defense), no el ETF. Sus mayores ponderaciones (sep-2026) son
GE Aerospace ≈ 21,5 %, RTX ≈ 17,2 %, Boeing ≈ 8,9 %, General Dynamics ≈ 4,8 % y
Lockheed Martin ≈ 4,7 %. Ya teníamos RTX y BA; el tercero es **LMT**, y la
elección está razonada:

- **GE Aerospace**, pese a ser la mayor ponderación y tener las opciones más
  líquidas del sector, se **escindió en tres empresas en abril de 2024** (GE
  Aerospace, GE Vernova, GE HealthCare). Su serie histórica previa **no
  corresponde a la entidad actual**, de modo que calibrar una distribución de
  retornos sobre ella sería metodológicamente incorrecto — exactamente lo que la
  consigna pide evitar.
- **LMT** es un *pure-play* de defensa con historia corporativa continua,
  opciones listadas profundas, y —lo más valioso para poner a prueba los
  detectores— un **régimen de volatilidad distinto** al de RTX y BA. Así la
  muestra cubre tres regímenes complementarios:
  - **BA**: la mayor volatilidad total, colas gordas y skew negativo marcado; sin
    dividendo (turnaround, 737 MAX). Turbulencia de continuo.
  - **RTX**: volatilidad media, dividendo bajo, y —de los tres— la distribución
    **más simétrica y de colas menos pesadas**. Saltos episódicos (recall del
    motor GTF de Pratt & Whitney, 2023).
  - **LMT**: la menor volatilidad de base pero con **colas gordas comparables a
    las de BA**, concentradas en saltos raros y severos (cargos por programas
    *fixed-price*, p. ej. 2025); dividendo alto. Explosivo sólo de vez en cuando.

Esa diversidad es deliberada: un detector de arbitrajes que sólo se prueba contra
un régimen no está validado. GD era la alternativa razonable; se prefirió LMT por
opciones más profundas y por aportar el régimen "vol baja + saltos severos", que
GD no representa tan nítidamente.

## 4. Modelo de retornos: difusión con saltos de Merton

Un GBM gaussiano no puede reproducir colas gordas ni asimetría. Elegimos una
**difusión con saltos de Merton** bajo la medida riesgo-neutral, porque con pocos
parámetros interpretables reproduce los *stylized facts* de los retornos de
acciones (colas gordas, skew negativo) y mantiene el pricing coherente. El
log-retorno de cada paso $\Delta t$ es

$$
r=\Big(k-q-\lambda\kappa-\tfrac12\sigma_{\text{diff}}^2\Big)\Delta t
  +\sigma_{\text{diff}}\sqrt{\Delta t}\,Z
  +\sum_{i=1}^{N_{\Delta t}} J_i,
$$

con $Z\sim\mathcal N(0,1)$, $N_{\Delta t}\sim\text{Poisson}(\lambda\Delta t)$,
$J_i\sim\mathcal N(\mu_J,\delta_J^2)$, $k$ la tasa libre de riesgo, $q$ el
dividendo continuo y $\kappa=e^{\mu_J+\delta_J^2/2}-1$ la corrección de
martingala (asegura que el activo descontado sea martingala pese a los saltos).

**La clave de la coherencia** (requisito 1): la varianza total anual del proceso
es

$$
\sigma_{\text{total}}^2=\sigma_{\text{diff}}^2+\lambda(\mu_J^2+\delta_J^2),
$$

y fijamos $\sigma_{\text{total}}$ igual a la **IV at-the-money** con la que se
valúan las opciones, despejando la vol difusiva
$\sigma_{\text{diff}}=\sqrt{\sigma_{\text{total}}^2-\lambda(\mu_J^2+\delta_J^2)}$.
Así la vol realizada del camino **iguala por construcción** a la implícita, y los
parámetros de salto controlan la *forma* de las colas sin descalibrar el nivel.

Por qué Merton y no algo más pesado (Heston, GARCH): la superficie de valuación
del proyecto es un árbol CRR con una $\sigma$ por contrato. Merton preserva esa
arquitectura y la interpretación uno-a-uno de la vol total con la IV; un modelo
de vol estocástica exigiría un pricer distinto y rompería la trazabilidad. Merton
es el mínimo modelo que corrige los tres defectos.

## 5. Parámetros por activo y su fundamento

Todos los valores objetivo se tomaron de fuentes públicas (13/09/2026) y viven en
un solo lugar (`ita_options/calibration.py`), tipados y con su nota de origen,
para re-estimarse con datos propios apenas el recorder tenga historia.

| Parámetro | RTX | BA | LMT | Fundamento |
|---|---:|---:|---:|---|
| Spot de referencia | 174 | 210 | 524 | Nivel de mercado sep-2026 |
| Dividendo continuo $q$ | 1,6 % | 0 % | 2,6 % | RTX 2,92 USD/acción; BA suspendido desde 2020; LMT 13,80 USD/acción |
| Vol total anual (= IV atm) | 28 % | 36 % | 26 % | Mezcla de IV30 (~30-32 / ~34 / ~32 %) y vol realizada trailing |
| Intensidad de saltos $\lambda$ | 6 | 8 | 4 | BA el más propenso a cracks; LMT saltos raros |
| Media del salto $\mu_J$ | −2,5 % | −3,5 % | −3,0 % | Los cracks de equities son a la baja (skew negativo) |
| Desvío del salto $\delta_J$ | 3,5 % | 5,5 % | 4,5 % | BA con los saltos más grandes |
| Skew del smile $\beta_1$ | 0,35 | 0,55 | 0,35 | BA con el skew de IV más pronunciado (riesgo de crack) |
| Curvatura $\beta_2$ | 0,90 | 1,30 | 0,85 | Sonrisa más marcada en BA |

Lógica financiera detrás de las diferencias:

- **Dividendo.** Entra en el drift riesgo-neutral y en la valuación CRR. Es un
  hecho objetivo de cada empresa, no un supuesto libre: BA no paga desde 2020;
  RTX y LMT sí, con yields distintos.
- **Nivel de volatilidad.** BA > RTX > LMT en vol de base refleja su realidad
  operativa (BA en turnaround, LMT defensa estable). Los tres tienen IV elevada
  en 2026, pero su distribución realizada difiere en las colas.
- **Saltos.** Reproducen los eventos idiosincrásicos: BA (737 MAX, pandemia,
  pérdidas trimestrales) tiene los saltos más frecuentes y grandes; RTX
  (shock del GTF) intermedios; LMT raros pero severos (cargos por programas).
- **Skew de la superficie.** Las puts OTM de BA se pagan más caras (protección
  contra caídas), de ahí el $\beta_1$ mayor.

## 6. Validación: la muestra reproduce cada activo

Sobre 24 réplicas independientes de 2.520 días cada una (los momentos de las
colas son ruidosos en una sola realización, así que se promedian):

| Activo | Vol objetivo | Vol simulada | Skew | Exceso de curtosis | Div |
|---|---:|---:|---:|---:|---:|
| RTX | 28 % | 28,0 % | −0,45 | +2,17 | 1,6 % |
| BA  | 36 % | 36,0 % | −0,97 | +6,30 | 0 % |
| LMT | 26 % | 26,0 % | −0,69 | +4,60 | 2,6 % |

Los tres cumplen: la vol simulada iguala al objetivo (= IV con la que se valúa),
el skew es negativo y el exceso de curtosis positivo (colas gordas). El
ordenamiento **robusto** (verificado sobre decenas de semillas) es: **RTX es el
más simétrico y de colas más finas**, mientras que **BA y LMT superan claramente
a RTX** en curtosis y en skew negativo. Entre BA y LMT no se afirma un orden
estricto: en promedio BA es el más extremo, pero LMT le rivaliza porque su riesgo
está concentrado en saltos raros y grandes — que es justo el hecho financiero que
queremos capturar. La figura `validacion_distribucion.png` muestra, por activo, el
histograma de retornos vs. la Normal ajustada (eje Y logarítmico, que hace
visibles las colas) y el Q-Q plot: la cola izquierda cae por debajo de la recta,
la firma gráfica de las colas gordas a la baja.

Esta comparación está **blindada por tests** (`tests/test_calibration.py`): si
alguien cambia un parámetro y rompe el parecido (vol fuera de objetivo, skew
positivo, colas finas, o RTX dejando de ser el más simétrico y tranquilo de los
tres), la suite falla.

## 7. Alcance y límite honesto

- Los niveles de vol/dividendo son de mercado a sep-2026; **cuando el recorder de
  Alpaca acumule historia se re-estiman de datos propios** — la calibración vive
  en un módulo aislado justamente para eso.
- El proceso es de **vol constante por activo** (con saltos): no modela
  *clustering* de volatilidad (GARCH/Heston). Es una decisión consciente para
  preservar la coherencia con el pricer CRR de una sola $\sigma$. Es la extensión
  natural si se quiere endurecer la muestra.
- La validación distribucional se hace sobre el **proceso diario** (miles de
  observaciones); la cadena intradía de la demo abarca una ventana corta cuyo fin
  es ejercitar la cañería de detección/backtest, no estimar una distribución. En
  esa ventana la vol realizada queda entre la difusiva y la total (más cerca de la
  total cuanto más grandes son los saltos del activo), lo que igual mantiene la
  coherencia con la IV de valuación.
