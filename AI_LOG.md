# AI log — Grupo 3 (Defensa / ITA)

The consigna asks us to record every interaction with AI: the prompt, the
response, and why we delegated that task. All the interactions below were with
**Claude** (Anthropic), one of the permitted models. Each entry records the
prompt word for word (or its opening, when it ran to several paragraphs), a
summary of the response, and our reason for delegating.

Criteria we followed throughout the project:

- **Delegated:** writing and testing code, auditing code we had inherited, and
  writing technical documentation, always with tests that pin down the expected
  behaviour.
- **Not delegated:** the investment decisions, choosing the underlyings, the
  hyperparameter grid (declared before seeing results), and reading the
  results. Every number in the report comes from a script in the repository,
  not from a chat answer.

> **To complete:** each member adds here any other AI interaction they had (for
> example, the August–September version Delfina and Tizi worked on, and any
> query about the report or the slides), with the same format.

---

## Session 1 — 3 to 5 September 2026: building the pipeline

Conversation: "Pipeline modular de opciones financieras con Alpaca" (Octavio, Claude).

**Opening prompt:**

> Rol y Objetivo: Actúa como un Quant Data Engineer Senior. Necesito construir
> un pipeline modular en Python utilizando el SDK `alpaca-py` para extraer y
> procesar un dataset de opciones financieras de los tickers RTX y BA. Este
> dataset alimentará un backtest de arbitraje sin sesgos (lookahead,
> survivorship). [...] Extracción con Alpaca [...] Cálculos locales
> (`mid_price`, `dte`, IV) [...] Filtros de liquidez (Volume > 0, Open
> Interest > 50, Bid > 0, spread < 15%) [...] Salida a `.parquet` [...] Usa type
> hints y docstrings.

**Response (summary):** it built the `ita-options` package (13 modules, about
4,000 lines): an asynchronous Alpaca gateway, a point-in-time recorder,
anti-lookahead alignment, the CRR binomial tree with IV inversion by Brent,
filters with an auditable report, six detectors and an event-driven backtest
with a lookahead barrier. It found and fixed bugs, including:

- the butterfly from the course notebook had its sign inverted;
- the calendar detector was measuring the spread;
- a `ask = 0` was being treated as a free call;
- a cross-contamination bug compared RTX against BA;
- an SDK field had the wrong name;
- the schema was dropping `is_tradable`.

It also replaced the $O(n^3)$ butterfly with a convex hull. In later turns of
the same conversation it added Polygon as a source, the daily mode, execution
at the open, the residual-edge check and the paper trading pilot (ledger,
`mleg` orders, reconciliation).

**Why we delegated:** writing and testing infrastructure code against an SDK
takes a lot of time and doesn't add to the financial analysis. We kept control
by requiring tests on synthetic chains that are arbitrage-free by
construction, against which every detector had to return zero signals.

## Session 2 — 7 September 2026: preparing the presentation

Conversation: "Preparación de presentación oral" (Octavio, Claude).

**Prompt:**

> Ya hice la presentacion. Te la mando. Yo voy a tener que decir la Slide 9,
> Slide 10 y Slide 11, por lo que necesito que veamos bien a fondo desde el
> principio los conceptos que aparecen en esas slides para poder explicarlo
> bien oralmente. [...]

**Response (summary):** it covered, layer by layer:

- replication and the risk-neutral measure;
- the CRR parameterization and early exercise;
- IV inversion;
- the difference between inverting IV contract by contract and calibrating a
  surface.

It flagged a risk: if each contract is priced with its own IV, the residual
of the model-based detector is zero by construction. It also noted that the
slides said 26 tests and that some formulas were cut off at the page edge.

**Why we delegated:** studying with a tutor to be able to defend the section
orally. We didn't generate content for the presentation in this session.

## Session 3 — 22 and 23 November 2026: incorporating the feedback on the interim submission

Conversation in the project "MAPF - Algo Trading & Arbitraje" (Octavio, Claude).

### 3.1 Analysis of the feedback

**Prompt:**

> Ya corrigieron esta primera parte del trabajo, y el profesor me dio las
> siguientes correcciones: [devolución completa del profesor, puntos 1 a 7]
> [...] Por ahora quiero que veas como estaba el trabajo y las correcciones que
> nos dio el profesor, analizalo todo [...]

**Response (summary):** it cloned the repository, read the report, the PDF and
the results, and mapped each point of the feedback onto the code. It found
one more bug in the reverse-conversion parity:

- the stock borrow cost was **added** to the credit instead of being
  subtracted;
- the stock half-spread had its sign inverted.

It also tallied the inconsistencies between the report, the PDF and
`ESTADO.md` (59, 98 and 102 tests; three different statuses for the
literature review and for paper trading).

**Why we delegated:** auditing about 8,500 lines against the feedback and
cross-checking numbers across documents is mechanical, and it's easy to miss
something.

### 3.2 Review of the version with corrections

**Prompts:**

> Ahora metete a esa carpeta ("TP-MAPF-main") y fijate si se corrigio
> correctamente lo que habia que corregir.

> A ver, metete aca ahora. Fijate si ahora podes.

> Ahora si, la movi a documents dentro de octav

**Response (summary):** it compared every file against the September version
and found that the core of the project (`arbitrage.py`, `backtest.py`,
`execution/`, the report and the PDFs) was unchanged. It also re-ran the
original detectors on the new DoltHub data: with real bid/ask, the only
violations were parity violations, and they clustered on days when the
option snapshot and the stock close weren't synchronous. It concluded that
points 2, 3, 4, 6 and 7 were not resolved and that 1 and 5 were partial.

**Why we delegated:** reviewing a parallel version of the repository file by
file.

### 3.3 Implementing the corrections

**Prompt:**

> Ok, vamos a hacer lo siguiente entonces: Vas a trabajar sobre el repo de
> Github que estaba antes de que le hicieran los cambios y vamos a corregir lo
> que nos dijo el profesor desde ese repo. Vos vas a hacer todo lo que tengas
> que hacer para hacer esas correcciones y para que quede bien [...] la idea es
> que vos puedas hacerlo autonomamente.

**Response (summary)**, working on the `correcciones-parte1` branch:

- **Data (point 1):** downloader for the DoltHub snapshot (181 sessions from
  March to November 2026, 95,290 rows with observed bid/ask), spot and
  dividends from Yahoo, and the ^IRX rate. A curation module with the
  implied-spot band to absorb the time gap between the snapshot and the
  stock close.
- **Parity (point 2):** American bounds with a discrete dividend, asymmetric
  funding, and the borrow and half-spread signs fixed. Dividends are
  point-in-time, using the declaration date from Polygon. Carry and early
  assignment in the backtest. Tests in `test_parity.py` and
  `test_dolthub_carry.py`.
- **Strategy for paper (points 3 and 6):** relative value against the
  calibrated binomial on verticals and butterflies, with multi-leg limit
  orders in maker mode and an exit on convergence. It plugs into the live
  loop, which already reconciled the ledger and applied the kill-switch.
- **Backtest and walk-forward (point 4):** backtest of that strategy over the
  daily snapshot with a conservative fill model. Walk-forward with
  calibration from March to June and month-by-month evaluation from July to
  November, for the binomial and for Black-Scholes.
- **Calibration (point 5):** a quadratic surface in log-moneyness per expiry,
  repricing error in sample and on the next session, and charts of IV
  against moneyness and maturity.
- **Report (point 7):** archiving the interim submission and a consolidated
  report with a single source for the numbers.

Two design errors came up during development and were fixed. Both are
documented in the code:

- the kill-switch measured the position at liquidation value, so it fired
  on the same day as the fill;
- the dividend projection didn't know about dividends declared before the
  sample.

**Why we delegated:** volume and coordination. The group defined the scope
(the professor's seven points) and approved the decisions that commit the
account: the API keys are loaded by us, outside the chat, in a `.env` file,
and we decided not to create a scheduled task, so paper trading is launched
by hand each session.

**Decisions made by us in this session:**

- work on the original repository and not on the parallel version;
- not to automate paper trading with a Windows scheduled task (prompt: "No
  crearla");
- to keep the hyperparameter grid declared before seeing results, and to
  report the sensitivity with more orders per session as exploratory and post
  hoc.

### 3.4 Final check and merge into `main`

**Prompts:**

> Perfecto, necesito que ahora corrobores x ultima vez si se hcieron las
> correcciones que nos dijo el profesor: [devolución repetida]

> Si, pushealo.

> Hacelo vos, quiero que sea el main a partir de ahora.

**Response (summary):** it went through the seven points again against the
code and the report, ran the tests and `check_informe.py`, pushed the branch
and made it `main`.

**Why we delegated:** a last cross-check before the corrected version became
the reference for the rest of the semester.

---

## Session 4 — 23 November 2026: Part 2 (Black-Scholes, Greeks and Monte Carlo)

Same conversation, in the project "MAPF - Algo Trading & Arbitraje" (Octavio, Claude).

### 4.1 Reading the brief and the class material, and the plan

**Prompts:**

> Bueno ahora arrancamos con la parte 2. Quiero que me hables en castellano.
> Primero vas a leer la consigna de lo que pide para la parte 2 que es el PDF
> q esta en el contexto del proyecto. Y luego cuando termines de leer eso, te
> vas a meter en la carpeta "TP-MAPF-correcciones" y dentro de esa carpeta vas
> a encontrar otra carpeta que se llama "material_de_clase_parte2" [...]

> Despreoucpate de eso por ahora. Armemos bien la parte 2 y luego vemos como
> hacemos con el tiempo que quede para ver cuantas ruedas llegamos a acumular.
> Comencemos con el plan

> 1. Lo que mas se acerque a los que dice la consigna. 2. Lo que mas se
> acerque a los que dice la consigna.

> Dale, primero quiero dejar todo listo y bien hecho y despues lo ponemos a
> correr todo junto.

**Response (summary):** it read the brief and the class material (stochastic
processes, Black-Scholes, Greeks, delta-hedging simulators) and proposed a
six-step plan. The two decisions it asked us about were answered with "as
close to the brief as possible":

- the binomial and Black-Scholes compete in the same paper account;
- Greek hedging runs live: delta with stock, and gamma with options.

**Why we delegated:** turning the brief into a step-by-step plan that reuses
the Part 1 pipeline. The scope and the two decisions were ours.

### 4.2 Steps 1 and 2: Black-Scholes in production and Greeks for both models

**Response (summary):**

- `CompetingModels`: both models propose structures in the same cycle, with a
  cap of 3 open structures per model and duplicates executed only once.
- `greeks.py`: Black-Scholes and the CRR tree with the same escrowed discrete
  dividend; tree Greeks taken from its nodes. It found a sign error in the put
  of the class Greeks table.

**Verification prompt:**

> Dale, pero primero quiero que corrobores que el paso 1 y 2 esten bien
> hechos.

**Response (summary):**

- It reproduced Hull's American put with a discrete dividend (4.4404 against
  4.44).
- It checked that the live loop and the backtest value the 26 August chain
  identically (maximum difference 0).
- It fixed two inconsistencies: the live smile used non-tradable rows, and
  the tree used 150 steps against 100 in the backtest.

### 4.3 Step 3: Black-Scholes against the binomial

**Prompts:**

> Dale, segui con el paso 3.

> [sobre "Recomiendo hacerlo: es el tratamiento que los datos respaldan y el
> que pidió la devolución del profesor."] Entonces, hacelo.

**Response (summary):**

- **Comparison** over 181 sessions and 36,515 contracts: early-exercise
  premium, repricing and a Diebold-Mariano test with Newey-West and HLN.
- **Results:** the binomial wins ($DM^* = 13.1$) where theory predicts, but
  the premium exceeds half the spread in only 0.3% of contracts.
- **Dividend treatment:** it added a third model (the tree with a continuous
  yield). The escrowed tree did better ($DM^* = 6.1$).
- **Change we approved:** the strategy moved to the escrowed tree (live,
  backtest and calibration). It recalculated the Part 1 numbers and updated
  the report; no conclusion changed.

**Why we delegated:** the statistical design and the computation. The
decision to change the pricer in production was ours, with the evidence in
front of us.

### 4.4 Step 4: Greek strategies

**Prompts:**

> Dale, segui con el paso 4.

> [sobre "Si igual la querés ejecutando, la habilito solo para compras de
> opciones"] Lo que mas se acerque a lo que pida la consigna.

**Response (summary):**

- **Hedging backtest** on DoltHub (1,749 positions, 3,272 position-days):
  - delta, delta-gamma and delta-vega, each with the Greeks of each model;
  - net effectiveness: delta 95%, delta-gamma 79%, delta-vega 63%;
  - the Greeks of the two models gave equivalent hedges.
- **Live hedger**, one book per model and underlying:
  - delta with stock, using a band;
  - gamma with the short-dated ATM call, in whole contracts and buys only,
    because options level 3 doesn't allow uncovered sales;
  - the books enter reconciliation and the daily loss, and the kill-switch
    closes them.

We chose to run delta-gamma in paper because the brief asks for the Greeks
in production, even though the backtest recommends delta.

### 4.5 Step 5: Monte Carlo stress test

**Prompt:**

> Dale, vamos con el paso 5.

**Response (summary):** a stress test of the book in production with:

- correlated GBM, Heston and Merton (calibrated and stressed);
- spreads tripled;
- deterministic shocks, and replays of COVID and of the April 2025 tariffs.

For calibration it used 10 years of Yahoo prices, downloaded on our machine,
and the ATM IV from DoltHub. It found:

- **Heston in the class notebook:** it diffuses the price with `Var[-1]`, so
  the stochastic volatility never reaches the price.
- **A production bug:** the live kill-switch and daily loss were measured at
  liquidation value. Every new structure would have been closed in the first
  cycle. They are now measured at the mid, as the backtest already did, and a
  regression test covers it.

**Why we delegated:** the simulation engine and the vectorised tree. The
stress assumptions (persistent mispricing, the volatility response in the
historical replays) are written down in the report as assumptions, not as
results.

### 4.6 Verification of steps 3 to 5

**Prompt:**

> Antes de seguir con el paso 6, quiero que corrobores que el paso 3, 4 y 5
> esten bien hechos y funcionen correctamente.

**Response (summary):**

- Reruns from scratch gave identical results.
- Diebold-Mariano matches `statsmodels`.
- Prices match independent pricers.
- Hedging P&L, recomputed straight from the quotes, matches.
- The stress-test accounting was checked by hand on a frozen market.
- The tail risk is stable across seeds (±4%).
- It fixed the stress paths so the dividend follows the same escrowed model
  as the pricer. The results did not change.

### 4.7 Step 6: decision, report, presentation and this log

**Prompt:**

> Dale, vamos con el paso 6.

**Response (summary):**

- `scorecard.py`, with criteria written down before the paper P&L exists.
- The consolidated report for the semester (sections 7 to 13), with 90
  numbers checked by `check_informe.py`.
- The Part 2 slides, and this entry.

**Why we delegated:** writing up results that come from scripts. The final
decision follows a rule written down before the paper results exist, and the
production criterion (paper P&L) is still pending.

**Decisions made by us in this session:**

- run both models in the same account and the Greeks in production;
- move the strategy to the escrowed tree;
- leave the paper run for the end, with everything ready;
- run delta-gamma in paper even though the evidence favours delta.

## Session 5 — 23 to 25 November 2026: historical simulation, production and submission

Same conversation (Octavio, Claude).

### 5.1 Paper trading or history? Both

**Prompts:**

> Que es lo que dice la consigna acerca de lo que habria que hacer en paper
> trading?

> En vez de hacerlo correr en vivo (Toma mucho tiempo y requiere que tenga mi
> compu prendida y activa), xq no lo hacemos correr con datos pasados para ver
> que nos encuentra ahi?

> Hagamos las dos. Pero quiero que la simulacion historica sea de mucho mucho
> tiempo atras, como si te dijiese 2019 hasta hoy si es posible (si no es
> posible desde 2019, entonces lo maximo que se pueda)

**Response (summary):**

- It explained that the brief requires live paper trading (the Black-Scholes
  model competing in production against the binomial), so a historical run
  complements it but does not replace it.
- It probed DoltHub and found that the option chain starts in January 2020
  (2019 is empty), with Monday/Wednesday/Friday snapshots until 2025 and RTX
  options only from April 2020. It downloaded 1.229 trading days (503.711
  quotes) on our computer.
- It wrote `ita_options/replay.py` and `scripts/run_replay.py`: the production
  system replayed snapshot by snapshot with the same signal functions, the
  competition caps, the Part 1 fill rule, the live risk rules and the three
  hedging modes, plus 5 tests.
- Result: 185 closed structures, about USD −100 each, every year and for both
  models; the model promised USD +11 and exiting at mid already loses USD 46 to
  68 (adverse selection). M4 on the simulation: $z = -0{,}21$.

**Why we delegated:** writing and testing the simulation engine and running
it. The idea of a historical run and the "as far back as possible" scope were
ours; the reading of the results is in the report.

### 5.2 Production on a free cloud machine

**Prompts:**

> Si me aseguras q es gratis, lo hacemos.

> Explicame bien como configurar el VM

> Que claves serian?

**Response (summary):**

- It checked Oracle's current Always Free terms (card for verification only,
  no charge without upgrading, 2 OCPU / 12 GB for Arm machines, idle machines
  can be stopped) and said it could not promise more than
  that.
- It wrote `deploy/oracle/setup_vm.sh` (packages, New York time zone, swap,
  repository, tests and `cron`), `paper_daily.sh` (daily run with the
  production settings) and `traer_resultados.ps1` (copies the reports and the
  ledger back), and guided the machine setup screen by screen.
- The first setup stopped silently at the `cron` step (an empty crontab plus
  `pipefail`); it fixed the script.
- Octavio created the account, the machine and the `.env` with the Alpaca
  paper keys himself; the AI never saw the keys.

**Why we delegated:** the deployment scripts and the step-by-step guide. The
account, the card and the keys stayed with us.

### 5.3 Summary for the presentation

**Prompt (opening):**

> Bueno, vamos con la presentacion ahora. Necesito que me hagas un resumen en
> PDF bien completo en el que expliques que es lo que hace el codigo, de que es
> capaz, como calcula cada cosa, en que se diferencia de lo que teniamos en la
> parte 1, donde esta corriendo, los resultados del backtest de 2020 hasta
> 2026 [...] Quiero que me digas que es lo que entendes de lo que te estoy
> pidiendo antes de q arranques. [...]

> este no, todo lo demas si. 1. Bien largo y con mucho grafico. 2. PDF y
> hacelo en latex para q se vea bien el pdf.

**Response (summary):** it listed what it understood and proposed extra
sections; we kept all but the slide-by-slide guide. It produced a 33-page
LaTeX document with 15 charts generated from the repository's result files
(`resumen_presentacion/`): what the system does, how each quantity is
computed, Part 1 against Part 2, the corrections in plain language, the Part 2
results, the 2020–2026 simulation, where it runs, limitations, likely
questions with answers, a glossary and sources.

**Why we delegated:** writing up and charting results that come from scripts,
so that the whole group can build the slides from one source.

### 5.4 First paper sessions: zero fills, and the fix

**Prompts:**

> Quiero ver como viene el Codigo a ver si hizo alguna operacion.

> Si, vamos con la 2. Primero corro ese comando q me pusiste arriba?

**Response (summary):**

- From the session reports Octavio pasted: on 23 and 24 November the system
  sent 155 orders and none filled. The maker order waited 180 seconds and was
  cancelled, and the wait blocked the cycle.
- It also found that the daily `git pull` on the machine failed because the
  scripts' execute bit counted as a local change; fixed in the repository.
- Option 2, chosen by us: the opening order now rests up to 30 minutes
  (`--rest-seconds 1800`) without blocking the cycle, occupies a slot like an
  open position, and is cancelled on expiry, on any halt and at the close.
  Seven new tests; it reaches the machine with the next daily pull.

**Why we delegated:** reading the logs and implementing the change with tests.
We chose among the three options it laid out.

### 5.5 Submission review

**Prompts:**

> Necesito que leas la consigna del TP y te fijes que es lo que hay que
> entregar, ademas del codigo y la presentacion.

> Quiero que primero revises que este todo OK con lo que ya esta hecho, en
> este caso el repo de Github. Quiero que lo analices y te fijes si eso ya
> estaria listo para entregar.

> Dale. / Dale, hacelo.

**Response (summary):**

- It listed the five deliverables of the brief and their status.
- It audited the repository from a clean clone: the 191 tests pass, the demo
  and every script run, and there are no keys in the 35 commits. It found
  outdated READMEs and `ESTADO.md`, undeclared plotting dependencies, a
  duplicated PDF, unused imports and missing docstrings, and fixed them.
- It added sections 12 (2020–2026 simulation) and 13 (paper in production) to
  the consolidated report, extended `check_informe.py` to 117 numbers checked
  against `reportes_dolthub/` and `reportes_replay/`, rebuilt the PDF and wrote
  this entry.

**Why we delegated:** an independent check of the repository against the
brief, and documentation of results that come from scripts.

**Decisions made by us in this session:**

- run both the historical simulation and the live paper trading;
- take the history as far back as the data allow;
- deploy on a free cloud machine instead of leaving our computer on;
- let maker orders rest 30 minutes instead of keeping the 3-minute wait.
