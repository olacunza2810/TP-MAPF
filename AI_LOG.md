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
