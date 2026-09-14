"""Detección de arbitrajes sobre la cadena de opciones.

Estructura tomada del notebook de cátedra *Calibration of the Binomial Pricing
Model* (celdas 35-42), con tres correcciones de signo y la incorporación de
costos de ejecución. Cada detector documenta la relación teórica que testea y la
diferencia respecto de la versión original.

Dos familias:

**Model-free.** Relaciones que se cumplen por no arbitraje puro, sin supuesto de
modelo. Si se violan, hay arbitraje aunque el binomial esté mal calibrado. Son
la evidencia más robusta que se puede presentar.

**Model-based.** Comparación contra el árbol CRR calibrado. Depende de que
:math:`\\sigma` sea correcta, así que es evidencia más débil y hay que
presentarla como tal.

Convención de lados
-------------------
Se **compra al ask** y se **vende al bid**, siempre. Todo test se plantea como
"¿puedo armar esta posición cobrando plata?", y el spread queda internalizado en
el planteo en vez de restarse después. Un test que use el mid o el ``lastPrice``
no mide una oportunidad ejecutable.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Iterator, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .volatility import crr_american_price

_LOG = logging.getLogger(__name__)

__all__ = [
    "ExecutionCosts",
    "detect_vertical_monotonicity",
    "detect_vertical_spread_bound",
    "detect_butterfly_convexity",
    "detect_put_call_parity",
    "detect_calendar",
    "calibrate_chain_sigma",
    "detect_model_dislocation",
    "run_all_detectors",
    "execution_edge",
]

DetectorName = Literal[
    "monotonicity",
    "vertical_bound",
    "butterfly",
    "put_call_parity",
    "calendar",
    "model_dislocation",
]

_RESULT_COLUMNS: tuple[str, ...] = (
    "detector",
    "underlying",
    "timestamp",
    "option_type",
    "expiration",
    "strikes",
    "legs",
    "gross_credit",
    "commission",
    "net_edge",
    "net_edge_usd",
    "min_volume",
    "min_open_interest",
    "max_spread_rel",
    "symbols",
    "leg_spec",
)


@dataclass(frozen=True, slots=True)
class ExecutionCosts:
    """Costos de ejecución de una estrategia multi-pata.

    El bid-ask ya está incorporado en cada test por la convención de lados, así
    que acá sólo entran los costos explícitos.

    Attributes:
        commission_per_contract: Comisión por contrato y por pata. IBKR cobra
            del orden de 0.65 USD; Alpaca no cobra comisión en opciones, pero
            asumir cero hace que el backtest no sea trasladable a IBKR.
        multiplier: Acciones por contrato. 100 para opciones sobre acciones
            estadounidenses.
        min_net_edge: Ganancia neta mínima por contrato, en USD, para reportar
            una oportunidad. Filtra las violaciones de un centavo, que son ruido
            de redondeo del feed y no sobreviven al slippage.
        stock_borrow_rate: Costo anual de shortear el subyacente. Sólo aplica a
            las estrategias que incluyen pata de acción (paridad put-call).
    """

    commission_per_contract: float = 0.65
    multiplier: float = 100.0
    min_net_edge: float = 5.0
    stock_borrow_rate: float = 0.0050


def _empty_result() -> pd.DataFrame:
    """DataFrame vacío con el esquema canónico de oportunidades."""
    return pd.DataFrame(columns=list(_RESULT_COLUMNS))


def _finalize(rows: list[dict[str, object]], costs: ExecutionCosts) -> pd.DataFrame:
    """Aplica el umbral de edge mínimo y ordena por atractivo."""
    if not rows:
        return _empty_result()
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame["net_edge_usd"] >= costs.min_net_edge]
    if frame.empty:
        return _empty_result()
    return frame.sort_values("net_edge_usd", ascending=False).reset_index(drop=True)


def _prepare(chain: pd.DataFrame, tradable_only: bool) -> pd.DataFrame:
    """Normaliza y filtra la cadena antes de aplicar los detectores.

    El filtro por ``is_tradable`` es lo que evita la clase de falso positivo más
    común: contratos con bid o ask en cero. En el notebook original, las
    oportunidades de mayor magnitud tenían ``ask = 0.00``, que no es un call
    gratis sino una cotización ausente.

    Args:
        chain: Cadena curada.
        tradable_only: Si ``True``, conserva sólo filas que pasaron los filtros
            de liquidez.

    Returns:
        Cadena lista para los detectores.
    """
    required = {"strike", "bid", "ask", "option_type", "expiration"}
    missing = required - set(chain.columns)
    if missing:
        raise KeyError(f"Columnas ausentes en la cadena: {sorted(missing)}")

    out = chain.copy()
    if tradable_only:
        if "is_tradable" not in out.columns:
            raise KeyError(
                "Se pidió tradable_only=True pero la cadena no trae 'is_tradable'. "
                "Correr filters.apply_liquidity_filters antes de detectar, o pasar "
                "tradable_only=False de forma explícita. Saltear el filtro en "
                "silencio produciría oportunidades sobre contratos inejecutables."
            )
        out = out.loc[out["is_tradable"]]
    for column in ("strike", "bid", "ask", "volume", "open_interest", "spread_rel"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["strike", "bid", "ask"])
    return out.loc[(out["bid"] > 0) & (out["ask"] > 0)]


def _leg_stats(rows: Sequence[pd.Series]) -> dict[str, object]:
    """Agrega los descriptores de liquidez del peor eslabón de la cadena.

    Una estrategia multi-pata es tan ejecutable como su pata menos líquida, así
    que se reporta el mínimo de volumen y open interest y el máximo de spread
    relativo, no el promedio.
    """
    def _extreme(column: str, reducer) -> float:
        # Con velas diarias el open interest nunca existe: devolver NaN sin la
        # advertencia de numpy por cada oportunidad.
        values = np.array([r.get(column, np.nan) for r in rows], dtype=float)
        values = values[np.isfinite(values)]
        return float(reducer(values)) if values.size else float("nan")

    return {
        "min_volume": _extreme("volume", np.min),
        "min_open_interest": _extreme("open_interest", np.min),
        "max_spread_rel": _extreme("spread_rel", np.max),
        "symbols": "|".join(str(r.get("symbol", "")) for r in rows),
    }


def _opt_leg(row: pd.Series, qty: float, price: float) -> dict[str, object]:
    """Describe una pata de opción con signo, precio de ejecución y contrato.

    Args:
        row: Fila de la cadena correspondiente al contrato.
        qty: Cantidad con signo. Positivo es largo, negativo es corto.
        price: Precio efectivo de ejecución: ask si se compra, bid si se vende.

    Returns:
        Diccionario con la descripción de la pata.
    """
    return {
        "kind": "option",
        "symbol": row.get("symbol"),
        "qty": float(qty),
        "price": float(price),
        "strike": float(row["strike"]),
        "option_type": str(row["option_type"]).lower()[:1],
        "expiration": row["expiration"],
    }


def _stock_leg(qty: float, price: float, symbol: str) -> dict[str, object]:
    """Describe una pata de acción, expresada en unidades de un contrato."""
    return {
        "kind": "stock",
        "symbol": symbol,
        "qty": float(qty),
        "price": float(price),
        "strike": float("nan"),
        "option_type": "",
        "expiration": pd.NaT,
    }


def _record(
    detector: DetectorName,
    rows: Sequence[pd.Series],
    gross_credit: float,
    n_contracts: float,
    legs: str,
    costs: ExecutionCosts,
    option_type: str,
    expiration: object,
    leg_spec: Sequence[dict[str, object]] = (),
) -> dict[str, object]:
    """Construye una fila de resultado con el edge neto de comisiones.

    ``leg_spec`` describe las patas de forma estructurada, con signo y precio de
    ejecución por pata. Es lo que consume el backtest: la columna ``legs`` es
    para leer, ``leg_spec`` es para operar.
    """
    commission = costs.commission_per_contract * n_contracts
    net_edge_usd = gross_credit * costs.multiplier - commission
    head = rows[0]
    return {
        "detector": detector,
        "underlying": head.get("underlying"),
        "timestamp": head.get("timestamp"),
        "option_type": option_type,
        "expiration": expiration,
        "strikes": "/".join(f"{float(r['strike']):g}" for r in rows),
        "legs": legs,
        "gross_credit": gross_credit,
        "commission": commission,
        "net_edge": gross_credit - commission / costs.multiplier,
        "net_edge_usd": net_edge_usd,
        "leg_spec": tuple(leg_spec),
        **_leg_stats(rows),
    }


def _by_expiry(
    chain: pd.DataFrame, option_type: str
) -> Iterator[tuple[object, pd.DataFrame]]:
    """Itera la cadena agrupada por subyacente y vencimiento.

    Agrupar sólo por vencimiento compararía contratos de acciones distintas
    entre sí: un call de BA con strike 190 contra uno de RTX con el mismo
    strike. Como las acciones cotizan a precios distintos, el mismo strike
    representa moneyness distintos y toda relación de no arbitraje se rompe.
    Las relaciones testeadas valen **dentro** de un subyacente, nunca entre
    subyacentes.
    """
    subset = chain.loc[chain["option_type"].astype(str).str.lower().str[0] == option_type]
    for _, group in subset.groupby(["underlying", "expiration"], sort=True):
        yield group["expiration"].iloc[0], group.sort_values("strike").reset_index(
            drop=True
        )


# --------------------------------------------------------------------------- #
# 1. Monotonicidad
# --------------------------------------------------------------------------- #


def detect_vertical_monotonicity(
    chain: pd.DataFrame,
    costs: ExecutionCosts | None = None,
    tradable_only: bool = True,
) -> pd.DataFrame:
    r"""Testea la monotonicidad del precio en el strike.

    El precio de un call no puede crecer con el strike, ni el de una put
    decrecer:

    .. math::

        K_1 < K_2 \;\Longrightarrow\; C(K_1) \ge C(K_2),
        \qquad P(K_1) \le P(K_2)

    Con lados de mercado, para calls hay arbitraje si

    .. math::

        \texttt{bid}(K_2) > \texttt{ask}(K_1)

    porque se compra :math:`C(K_1)` al ask, se vende :math:`C(K_2)` al bid, se
    cobra un crédito y el payoff del bull spread nunca es negativo.

    Respecto del notebook: la dirección del test es la misma (celda 40). Lo que
    se agrega es el filtro de cotizaciones nulas y la extensión a todos los
    pares :math:`K_1 < K_2`, no sólo strikes consecutivos: una violación puede
    aparecer entre strikes salteados aunque cada par contiguo se cumpla.

    Args:
        chain: Cadena curada de un instante.
        costs: Costos de ejecución.
        tradable_only: Restringir a filas que pasaron los filtros de liquidez.

    Returns:
        DataFrame de oportunidades en el esquema canónico.
    """
    costs = costs or ExecutionCosts()
    data = _prepare(chain, tradable_only)
    rows: list[dict[str, object]] = []

    for kind in ("c", "p"):
        for expiry, group in _by_expiry(data, kind):
            for i, j in itertools.combinations(range(len(group)), 2):
                low, high = group.loc[i], group.loc[j]
                if kind == "c":
                    # se compra el strike bajo al ask y se vende el alto al bid
                    credit = float(high["bid"]) - float(low["ask"])
                    legs = f"+C({low['strike']:g})@ask -C({high['strike']:g})@bid"
                    pair = (low, high)
                    spec = (
                        _opt_leg(low, +1.0, float(low["ask"])),
                        _opt_leg(high, -1.0, float(high["bid"])),
                    )
                else:
                    credit = float(low["bid"]) - float(high["ask"])
                    legs = f"+P({high['strike']:g})@ask -P({low['strike']:g})@bid"
                    pair = (high, low)
                    spec = (
                        _opt_leg(high, +1.0, float(high["ask"])),
                        _opt_leg(low, -1.0, float(low["bid"])),
                    )
                if credit > 0:
                    rows.append(
                        _record(
                            "monotonicity", pair, credit, 2.0, legs, costs,
                            kind, expiry, spec,
                        )
                    )
    return _finalize(rows, costs)


# --------------------------------------------------------------------------- #
# 2. Cota del spread vertical
# --------------------------------------------------------------------------- #


def detect_vertical_spread_bound(
    chain: pd.DataFrame,
    costs: ExecutionCosts | None = None,
    tradable_only: bool = True,
) -> pd.DataFrame:
    r"""Testea la cota superior del spread vertical.

    La monotonicidad acota el spread por abajo; esta relación lo acota por
    arriba. El payoff máximo de un bull call spread es :math:`K_2 - K_1`, así
    que su precio no puede excederlo:

    .. math::

        C(K_1) - C(K_2) \le K_2 - K_1

    Hay arbitraje si :math:`\texttt{bid}(K_1) - \texttt{ask}(K_2) > K_2 - K_1`:
    se vende el spread por más de lo que puede llegar a valer.

    Este test no está en el notebook. Es el complemento natural de la
    monotonicidad y sale gratis con los mismos datos.

    Nota sobre descuento: para opciones americanas la cota va sin descontar,
    porque el ejercicio anticipado permite realizar :math:`K_2 - K_1` en
    cualquier momento. Usar :math:`(K_2-K_1)e^{-r\tau}` sería la versión
    europea, más laxa y por lo tanto menos conservadora.

    Args:
        chain: Cadena curada de un instante.
        costs: Costos de ejecución.
        tradable_only: Restringir a filas explotables.

    Returns:
        DataFrame de oportunidades.
    """
    costs = costs or ExecutionCosts()
    data = _prepare(chain, tradable_only)
    rows: list[dict[str, object]] = []

    for kind in ("c", "p"):
        for expiry, group in _by_expiry(data, kind):
            for i, j in itertools.combinations(range(len(group)), 2):
                low, high = group.loc[i], group.loc[j]
                width = float(high["strike"]) - float(low["strike"])
                if kind == "c":
                    credit = float(low["bid"]) - float(high["ask"]) - width
                    legs = (
                        f"-C({low['strike']:g})@bid +C({high['strike']:g})@ask "
                        f"(colateral {width:g})"
                    )
                    spec = (
                        _opt_leg(low, -1.0, float(low["bid"])),
                        _opt_leg(high, +1.0, float(high["ask"])),
                    )
                else:
                    credit = float(high["bid"]) - float(low["ask"]) - width
                    legs = (
                        f"-P({high['strike']:g})@bid +P({low['strike']:g})@ask "
                        f"(colateral {width:g})"
                    )
                    spec = (
                        _opt_leg(high, -1.0, float(high["bid"])),
                        _opt_leg(low, +1.0, float(low["ask"])),
                    )
                if credit > 0:
                    rows.append(
                        _record(
                            "vertical_bound", (low, high), credit, 2.0, legs,
                            costs, kind, expiry, spec,
                        )
                    )
    return _finalize(rows, costs)


# --------------------------------------------------------------------------- #
# 3. Convexidad
# --------------------------------------------------------------------------- #


def detect_butterfly_convexity(
    chain: pd.DataFrame,
    costs: ExecutionCosts | None = None,
    tradable_only: bool = True,
    consecutive_only: bool = False,
) -> pd.DataFrame:
    r"""Testea la convexidad del precio en el strike.

    Para :math:`K_1 < K_2 < K_3` y :math:`w = \frac{K_3-K_2}{K_3-K_1}`, la
    convexidad del payoff implica

    .. math::

        C(K_2) \le w\,C(K_1) + (1-w)\,C(K_3)

    **Corrección respecto del notebook (celda 41).** El original testea

    .. math::

        \texttt{ask}(K_2) - \big[w\,\texttt{bid}(K_1)
            + (1-w)\,\texttt{bid}(K_3)\big] < 0

    es decir, comprar el cuerpo y vender las alas. Ese portafolio tiene payoff
    :math:`\le 0` en todo estado del mundo, así que cobrar por armarlo no es un
    arbitraje: es recibir plata hoy a cambio de pagar al vencimiento. El test
    dispara sobre cualquier cadena convexa, incluso con spread cero.

    La mariposa correcta es **larga las alas y corta el cuerpo**, que es la que
    tiene payoff :math:`\ge 0`. Hay arbitraje si

    .. math::

        w\,\texttt{ask}(K_1) + (1-w)\,\texttt{ask}(K_3) < \texttt{bid}(K_2)

    Args:
        chain: Cadena curada de un instante.
        costs: Costos de ejecución.
        tradable_only: Restringir a filas explotables.
        consecutive_only: Si ``True``, sólo tríos de strikes contiguos, como en
            el notebook. Por defecto se evalúan todos los tríos: una violación
            de convexidad puede quedar oculta entre strikes salteados.

    Returns:
        DataFrame de oportunidades.
    """
    costs = costs or ExecutionCosts()
    data = _prepare(chain, tradable_only)
    rows: list[dict[str, object]] = []

    for kind in ("c", "p"):
        for expiry, group in _by_expiry(data, kind):
            n = len(group)
            triplets = (
                ((i, i + 1, i + 2) for i in range(n - 2))
                if consecutive_only
                else itertools.combinations(range(n), 3)
            )
            for i, j, k in triplets:
                low, mid, high = group.loc[i], group.loc[j], group.loc[k]
                k1, k2, k3 = (float(r["strike"]) for r in (low, mid, high))
                if not k1 < k2 < k3:
                    continue
                w = (k3 - k2) / (k3 - k1)
                wing_cost = w * float(low["ask"]) + (1.0 - w) * float(high["ask"])
                credit = float(mid["bid"]) - wing_cost
                if credit > 0:
                    label = "C" if kind == "c" else "P"
                    legs = (
                        f"+{w:.2f}{label}({k1:g})@ask "
                        f"+{1-w:.2f}{label}({k3:g})@ask "
                        f"-1{label}({k2:g})@bid"
                    )
                    spec = (
                        _opt_leg(low, +w, float(low["ask"])),
                        _opt_leg(high, +(1.0 - w), float(high["ask"])),
                        _opt_leg(mid, -1.0, float(mid["bid"])),
                    )
                    rows.append(
                        _record(
                            "butterfly", (low, mid, high), credit, 3.0, legs,
                            costs, kind, expiry, spec,
                        )
                    )
    return _finalize(rows, costs)


# --------------------------------------------------------------------------- #
# 4. Paridad put-call
# --------------------------------------------------------------------------- #


def detect_put_call_parity(
    chain: pd.DataFrame,
    costs: ExecutionCosts | None = None,
    tradable_only: bool = True,
    underlying_spread: float = 0.01,
) -> pd.DataFrame:
    r"""Testea la paridad put-call en su forma americana.

    **Corrección respecto del notebook (celda 39).** El original plantea la
    igualdad europea

    .. math::

        C - P = S_0 e^{-q\tau} - K e^{-r\tau}

    que **no vale** para opciones americanas: el derecho a ejercicio anticipado
    rompe la igualdad y sólo sobreviven las desigualdades

    .. math::

        S e^{-q\tau} - K \;\le\; C_A - P_A \;\le\; S - K e^{-r\tau}

    Como RTX y BA son americanas, aplicar la versión europea genera falsos
    positivos sistemáticos, concentrados justamente en los contratos ITM donde
    la prima de ejercicio anticipado es mayor.

    Violación superior (conversión): se vende el call al bid, se compra la put
    al ask y se compra la acción al ask.

    Violación inferior (conversión reversa): se compra el call al ask, se vende
    la put al bid y se shortea la acción al bid, pagando el costo de borrow.

    Args:
        chain: Cadena curada con ``underlying_price``, ``tau``,
            ``risk_free_rate`` y ``dividend_yield``.
        costs: Costos de ejecución.
        tradable_only: Restringir a filas explotables.
        underlying_spread: Spread absoluto asumido sobre el subyacente, en USD.
            RTX y BA cotizan típicamente con un tick de diferencia.

    Returns:
        DataFrame de oportunidades.
    """
    costs = costs or ExecutionCosts()
    data = _prepare(chain, tradable_only)
    needed = {"underlying_price", "tau", "risk_free_rate", "dividend_yield"}
    if not needed.issubset(data.columns):
        _LOG.warning("Paridad omitida: faltan %s", sorted(needed - set(data.columns)))
        return _empty_result()

    rows: list[dict[str, object]] = []
    kind = data["option_type"].astype(str).str.lower().str[0]
    calls = data.loc[kind == "c"]
    puts = data.loc[kind == "p"]

    merged = calls.merge(
        puts, on=["underlying", "expiration", "strike"], suffixes=("_c", "_p")
    )
    for _, row in merged.iterrows():
        spot = float(row["underlying_price_c"])
        strike = float(row["strike"])
        tau = float(row["tau_c"])
        rate = float(row["risk_free_rate_c"])
        div = float(row["dividend_yield_c"])
        if not np.isfinite([spot, tau, rate, div]).all() or tau <= 0:
            continue

        half = underlying_spread / 2.0
        borrow = spot * costs.stock_borrow_rate * tau
        call_row = pd.Series({
            "symbol": row.get("symbol_c"), "strike": strike,
            "option_type": "c", "expiration": row["expiration"],
        })
        put_row = pd.Series({
            "symbol": row.get("symbol_p"), "strike": strike,
            "option_type": "p", "expiration": row["expiration"],
        })

        upper = (spot - half) - strike * np.exp(-rate * tau)
        credit_conv = float(row["bid_c"]) - float(row["ask_p"]) - upper
        if credit_conv > 0:
            rows.append(
                _record(
                    "put_call_parity",
                    (row.rename(lambda c: c.removesuffix("_c")),),
                    credit_conv,
                    2.0,
                    f"conversión: -C({strike:g})@bid +P({strike:g})@ask +S@ask",
                    costs,
                    "cp",
                    row["expiration"],
                    (
                        _opt_leg(call_row, -1.0, float(row["bid_c"])),
                        _opt_leg(put_row, +1.0, float(row["ask_p"])),
                        _stock_leg(+1.0, spot + half, str(row.get("underlying", ""))),
                    ),
                )
            )

        lower = (spot + half) * np.exp(-div * tau) - strike + borrow
        credit_rev = lower - (float(row["ask_c"]) - float(row["bid_p"]))
        if credit_rev > 0:
            rows.append(
                _record(
                    "put_call_parity",
                    (row.rename(lambda c: c.removesuffix("_c")),),
                    credit_rev,
                    2.0,
                    f"conversión reversa: +C({strike:g})@ask -P({strike:g})@bid -S@bid",
                    costs,
                    "cp",
                    row["expiration"],
                    (
                        _opt_leg(call_row, +1.0, float(row["ask_c"])),
                        _opt_leg(put_row, -1.0, float(row["bid_p"])),
                        _stock_leg(-1.0, spot - half, str(row.get("underlying", ""))),
                    ),
                )
            )
    return _finalize(rows, costs)


# --------------------------------------------------------------------------- #
# 5. Calendario
# --------------------------------------------------------------------------- #


def detect_calendar(
    chain: pd.DataFrame,
    costs: ExecutionCosts | None = None,
    tradable_only: bool = True,
) -> pd.DataFrame:
    r"""Testea la monotonicidad del precio en el vencimiento.

    Una opción americana con más tiempo contiene todos los derechos de una con
    menos tiempo, así que no puede valer menos:

    .. math::

        T_2 > T_1 \;\Longrightarrow\; C(T_2, K) \ge C(T_1, K)

    **Corrección respecto del notebook (celda 42).** El original testea

    .. math::

        \texttt{ask}_{\text{short}} > \texttt{bid}_{\text{long}}

    que corresponde a comprar la corta y vender la larga: payoff :math:`\le 0`.
    Además, como se compra al ask y se vende al bid, esa desigualdad se cumple
    por construcción cada vez que hay spread. En el output guardado del
    notebook todos los hits tienen costo positivo de aproximadamente 1.80 con
    vencimientos separados por un día: es el bid-ask, no una dislocación.

    El test correcto es :math:`\texttt{ask}_{\text{long}} <
    \texttt{bid}_{\text{short}}`: comprar la larga y vender la corta con
    crédito.

    Args:
        chain: Cadena curada de un instante, con varios vencimientos.
        costs: Costos de ejecución.
        tradable_only: Restringir a filas explotables.

    Returns:
        DataFrame de oportunidades.
    """
    costs = costs or ExecutionCosts()
    data = _prepare(chain, tradable_only)
    rows: list[dict[str, object]] = []

    kind_series = data["option_type"].astype(str).str.lower().str[0]
    for kind in ("c", "p"):
        subset = data.loc[kind_series == kind]
        # Igual que en _by_expiry: el mismo strike en dos acciones distintas no
        # es comparable, así que el subyacente entra en la clave de agrupación.
        for (_, strike), group in subset.groupby(["underlying", "strike"]):
            group = group.sort_values("expiration").reset_index(drop=True)
            for i, j in itertools.combinations(range(len(group)), 2):
                short_leg, long_leg = group.loc[i], group.loc[j]
                if pd.Timestamp(long_leg["expiration"]) <= pd.Timestamp(
                    short_leg["expiration"]
                ):
                    continue
                credit = float(short_leg["bid"]) - float(long_leg["ask"])
                if credit > 0:
                    label = "C" if kind == "c" else "P"
                    legs = (
                        f"+{label}(T_long,{strike:g})@ask "
                        f"-{label}(T_short,{strike:g})@bid"
                    )
                    spec = (
                        _opt_leg(long_leg, +1.0, float(long_leg["ask"])),
                        _opt_leg(short_leg, -1.0, float(short_leg["bid"])),
                    )
                    rows.append(
                        _record(
                            "calendar", (short_leg, long_leg), credit, 2.0, legs,
                            costs, kind,
                            f"{short_leg['expiration']}->{long_leg['expiration']}",
                            spec,
                        )
                    )
    return _finalize(rows, costs)


# --------------------------------------------------------------------------- #
# 6. Calibración global y dislocación contra modelo
# --------------------------------------------------------------------------- #


def calibrate_chain_sigma(
    chain: pd.DataFrame,
    steps: int = 128,
    sigma_grid: tuple[float, float, int] = (0.05, 1.50, 60),
    weight_by_vega: bool = True,
) -> float:
    r"""Calibra un único :math:`\sigma` que minimiza el error sobre la cadena.

    Implementa el ejercicio que el notebook propone al final de la celda 34:
    una volatilidad implícita "global" que minimiza la distancia entre la curva
    de precios del modelo y la de mercado sobre toda la cadena, en vez de una
    IV por contrato.

    .. math::

        \hat{\sigma} = \arg\min_\sigma \sum_i \omega_i
            \big(V^{\text{CRR}}_i(\sigma) - m_i\big)^2

    El ponderador :math:`\omega_i` por vega evita que los contratos deep OTM,
    cuyo precio es casi insensible a :math:`\sigma`, dominen el ajuste por ser
    numerosos. Sin ponderar, la calibración se sesga hacia las alas.

    Se resuelve por búsqueda en grilla y no por Newton: la función objetivo
    sobre una cadena real tiene mínimos locales, y una grilla gruesa es más
    robusta y más fácil de defender que un optimizador que puede quedar
    atrapado.

    Args:
        chain: Cadena curada de un instante y un vencimiento.
        steps: Pasos del árbol CRR.
        sigma_grid: Tupla ``(min, max, puntos)`` de la grilla de búsqueda.
        weight_by_vega: Ponderar el error por vega.

    Returns:
        El :math:`\hat{\sigma}` óptimo, o ``nan`` si no hay datos suficientes.
    """
    usable = chain.dropna(
        subset=["mid_price", "underlying_price", "strike", "tau"]
    )
    usable = usable.loc[usable["tau"] > 0]
    if usable.empty:
        return float("nan")

    lo, hi, points = sigma_grid
    grid = np.linspace(lo, hi, points)
    weights = (
        usable["vega"].fillna(0.0).to_numpy()
        if weight_by_vega and "vega" in usable.columns
        else np.ones(len(usable))
    )
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones(len(usable))

    best_sigma, best_error = float("nan"), np.inf
    for sigma in grid:
        errors = np.empty(len(usable))
        for idx, (_, row) in enumerate(usable.iterrows()):
            theoretical = crr_american_price(
                spot=float(row["underlying_price"]),
                strike=float(row["strike"]),
                tau=float(row["tau"]),
                rate=float(row.get("risk_free_rate", 0.0)),
                dividend_yield=float(row.get("dividend_yield", 0.0)),
                sigma=float(sigma),
                flag="c" if str(row["option_type"]).lower().startswith("c") else "p",
                steps=steps,
            )
            errors[idx] = theoretical - float(row["mid_price"])
        weighted = float(np.sum(weights * errors**2) / np.sum(weights))
        if weighted < best_error:
            best_sigma, best_error = float(sigma), weighted

    _LOG.info("Sigma global calibrada: %.4f (RMSE ponderado %.4f)", best_sigma,
              np.sqrt(best_error))
    return best_sigma


def detect_model_dislocation(
    chain: pd.DataFrame,
    sigma: float,
    costs: ExecutionCosts | None = None,
    steps: int = 128,
    tradable_only: bool = True,
) -> pd.DataFrame:
    r"""Compara el precio del árbol CRR contra los lados de mercado.

    Réplica de las celdas 36-38 del notebook, con dos diferencias.

    Primero, se compara contra bid y ask, no contra ``lastPrice``. El precio del
    último trade puede tener horas de antigüedad y estar en cualquier punto del
    spread; compararlo contra un modelo produce dislocaciones aparentes que son
    sólo staleness.

    Segundo, el edge se mide contra el lado que efectivamente se cruzaría:

    .. math::

        \text{edge}_{\text{compra}} = V^{\text{CRR}} - \texttt{ask},
        \qquad
        \text{edge}_{\text{venta}} = \texttt{bid} - V^{\text{CRR}}

    **Advertencia de interpretación.** Este detector, a diferencia de los cinco
    anteriores, depende de que :math:`\sigma` sea correcta. Una dislocación acá
    puede ser una oportunidad o puede ser evidencia de que el modelo de un solo
    :math:`\sigma` no captura el smile. En la presentación conviene mostrarla
    junto al smile: si las dislocaciones se concentran en las alas, es el
    modelo, no el mercado.

    Args:
        chain: Cadena curada de un instante.
        sigma: Volatilidad calibrada, típicamente de
            :func:`calibrate_chain_sigma`.
        costs: Costos de ejecución.
        steps: Pasos del árbol.
        tradable_only: Restringir a filas explotables.

    Returns:
        DataFrame de oportunidades.
    """
    costs = costs or ExecutionCosts()
    data = _prepare(chain, tradable_only)
    if not np.isfinite(sigma) or sigma <= 0:
        return _empty_result()

    rows: list[dict[str, object]] = []
    for _, row in data.iterrows():
        tau = float(row.get("tau", np.nan))
        spot = float(row.get("underlying_price", np.nan))
        if not np.isfinite([tau, spot]).all() or tau <= 0:
            continue
        flag = "c" if str(row["option_type"]).lower().startswith("c") else "p"
        theoretical = crr_american_price(
            spot=spot,
            strike=float(row["strike"]),
            tau=tau,
            rate=float(row.get("risk_free_rate", 0.0)),
            dividend_yield=float(row.get("dividend_yield", 0.0)),
            sigma=float(sigma),
            flag=flag,
            steps=steps,
        )
        buy_edge = theoretical - float(row["ask"])
        sell_edge = float(row["bid"]) - theoretical
        if buy_edge > 0:
            rows.append(
                _record(
                    "model_dislocation", (row,), buy_edge, 1.0,
                    f"+{flag.upper()}({float(row['strike']):g})@ask "
                    f"(modelo {theoretical:.2f})",
                    costs, flag, row["expiration"],
                    (_opt_leg(row, +1.0, float(row["ask"])),),
                )
            )
        elif sell_edge > 0:
            rows.append(
                _record(
                    "model_dislocation", (row,), sell_edge, 1.0,
                    f"-{flag.upper()}({float(row['strike']):g})@bid "
                    f"(modelo {theoretical:.2f})",
                    costs, flag, row["expiration"],
                    (_opt_leg(row, -1.0, float(row["bid"])),),
                )
            )
    return _finalize(rows, costs)


# --------------------------------------------------------------------------- #
# Orquestador
# --------------------------------------------------------------------------- #


def run_all_detectors(
    chain: pd.DataFrame,
    costs: ExecutionCosts | None = None,
    tradable_only: bool = True,
    include_model: bool = False,
    sigma: float | None = None,
) -> pd.DataFrame:
    """Corre todos los detectores model-free y opcionalmente el model-based.

    Args:
        chain: Cadena curada correspondiente a un único instante.
        costs: Costos de ejecución.
        tradable_only: Restringir a filas explotables.
        include_model: Incluir la dislocación contra modelo.
        sigma: Volatilidad a usar. Si es ``None`` e ``include_model`` es
            ``True``, se calibra con :func:`calibrate_chain_sigma`.

    Returns:
        DataFrame único con todas las oportunidades, ordenado por edge neto.
    """
    costs = costs or ExecutionCosts()
    frames = [
        detect_vertical_monotonicity(chain, costs, tradable_only),
        detect_vertical_spread_bound(chain, costs, tradable_only),
        detect_butterfly_convexity(chain, costs, tradable_only),
        detect_put_call_parity(chain, costs, tradable_only),
        detect_calendar(chain, costs, tradable_only),
    ]
    if include_model:
        effective_sigma = (
            sigma if sigma is not None else calibrate_chain_sigma(chain)
        )
        frames.append(
            detect_model_dislocation(chain, effective_sigma, costs, tradable_only=tradable_only)
        )

    frames = [f for f in frames if not f.empty]
    if not frames:
        return _empty_result()
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("net_edge_usd", ascending=False).reset_index(drop=True)


def execution_edge(
    signal: Mapping[str, object],
    fills: Sequence[float],
    costs: ExecutionCosts,
    contracts: float = 1.0,
) -> float:
    r"""Edge en USD de una señal re-cotizada con los precios de ejecución.

    Es la fórmula única que usan el backtest y la ejecución en vivo. El crédito
    de cada detector tiene una parte de caja (lo que se cobra al armar las
    patas) y, en algunos, una parte estructural que no depende de los precios:
    el ancho del spread vertical, el strike descontado de la paridad o el valor
    del modelo. Esa parte se obtiene de la propia señal,

    .. math::

        	ext{estructural} = 	ext{crédito bruto} - 	ext{caja}_{	ext{señal}},

    y el edge al ejecutar es
    :math:`	ext{caja}_{	ext{ejecución}} + 	ext{estructural} - 	ext{comisión}`,
    con la misma comisión que usó el detector. Si los precios no se movieron,
    coincide exactamente con ``net_edge_usd``.

    Args:
        signal: Fila de oportunidad con ``leg_spec``, ``gross_credit`` y
            ``commission``.
        fills: Precio de ejecución de cada pata, en el orden de ``leg_spec``.
        costs: Costos, para el multiplicador.
        contracts: Tamaño de la posición.

    Returns:
        Edge neto en USD para ``contracts`` unidades.

    Raises:
        ValueError: si la cantidad de precios no coincide con la de patas.
    """
    legs = tuple(signal["leg_spec"])  # type: ignore[arg-type]
    if len(fills) != len(legs):
        raise ValueError(
            f"Se recibieron {len(fills)} precios para {len(legs)} patas."
        )
    multiplier = costs.multiplier
    signal_cash = -sum(float(leg["qty"]) * float(leg["price"]) for leg in legs) * multiplier
    cash = -sum(
        float(leg["qty"]) * float(fill) for leg, fill in zip(legs, fills, strict=True)
    ) * multiplier
    structural = float(signal["gross_credit"]) * multiplier - signal_cash  # type: ignore[arg-type]
    return (cash + structural - float(signal["commission"])) * contracts  # type: ignore[arg-type]
