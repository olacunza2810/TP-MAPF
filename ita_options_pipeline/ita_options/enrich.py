"""Alineación temporal y cálculo de campos derivados.

El corazón anti-lookahead del pipeline está en :func:`align_underlying`. Todo lo
demás son transformaciones locales.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from .config import DailyBarAssumptions, PricingAssumptions
from .volatility import bsm_vega, implied_volatility_vectorized

_LOG = logging.getLogger(__name__)

__all__ = [
    "quotes_from_daily_bars",
    "compute_mid_and_spread",
    "align_underlying",
    "compute_time_to_expiry",
    "attach_lagged_open_interest",
    "compute_implied_volatility",
]


def quotes_from_daily_bars(
    frame: pd.DataFrame, assumptions: DailyBarAssumptions
) -> pd.DataFrame:
    r"""Construye bid y ask sintéticos a partir de una vela diaria.

    Con :math:`p` el precio de referencia (``close`` o ``vwap``) y :math:`s` el
    spread relativo supuesto:

    .. math::

        h = \max\!\left(\frac{p\,s}{2},\; h_{\min}\right), \qquad
        b = \max(p - h,\, 0), \qquad a = p + h

    El VWAP resume toda la rueda y es menos sensible a un último trade aislado;
    el cierre es el precio vigente en el instante en que se sella la fila. Si
    se pide VWAP y falta, se usa el cierre.

    Los lados resultantes **no son observaciones**: quedan marcados con
    ``spread_is_proxy=True`` y todo edge calculado sobre ellos depende de
    ``assumed_relative_spread``.

    Args:
        frame: Velas con ``close`` y opcionalmente ``vwap``.
        assumptions: Fuente de precio y spread supuesto.

    Returns:
        Copia con ``bid``, ``ask``, ``last_trade_price`` y ``spread_is_proxy``.

    Raises:
        KeyError: si falta ``close``.
        ValueError: si la fuente de precio es desconocida.
    """
    if assumptions.price_source not in ("close", "vwap"):
        raise ValueError(f"Fuente de precio desconocida: {assumptions.price_source}")
    if "close" not in frame.columns:
        raise KeyError("Las velas diarias necesitan la columna 'close'.")

    out = frame.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    price = close
    if assumptions.price_source == "vwap" and "vwap" in out.columns:
        price = pd.to_numeric(out["vwap"], errors="coerce").fillna(close)

    half = np.maximum(
        price * assumptions.assumed_relative_spread / 2.0, assumptions.min_half_spread
    )
    out["bid"] = (price - half).clip(lower=0.0)
    out["ask"] = price + half
    out["last_trade_price"] = close
    out["spread_is_proxy"] = True
    return out


def compute_mid_and_spread(frame: pd.DataFrame) -> pd.DataFrame:
    r"""Calcula mid, spread absoluto y spread relativo.

    .. math::

        m = \frac{b + a}{2}, \qquad
        s_{\text{abs}} = a - b, \qquad
        s_{\text{rel}} = \frac{a - b}{m}

    El mid aritmético es el estándar de mercado pero está sesgado cuando el
    spread es ancho respecto del precio: para un contrato con bid 0.05 y ask
    0.20, el mid 0.125 no es ejecutable en ninguna dirección. El filtro de
    ``spread_rel`` existe precisamente para excluir esos casos antes de que el
    modelo los interprete como señal.

    Los quotes cruzados (:math:`a < b`) se marcan con ``spread_abs`` negativo y
    deben descartarse aguas abajo: son estados transitorios del libro o errores
    del feed, nunca oportunidades.

    Args:
        frame: DataFrame con columnas ``bid`` y ``ask``.

    Returns:
        Copia del frame con ``mid_price``, ``spread_abs`` y ``spread_rel``.
    """
    out = frame.copy()
    bid = pd.to_numeric(out["bid"], errors="coerce")
    ask = pd.to_numeric(out["ask"], errors="coerce")
    out["mid_price"] = (bid + ask) / 2.0
    out["spread_abs"] = ask - bid
    out["spread_rel"] = np.where(
        out["mid_price"] > 0.0, out["spread_abs"] / out["mid_price"], np.nan
    )
    crossed = int((out["spread_abs"] < 0).sum())
    if crossed:
        _LOG.warning("%d quotes cruzados detectados (ask < bid).", crossed)
    return out


def align_underlying(
    options: pd.DataFrame,
    underlying_bars: pd.DataFrame,
    bar_duration: timedelta = timedelta(minutes=1),
    tolerance: timedelta = timedelta(minutes=5),
) -> pd.DataFrame:
    """Asigna a cada contrato el precio del subyacente vigente en su timestamp.

    Dos decisiones sostienen la ausencia de lookahead:

    1. **Dirección ``backward``.** Cada quote de opción se aparea únicamente con
       una barra del subyacente cuyo timestamp sea anterior o igual. Un
       ``merge_asof`` con dirección ``nearest`` puede tomar la barra siguiente y
       filtrar el futuro dentro del dataset.

    2. **Corrimiento por duración de la barra.** Alpaca etiqueta las barras con
       el instante de *apertura* del intervalo. Una barra rotulada 14:30 cubre
       hasta las 14:31, así que su ``close`` recién se conoce a las 14:31. El
       merge se hace contra ``available_at = timestamp + bar_duration``. Sin
       este corrimiento se introduce hasta una barra completa de lookahead, que
       en granularidad de 1 minuto alcanza para inventar arbitrajes que no
       existen.

    Adicionalmente se impone ``tolerance``: si la barra vigente más reciente es
    demasiado vieja, el spot queda nulo. Esto elimina los quotes stale de fuera
    de la rueda, que de otro modo se aparearían con el cierre de la sesión
    anterior.

    Args:
        options: Quotes de opciones con columnas ``timestamp`` y ``underlying``.
        underlying_bars: Barras con ``timestamp``, ``symbol`` y ``close``.
        bar_duration: Duración del intervalo de las barras del subyacente.
        tolerance: Antigüedad máxima admitida del spot.

    Returns:
        DataFrame de opciones con ``underlying_price``, ``underlying_bar_ts``,
        ``underlying_available_at`` y ``alignment_lag_s``.

    Raises:
        KeyError: si faltan columnas requeridas en alguno de los frames.
    """
    for column in ("timestamp", "underlying"):
        if column not in options:
            raise KeyError(f"'{column}' ausente en el frame de opciones.")
    for column in ("timestamp", "symbol", "close"):
        if column not in underlying_bars:
            raise KeyError(f"'{column}' ausente en el frame de barras.")

    bars = underlying_bars[["timestamp", "symbol", "close"]].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars["available_at"] = bars["timestamp"] + bar_duration
    bars = bars.rename(
        columns={
            "symbol": "underlying",
            "close": "underlying_price",
            "timestamp": "underlying_bar_ts",
        }
    ).sort_values("available_at")
    # El parquet devuelve ``underlying`` como StringDtype y el SDK entrega
    # ``symbol`` como str: pandas 3 rechaza cruzar claves de tipos distintos.
    # Lo mismo con la resolución temporal: parquet puede volver en ns o en us.
    bars["underlying"] = bars["underlying"].astype("string")
    bars["available_at"] = bars["available_at"].dt.as_unit("ns")

    # Un frame leído del data lake ya trae estas columnas vacías, porque
    # ``enforce_schema`` completa el esquema: si no se quitan, el merge las
    # duplica.
    derived = ("underlying_price", "underlying_bar_ts", "underlying_available_at",
               "alignment_lag_s")
    left = options.drop(columns=[c for c in derived if c in options.columns])
    left["underlying"] = left["underlying"].astype("string")
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True).dt.as_unit("ns")
    left = left.sort_values("timestamp")

    merged = pd.merge_asof(
        left,
        bars,
        left_on="timestamp",
        right_on="available_at",
        by="underlying",
        direction="backward",
        tolerance=tolerance,
        allow_exact_matches=True,
    )
    merged = merged.rename(columns={"available_at": "underlying_available_at"})
    merged["alignment_lag_s"] = (
        merged["timestamp"] - merged["underlying_available_at"]
    ).dt.total_seconds()

    unmatched = int(merged["underlying_price"].isna().sum())
    if unmatched:
        _LOG.warning(
            "%d de %d quotes sin spot dentro de la tolerancia (%s).",
            unmatched,
            len(merged),
            tolerance,
        )
    return merged


def compute_time_to_expiry(
    frame: pd.DataFrame, day_count: int = 365
) -> pd.DataFrame:
    r"""Calcula días al vencimiento y :math:`\tau` en años.

    El vencimiento efectivo de una opción sobre acciones estadounidenses es el
    cierre del tercer viernes a las 16:00 ET, no la medianoche del día de
    expiración. Se ancla a las 20:00 UTC, que corresponde a las 16:00 ET en
    horario de verano. Para rigor total en la frontera habría que resolver el
    huso con ``zoneinfo``; el error residual es de una hora sobre un horizonte
    de semanas, pero se documenta porque afecta a los contratos de 0-1 DTE.

    .. math::

        \tau = \frac{T - t}{\text{day\_count}}

    Args:
        frame: DataFrame con ``timestamp`` y ``expiration``.
        day_count: Base anual. 365 calendario por convención de mercado en IV.

    Returns:
        Copia con ``dte`` (entero, días calendario) y ``tau`` (años).
    """
    out = frame.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    expiry = pd.to_datetime(out["expiration"]).dt.tz_localize(None)
    expiry_dt = (expiry + pd.Timedelta(hours=20)).dt.tz_localize("UTC")

    delta_seconds = (expiry_dt - ts).dt.total_seconds()
    out["tau"] = delta_seconds / (day_count * 24 * 3600)
    out["dte"] = np.ceil(delta_seconds / 86400.0).clip(lower=0).astype("int32")
    return out


def attach_lagged_open_interest(
    quotes: pd.DataFrame, contract_master: pd.DataFrame
) -> pd.DataFrame:
    """Adjunta el open interest respetando su rezago de publicación.

    El OI que Alpaca devuelve proviene del cálculo end-of-day de OCC y viene
    acompañado de su ``open_interest_date``. Filtrar la rueda del día ``t`` con
    el OI del día ``t`` es lookahead: ese número recién se publica después del
    cierre. El merge exige ``open_interest_date < trade_date``.

    Args:
        quotes: Quotes con ``symbol`` y ``trade_date``.
        contract_master: Maestro con ``symbol``, ``open_interest`` y
            ``open_interest_date``.

    Returns:
        Quotes con ``open_interest`` y ``open_interest_date`` rezagados.
    """
    master = contract_master[
        ["symbol", "open_interest", "open_interest_date"]
    ].copy()
    master["open_interest_date"] = pd.to_datetime(
        master["open_interest_date"]
    ).dt.tz_localize(None)

    out = quotes.drop(
        columns=[c for c in ("open_interest", "open_interest_date") if c in quotes],
        errors="ignore",
    ).merge(master, on="symbol", how="left")

    trade_date = pd.to_datetime(out["trade_date"]).dt.tz_localize(None)
    stale = out["open_interest_date"] >= trade_date
    out.loc[stale, "open_interest"] = np.nan
    if int(stale.sum()):
        _LOG.info(
            "%d filas con OI no publicado al momento del quote: anulado.",
            int(stale.sum()),
        )
    return out


def compute_implied_volatility(
    frame: pd.DataFrame,
    assumptions: PricingAssumptions,
    american: bool = True,
) -> pd.DataFrame:
    """Invierte la IV contrato por contrato y calcula el vega de diagnóstico.

    Se calculan dos IVs: la americana por árbol CRR, que es la correcta para
    RTX y BA, y la europea por BSM. La diferencia entre ambas cuantifica la
    prima de ejercicio anticipado y sirve de control: si es grande en calls OTM
    sin dividendos inminentes, hay un problema en los inputs, no en el mercado.

    Args:
        frame: DataFrame enriquecido con ``mid_price``, ``underlying_price``,
            ``strike``, ``tau``, ``option_type`` y ``underlying``.
        assumptions: Supuestos de tasa y dividend yield.
        american: Si ``True`` calcula también la IV con ejercicio anticipado.

    Returns:
        Copia con ``iv_american``, ``iv_european``, ``vega``, ``risk_free_rate``,
        ``dividend_yield`` y ``violates_bounds``.
    """
    out = frame.copy()
    out["risk_free_rate"] = assumptions.risk_free_rate
    out["dividend_yield"] = (
        out["underlying"].map(dict(assumptions.dividend_yield)).fillna(0.0)
    )

    valid = (
        out["mid_price"].gt(0)
        & out["underlying_price"].gt(0)
        & out["tau"].gt(0)
        & out["strike"].gt(0)
    )
    subset = out.loc[valid]
    _LOG.info("Invirtiendo IV sobre %d de %d filas.", len(subset), len(out))

    for column, is_american in (("iv_european", False), ("iv_american", american)):
        if column == "iv_american" and not american:
            out["iv_american"] = np.nan
            continue
        out.loc[valid, column] = implied_volatility_vectorized(
            prices=subset["mid_price"].to_numpy(dtype=np.float64),
            spots=subset["underlying_price"].to_numpy(dtype=np.float64),
            strikes=subset["strike"].to_numpy(dtype=np.float64),
            taus=subset["tau"].to_numpy(dtype=np.float64),
            rates=subset["risk_free_rate"].to_numpy(dtype=np.float64),
            dividend_yields=subset["dividend_yield"].to_numpy(dtype=np.float64),
            flags=subset["option_type"].astype(str).to_numpy(),
            american=is_american,
            steps=assumptions.binomial_steps,
        )

    reference_iv = out["iv_american"].fillna(out["iv_european"])
    out["vega"] = [
        bsm_vega(s, k, t, r, q, sig)
        if all(np.isfinite([s, k, t, r, q, sig])) and t > 0 and sig > 0
        else np.nan
        for s, k, t, r, q, sig in zip(
            out["underlying_price"],
            out["strike"],
            out["tau"],
            out["risk_free_rate"],
            out["dividend_yield"],
            reference_iv,
            strict=False,
        )
    ]
    out["violates_bounds"] = valid & reference_iv.isna()

    out["moneyness"] = out["underlying_price"] / out["strike"]
    out["log_moneyness"] = np.log(out["moneyness"].where(out["moneyness"] > 0))
    return out
