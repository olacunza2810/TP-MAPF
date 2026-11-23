r"""Curación de la foto diaria de DoltHub: bid y ask observados.

Es la respuesta al punto 1 de la devolución: el backtest de la entrega parcial
ejecutaba contra un bid/ask sintético construido alrededor de cierres no
simultáneos. Acá cada pata se compra al ask y se vende al bid **observados**.

Tres decisiones de diseño
-------------------------

**Sellado temporal.** DoltHub no documenta la hora de la foto. Cada fila se
sella a las 16:00 de Nueva York de su rueda, y la señal de la rueda :math:`t`
se ejecuta contra la foto de :math:`t+1` (o :math:`t+2`). Nunca se usa
información de una foto posterior a la de la señal.

**Desfase entre la foto y el cierre de la acción.** Con bid/ask reales, las
violaciones de paridad que quedan se concentran en los días en que el spot
implícito por la propia cadena se separa del cierre de la acción: la foto se
toma durante la rueda. Para cada subyacente y rueda se estima

.. math::

    \hat S_t = \operatorname{mediana}_{K \approx S}\left(
        C^{mid}_K - P^{mid}_K + K e^{-r\tau} + D\right)

sobre los pares cerca del dinero, y se define la incertidumbre del spot
:math:`u_t = |\hat S_t - S_t|`. La paridad cotiza la acción en la banda
:math:`S_t \pm (\tfrac{s}{2} + u_t)`: una violación tiene que superar el desfase
para contar. Sin pares suficientes, :math:`u_t` toma un valor conservador.

**Dividendos point-in-time.** El detector necesita el valor presente de los
dividendos con fecha ex dentro de la vida de cada opción. Usar las fechas
realizadas sería lookahead para los dividendos que todavía no se habían
anunciado. En :math:`t` entran los dividendos ya **declarados** (Polygon trae
la fecha de declaración) y, más allá del último declarado, se repite su monto
cada 91 días. Los dividendos **realizados** se usan sólo para el P&L
(:class:`CarryModel`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .backtest import CarryModel
from .config import LiquidityThresholds
from .daily import session_close
from .filters import FilterReport, apply_liquidity_filters

_LOG = logging.getLogger(__name__)

#: Tolerancia de moneyness y plazo para estimar el spot implícito.
_ATM_BAND = 0.03
_ATM_MIN_DTE = 7
_ATM_MAX_DTE = 60
#: Incertidumbre del spot cuando no hay pares suficientes, como fracción del spot.
_FALLBACK_UNCERTAINTY = 0.01
#: Frecuencia de la proyección de dividendos (trimestral).
_DIVIDEND_PERIOD_DAYS = 91


@dataclass(frozen=True, slots=True)
class DoltHubData:
    """Insumos crudos de la descarga (``scripts/download_dolthub.py``)."""

    chain: pd.DataFrame
    bars: pd.DataFrame
    dividends: pd.DataFrame
    rates: pd.DataFrame

    @classmethod
    def load(cls, data_root: Path, tickers: Sequence[str]) -> "DoltHubData":
        """Lee la cadena consolidada, las velas, los dividendos y las tasas.

        Raises:
            FileNotFoundError: si falta algún archivo de la descarga.
        """
        root = Path(data_root)
        chain = pd.read_parquet(root / "dolthub" / "option_chain.parquet")
        chain = chain.loc[chain["underlying"].isin(tickers)].copy()
        bars = pd.concat(
            [pd.read_parquet(root / "underlying_bars" / f"{t}_1day.parquet") for t in tickers],
            ignore_index=True,
        )
        dividends = merge_dividends(
            pd.read_parquet(root / "dolthub" / "dividends.parquet"),
            root / "dolthub" / "dividends_polygon.parquet",
        )
        rates = pd.read_parquet(root / "dolthub" / "rates.parquet")
        return cls(chain=chain, bars=bars, dividends=dividends, rates=rates)

    def carry_model(self, funding_spread: float = 0.005, borrow_rate: float = 0.005) -> CarryModel:
        """Modelo de carry con la tasa observada y los dividendos realizados."""
        rate_by_date = dict(zip(pd.to_datetime(self.rates["date"]).dt.date, self.rates["rate"]))
        return CarryModel(
            rate=float(self.rates["rate"].median()),
            rate_by_date=rate_by_date,
            funding_spread=funding_spread,
            borrow_rate=borrow_rate,
            dividends=self.dividends,
        )


def occ_symbol(underlying: str, expiration: date, option_type: str, strike: float) -> str:
    """Símbolo OCC, el mismo formato que usa Alpaca: ``LMT260918C00600000``."""
    flag = "C" if option_type.lower().startswith("c") else "P"
    return f"{underlying}{expiration:%y%m%d}{flag}{int(round(strike * 1000)):08d}"


#: Anticipación supuesta del anuncio cuando no hay fecha de declaración.
_DECLARATION_LAG_DAYS = 30


def merge_dividends(yahoo: pd.DataFrame, polygon_path: Path) -> pd.DataFrame:
    """Une los dividendos de Yahoo con los de Polygon, que traen fecha de declaración.

    La fecha de declaración es la que permite saber qué se conocía en cada
    rueda. Para los dividendos que sólo trae Yahoo (los posteriores a la
    descarga de Polygon) se supone un anuncio :data:`_DECLARATION_LAG_DAYS`
    días antes de la fecha ex, que es lo habitual para RTX y LMT.

    Returns:
        ``ticker, ex_dividend_date, declaration_date, cash_amount, fuente``.
    """
    frames = []
    if Path(polygon_path).exists():
        poly = pd.read_parquet(polygon_path)
        if "distribution_type" in poly:
            poly = poly.loc[poly["distribution_type"] == "recurring"]
        frames.append(pd.DataFrame({
            "ticker": poly["ticker"],
            "ex_dividend_date": pd.to_datetime(poly["ex_dividend_date"]).dt.date,
            "declaration_date": pd.to_datetime(poly["declaration_date"]).dt.date,
            "cash_amount": poly["cash_amount"].astype(float),
            "fuente": "polygon",
        }))
    y = yahoo.copy()
    y["ex_dividend_date"] = pd.to_datetime(y["ex_dividend_date"]).dt.date
    frames.append(pd.DataFrame({
        "ticker": y["ticker"], "ex_dividend_date": y["ex_dividend_date"],
        "declaration_date": [d - timedelta(days=_DECLARATION_LAG_DAYS) for d in y["ex_dividend_date"]],
        "cash_amount": y["cash_amount"].astype(float), "fuente": "yahoo",
    }))
    out = pd.concat(frames, ignore_index=True)
    # Ante el mismo dividendo en las dos fuentes, queda Polygon (fecha de declaración real).
    out = out.sort_values("fuente").drop_duplicates(["ticker", "ex_dividend_date"], keep="first")
    return out.sort_values(["ticker", "ex_dividend_date"]).reset_index(drop=True)


def projected_dividends(
    dividends: pd.DataFrame, ticker: str, asof: date, horizon: date
) -> list[tuple[date, float]]:
    """Dividendos esperados con fecha ex en ``(asof, horizon]``, sabiendo sólo lo de ``asof``.

    1. Los dividendos ya **declarados** en ``asof`` (``declaration_date <= asof``)
       con fecha ex futura entran con su fecha y su monto.
    2. Más allá del último declarado se repite su monto cada
       :data:`_DIVIDEND_PERIOD_DAYS` días.

    Un ticker sin dividendos declarados (BA) no proyecta nada.
    """
    table = dividends.loc[dividends["ticker"] == ticker]
    if table.empty:
        return []
    ex = pd.to_datetime(table["ex_dividend_date"]).dt.date
    if "declaration_date" in table:
        declared = pd.to_datetime(table["declaration_date"]).dt.date
    else:
        declared = ex.map(lambda d: d - timedelta(days=_DECLARATION_LAG_DAYS))
    known = table.assign(_ex=ex, _decl=declared)
    known = known.loc[known["_decl"] <= asof]
    if known.empty:
        return []
    out = [(d, float(a)) for d, a in zip(known["_ex"], known["cash_amount"]) if asof < d <= horizon]
    last = known.sort_values("_ex").iloc[-1]
    nxt = last["_ex"] + timedelta(days=_DIVIDEND_PERIOD_DAYS)
    while nxt <= horizon:
        if nxt > asof:
            out.append((nxt, float(last["cash_amount"])))
        nxt += timedelta(days=_DIVIDEND_PERIOD_DAYS)
    return sorted(out)


def _pv_dividends(
    frame: pd.DataFrame, dividends: pd.DataFrame
) -> pd.Series:
    """Valor presente de los dividendos proyectados dentro de la vida de cada fila."""
    out = pd.Series(0.0, index=frame.index)
    keys = frame[["underlying", "trade_date", "expiration", "risk_free_rate"]].drop_duplicates()
    for row in keys.itertuples(index=False):
        flows = projected_dividends(dividends, row.underlying, row.trade_date, row.expiration)
        if not flows:
            continue
        pv = sum(a * np.exp(-row.risk_free_rate * (d - row.trade_date).days / 365.0) for d, a in flows)
        mask = (
            (frame["underlying"] == row.underlying)
            & (frame["trade_date"] == row.trade_date)
            & (frame["expiration"] == row.expiration)
        )
        out.loc[mask] = pv
    return out


def attach_projected_dividends(chain: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """Agrega ``pv_dividends`` y el yield continuo equivalente a una cadena en vivo.

    Es la misma convención que usa el backtest: la paridad recibe el valor
    presente de los dividendos discretos proyectados point-in-time, y el árbol
    un :math:`q` tal que :math:`S e^{-q\\tau} = S - D`. Sin esto, el loop en
    vivo volvía al yield continuo constante que la devolución objetó.
    """
    if chain.empty or dividends is None or dividends.empty:
        return chain
    out = chain.copy()
    work = pd.DataFrame({
        "underlying": out["underlying"],
        "trade_date": pd.to_datetime(out["trade_date"]).dt.date,
        "expiration": pd.to_datetime(out["expiration"]).dt.date,
        "risk_free_rate": out["risk_free_rate"].astype(float),
    }, index=out.index)
    out["pv_dividends"] = _pv_dividends(work, dividends)
    ratio = (1.0 - out["pv_dividends"] / out["underlying_price"]).clip(lower=0.5)
    out["dividend_yield"] = (-np.log(ratio) / out["tau"].clip(lower=1e-6)).fillna(0.0)
    return out


def implied_spot(chain: pd.DataFrame) -> pd.DataFrame:
    r"""Spot implícito por paridad en los pares cercanos al dinero.

    Para opciones americanas cortas y cerca del dinero la prima de ejercicio
    anticipado es chica, y la relación europea con dividendos da una buena
    estimación del spot al que la cadena estaba cotizando cuando se tomó la foto.

    Returns:
        Una fila por ``(underlying, trade_date)`` con ``implied_spot``,
        ``n_pairs`` y ``spot_gap_pct`` contra ``underlying_price``.
    """
    base = chain.loc[(chain["bid"] > 0) & (chain["ask"] > 0)]
    keys = ["underlying", "trade_date", "expiration", "strike"]
    calls = base.loc[base["option_type"] == "c", keys + ["mid_price", "underlying_price", "tau", "risk_free_rate", "pv_dividends", "dte"]]
    puts = base.loc[base["option_type"] == "p", keys + ["mid_price"]]
    pairs = calls.merge(puts, on=keys, suffixes=("_c", "_p"))
    pairs = pairs.loc[
        (np.abs(np.log(pairs["strike"] / pairs["underlying_price"])) <= _ATM_BAND)
        & pairs["dte"].between(_ATM_MIN_DTE, _ATM_MAX_DTE)
    ]
    pairs = pairs.assign(
        s_hat=pairs["mid_price_c"] - pairs["mid_price_p"]
        + pairs["strike"] * np.exp(-pairs["risk_free_rate"] * pairs["tau"])
        + pairs["pv_dividends"]
    )
    out = (
        pairs.groupby(["underlying", "trade_date"])
        .agg(implied_spot=("s_hat", "median"), n_pairs=("s_hat", "size"), underlying_price=("underlying_price", "first"))
        .reset_index()
    )
    out["spot_gap_pct"] = 100.0 * (out["implied_spot"] / out["underlying_price"] - 1.0)
    return out


def curate(
    data: DoltHubData,
    thresholds: LiquidityThresholds | None = None,
    min_pairs: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, FilterReport, pd.DataFrame]:
    """Arma el dataset curado para el backtest y el subyacente con bid/ask.

    Args:
        data: Insumos de la descarga.
        thresholds: Filtros de liquidez; por defecto :meth:`LiquidityThresholds.for_dolthub`.
        min_pairs: Pares mínimos cerca del dinero para estimar el spot implícito.

    Returns:
        ``(cadena, subyacente, reporte_de_filtros, desfase_de_spot)``. La cadena
        incluye todas las filas, con ``is_tradable``: para marcar posiciones hace
        falta el precio aunque el contrato haya dejado de ser líquido.
    """
    thresholds = thresholds or LiquidityThresholds.for_dolthub()
    chain = data.chain.copy()
    chain["trade_date"] = pd.to_datetime(chain["trade_date"]).dt.date
    chain["expiration"] = pd.to_datetime(chain["expiration"]).dt.date

    bars = data.bars.copy()
    bars["date"] = pd.to_datetime(bars["date"]).dt.date
    spot = bars.rename(columns={"symbol": "underlying", "date": "trade_date", "close": "underlying_price"})
    chain = chain.merge(spot[["underlying", "trade_date", "underlying_price"]], on=["underlying", "trade_date"], how="left")

    rates = data.rates.copy()
    rates["date"] = pd.to_datetime(rates["date"]).dt.date
    rates = rates.sort_values("date")
    chain = chain.sort_values("trade_date")
    chain = pd.merge_asof(
        chain.assign(_d=pd.to_datetime(chain["trade_date"])),
        rates.assign(_d=pd.to_datetime(rates["date"]))[["_d", "rate"]],
        on="_d", direction="backward",
    ).drop(columns="_d").rename(columns={"rate": "risk_free_rate"})

    chain["timestamp"] = session_close(chain["trade_date"]).to_numpy()
    expiry_close = session_close(chain["expiration"]).to_numpy()
    chain["tau"] = (pd.to_datetime(expiry_close) - pd.to_datetime(chain["timestamp"].to_numpy())).total_seconds() / (365.0 * 86400)
    chain["dte"] = [(e - t).days for e, t in zip(chain["expiration"], chain["trade_date"])]
    chain["mid_price"] = (chain["bid"] + chain["ask"]) / 2.0
    chain["spread_abs"] = chain["ask"] - chain["bid"]
    chain["spread_rel"] = chain["spread_abs"] / chain["mid_price"].where(chain["mid_price"] > 0)
    chain["symbol"] = [
        occ_symbol(u, e, f, k)
        for u, e, f, k in zip(chain["underlying"], chain["expiration"], chain["option_type"], chain["strike"])
    ]
    chain["dividend_yield"] = 0.0
    chain["pv_dividends"] = _pv_dividends(chain, data.dividends)
    chain["volume"] = np.nan
    chain["open_interest"] = np.nan
    chain["spread_is_proxy"] = False

    gaps = implied_spot(chain)
    chain = chain.merge(gaps[["underlying", "trade_date", "implied_spot", "n_pairs"]], on=["underlying", "trade_date"], how="left")
    reliable = chain["n_pairs"].fillna(0) >= min_pairs
    chain["spot_uncertainty"] = np.where(
        reliable,
        (chain["implied_spot"] - chain["underlying_price"]).abs(),
        chain["underlying_price"] * _FALLBACK_UNCERTAINTY,
    )
    chain["expiration"] = pd.to_datetime(chain["expiration"])

    curated, report = apply_liquidity_filters(chain, thresholds, drop=False)

    band = 0.005 + curated.groupby(["underlying", "trade_date"])["spot_uncertainty"].first()
    under = bars.rename(columns={"date": "trade_date"}).copy()
    under["timestamp"] = session_close(under["trade_date"]).to_numpy()
    under = under.merge(
        band.rename("band").reset_index().rename(columns={"underlying": "symbol"}),
        on=["symbol", "trade_date"], how="left",
    )
    under["band"] = under["band"].fillna(0.005 + under["close"] * _FALLBACK_UNCERTAINTY)
    under["bid"] = under["close"] - under["band"]
    under["ask"] = under["close"] + under["band"]
    _LOG.info(
        "DoltHub curado: %d filas, %d explotables, %d ruedas; desfase de spot mediano %.2f%%",
        len(curated), int(curated["is_tradable"].sum()), curated["trade_date"].nunique(),
        float(gaps["spot_gap_pct"].abs().median()) if not gaps.empty else float("nan"),
    )
    return curated, under, report, gaps
