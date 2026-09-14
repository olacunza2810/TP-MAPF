"""Datos en vivo para el piloto: universo, snapshots, cadena curada y detección.

:func:`build_live_chain` es puro y reusa el enriquecimiento y los filtros del
pipeline en modo ``nbbo``; :class:`AlpacaMarketData` lo alimenta desde la API.
La IV no se calcula: los detectores model-free no la necesitan y el árbol CRR
sobre toda la cadena en cada ciclo sería el cuello de botella.

**Spot alineado al quote.** Con el feed ``indicative`` los quotes de opciones
llegan demorados. Compararlos contra el último quote de la acción, que está en
tiempo real, fabrica violaciones de paridad cada vez que la acción se mueve: en
el dry-run del 14/09/2026 aparecieron dos reversas con 3,4 a 3,7 USD de edge por
acción que duraron un solo ciclo. Por eso el spot de cada contrato es el cierre
de la barra de 1 minuto disponible en el timestamp de **su** quote
(:func:`enrich.align_underlying`), el mismo control anti-lookahead del resto del
pipeline. Los contratos sin barra dentro de la tolerancia quedan sin spot y
fuera de la cadena operable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import timedelta
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ..arbitrage import (
    ExecutionCosts,
    detect_butterfly_convexity,
    detect_put_call_parity,
    detect_vertical_monotonicity,
    detect_vertical_spread_bound,
)
from ..config import ExecutionSettings, LiquidityThresholds, PipelineConfig, PricingAssumptions
from ..enrich import (
    align_underlying,
    attach_lagged_open_interest,
    compute_mid_and_spread,
    compute_time_to_expiry,
)
from ..filters import FilterReport, apply_liquidity_filters

__all__ = ["MarketData", "build_live_chain", "detect_live", "AlpacaMarketData"]

_MASTER_COLUMNS = ["symbol", "underlying", "expiration", "strike", "option_type",
                   "style", "multiplier"]


class MarketData(Protocol):
    """Fuente de datos del loop."""

    def spots(self) -> dict[str, float]: ...

    def universe(self, spots: Mapping[str, float]) -> pd.DataFrame: ...

    def quotes(self, symbols: Sequence[str]) -> pd.DataFrame: ...

    def bars(self, now: pd.Timestamp) -> pd.DataFrame: ...


#: Tolerancia para aparear un quote con la barra de 1 minuto vigente.
SPOT_ALIGNMENT_TOLERANCE = timedelta(minutes=5)


def build_live_chain(
    quotes: pd.DataFrame,
    master: pd.DataFrame,
    spots: Mapping[str, float],
    pricing: PricingAssumptions,
    liquidity: LiquidityThresholds,
    now: pd.Timestamp,
    max_quote_age_s: float,
    bars: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, FilterReport | None]:
    """Arma la cadena del instante, lista para los detectores.

    Con ``bars`` (barras de 1 minuto con ``timestamp``, ``symbol`` y ``close``)
    el spot de cada contrato se alinea al timestamp de su quote; sin barras se
    usa ``spots``, el último quote de la acción, y ``spot_source`` lo indica.
    Además de los filtros de liquidez, excluye los quotes con más de
    ``max_quote_age_s`` de antigüedad (``excl_quote_viejo``).
    """
    if quotes.empty or master.empty:
        return pd.DataFrame(), None
    frame = quotes.merge(master[_MASTER_COLUMNS], on="symbol", how="inner")
    if frame.empty:
        return frame, None
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["trade_date"] = now.tz_convert("UTC").tz_localize(None).normalize()
    frame = compute_mid_and_spread(frame)
    if bars is not None and not bars.empty:
        frame = align_underlying(frame, bars, bar_duration=timedelta(minutes=1),
                                 tolerance=SPOT_ALIGNMENT_TOLERANCE)
        frame["spot_source"] = "barra_alineada"
    else:
        frame["underlying_price"] = frame["underlying"].map(dict(spots)).astype(float)
        frame["spot_source"] = "ultimo_quote"
    frame = compute_time_to_expiry(frame, pricing.day_count)
    frame["risk_free_rate"] = pricing.risk_free_rate
    frame["dividend_yield"] = (
        frame["underlying"].map(dict(pricing.dividend_yield)).fillna(0.0).astype(float)
    )
    if "volume" not in frame.columns:
        frame["volume"] = np.nan
    frame = attach_lagged_open_interest(
        frame, master[["symbol", "open_interest", "open_interest_date"]]
    )
    frame, report = apply_liquidity_filters(frame, liquidity)
    frame["quote_age_s"] = (now - frame["timestamp"]).dt.total_seconds()
    frame["excl_quote_viejo"] = frame["quote_age_s"] > max_quote_age_s
    frame["is_tradable"] = frame["is_tradable"] & ~frame["excl_quote_viejo"]
    return frame, report


def detect_live(
    chain: pd.DataFrame, costs: ExecutionCosts, detectors: Sequence[str]
) -> pd.DataFrame:
    """Corre los detectores habilitados en vivo, ordenados por edge neto.

    Las butterflies se evalúan sólo sobre strikes consecutivos: sobre todos los
    tríos tardan del orden de dos minutos por cadena real.
    """
    if chain.empty or not chain["is_tradable"].any():
        return pd.DataFrame()
    runners = {
        "monotonicity": lambda: detect_vertical_monotonicity(chain, costs),
        "vertical_bound": lambda: detect_vertical_spread_bound(chain, costs),
        "butterfly": lambda: detect_butterfly_convexity(chain, costs, consecutive_only=True),
        "put_call_parity": lambda: detect_put_call_parity(chain, costs),
    }
    frames = [runners[name]() for name in detectors if name in runners]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("net_edge_usd", ascending=False)
        .reset_index(drop=True)
    )


class AlpacaMarketData:
    """Implementación de :class:`MarketData` sobre el gateway del pipeline.

    Crea un gateway por llamada: sus primitivas de concurrencia quedan ligadas
    al event loop de ``asyncio.run`` y no se pueden reusar entre ciclos.
    """

    def __init__(self, config: PipelineConfig, settings: ExecutionSettings) -> None:
        self._config = replace(
            config, ingestion=replace(config.ingestion, strike_window=settings.strike_window)
        )
        self._settings = settings

    def _gateway(self) -> Any:
        from ..clients import AsyncAlpacaGateway

        return AsyncAlpacaGateway(self._config.credentials, self._config.ingestion)

    def spots(self) -> dict[str, float]:
        async def run() -> dict[str, float]:
            return await self._gateway().fetch_stock_latest_mids(self._config.underlyings)

        return asyncio.run(run())

    def universe(self, spots: Mapping[str, float]) -> pd.DataFrame:
        from ..ingest import UniverseBuilder

        async def run() -> pd.DataFrame:
            return await UniverseBuilder(self._gateway(), self._config).build(
                dict(spots), horizon_days=self._settings.max_dte
            )

        master = asyncio.run(run())
        return master.loc[master["tradable"]] if "tradable" in master.columns else master

    def quotes(self, symbols: Sequence[str]) -> pd.DataFrame:
        from ..ingest import SnapshotRecorder

        async def run() -> pd.DataFrame:
            return await SnapshotRecorder(self._gateway(), self._config).capture(list(symbols))

        return asyncio.run(run())

    def bars(self, now: pd.Timestamp) -> pd.DataFrame:
        """Barras IEX de 1 minuto de los subyacentes en los últimos 90 minutos."""
        from alpaca.data.enums import DataFeed

        async def run() -> pd.DataFrame:
            result = await self._gateway().fetch_underlying_bars(
                symbols=self._config.underlyings,
                start=(now - pd.Timedelta(minutes=90)).to_pydatetime(),
                end=now.to_pydatetime(),
                feed=DataFeed.IEX,
            )
            frame = getattr(result, "df", pd.DataFrame())
            if frame.empty:
                return pd.DataFrame(columns=["timestamp", "symbol", "close"])
            return frame.reset_index()[["timestamp", "symbol", "close"]]

        return asyncio.run(run())
