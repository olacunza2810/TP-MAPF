"""Ingesta: construcción de universo, grabación point-in-time y backfill.

Dos modos, porque Alpaca sólo permite dos cosas distintas:

``record``
    Graba snapshots del chain a intervalos fijos. Es la única fuente de bid/ask
    reales. Construye la serie histórica hacia adelante desde el día que se
    prende. Es lo que hay que dejar corriendo ya para tener dataset propio en la
    Parte 2.

``backfill``
    Descarga bars OHLCV por contrato hacia atrás. Da historia inmediata pero sin
    NBBO: el spread se estima y queda marcado con ``spread_is_proxy=True``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

import pandas as pd

from .clients import AsyncAlpacaGateway
from .config import PipelineConfig

_LOG = logging.getLogger(__name__)

__all__ = ["UniverseBuilder", "SnapshotRecorder", "HistoricalBackfill"]


def _as_float(value: Any) -> float:
    """Convierte a float devolviendo ``nan`` ante valores nulos o inválidos."""
    try:
        return float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


class UniverseBuilder:
    """Construye y persiste el maestro point-in-time de contratos.

    Sobre survivorship bias
    -----------------------
    El endpoint de contratos de Alpaca devuelve el universo **vigente**. Los
    contratos ya vencidos van desapareciendo, de modo que reconstruir hoy el
    universo de hace seis meses produce un dataset que sólo contiene lo que
    sobrevivió: survivorship bias en su forma más pura.

    La única mitigación real es no reconstruir: grabar el universo cada día en
    un maestro append-only con la fecha de observación. A partir de ahí, el
    backtest consulta "qué contratos existían el día ``t``" contra lo que
    efectivamente se observó ese día, no contra lo que existe hoy.

    Mientras el maestro propio no tenga profundidad suficiente, los backtests
    sobre períodos anteriores al inicio de la grabación arrastran el sesgo y
    deben reportarse como tales.
    """

    def __init__(self, gateway: AsyncAlpacaGateway, config: PipelineConfig) -> None:
        """Inicializa el builder."""
        self._gateway = gateway
        self._config = config

    async def build(
        self,
        spot_prices: dict[str, float] | None = None,
        horizon_days: int | None = None,
    ) -> pd.DataFrame:
        """Descarga el universo de contratos y lo normaliza a DataFrame.

        Args:
            spot_prices: Spot por ticker, usado para acotar strikes a la ventana
                configurada. Si es ``None`` se descarga el universo completo.
            horizon_days: Vencimiento máximo en días. Por defecto usa
                ``liquidity.max_dte``.

        Returns:
            DataFrame con una fila por contrato y ``observed_at`` de la corrida.
        """
        horizon = horizon_days or self._config.liquidity.max_dte
        today = date.today()
        frames: list[pd.DataFrame] = []

        for ticker in self._config.underlyings:
            strike_lo = strike_hi = None
            if spot_prices and ticker in spot_prices:
                spot = spot_prices[ticker]
                window = self._config.ingestion.strike_window
                strike_lo, strike_hi = spot * (1 - window), spot * (1 + window)

            contracts = await self._gateway.fetch_option_contracts(
                underlyings=[ticker],
                expiration_gte=today,
                expiration_lte=today + timedelta(days=horizon),
                strike_gte=strike_lo,
                strike_lte=strike_hi,
            )
            frames.append(self._to_frame(contracts))

        master = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        master["observed_at"] = pd.Timestamp.now(tz="UTC")
        return master

    @staticmethod
    def _to_frame(contracts: Iterable[Any]) -> pd.DataFrame:
        """Aplana objetos ``OptionContract`` del SDK a un DataFrame tipado."""
        rows = [
            {
                "symbol": c.symbol,
                "underlying": c.underlying_symbol,
                "expiration": pd.to_datetime(c.expiration_date),
                "strike": _as_float(c.strike_price),
                "option_type": str(getattr(c.type, "value", c.type)).lower()[:1],
                "style": str(getattr(c.style, "value", c.style)).lower(),
                # Alpaca expone el multiplicador como ``size``: la cantidad de
                # acciones entregables por contrato, normalmente 100. No existe
                # un campo ``multiplier`` en OptionContract.
                "multiplier": _as_float(getattr(c, "size", None)) or 100.0,
                "open_interest": _as_float(getattr(c, "open_interest", None)),
                "open_interest_date": pd.to_datetime(
                    getattr(c, "open_interest_date", None)
                ),
                "close_price": _as_float(getattr(c, "close_price", None)),
                "tradable": bool(getattr(c, "tradable", True)),
                "status": str(getattr(c.status, "value", getattr(c, "status", ""))),
            }
            for c in contracts
        ]
        return pd.DataFrame(rows)


class SnapshotRecorder:
    """Graba snapshots del chain construyendo la serie point-in-time.

    Cada pasada produce una foto del NBBO de todos los contratos del universo,
    sellada con el timestamp del quote (``timestamp``) y con el instante en que
    el proceso la observó (``observed_at``). Distinguir ambos importa: el
    primero es la marca temporal del mercado y el segundo la del sistema; su
    diferencia mide cuán stale está el quote y permite descartar contratos que
    no se actualizan.
    """

    def __init__(self, gateway: AsyncAlpacaGateway, config: PipelineConfig) -> None:
        """Inicializa el recorder."""
        self._gateway = gateway
        self._config = config

    async def capture(self, symbols: Sequence[str]) -> pd.DataFrame:
        """Toma una foto del universo.

        Args:
            symbols: Símbolos OCC a capturar.

        Returns:
            DataFrame con una fila por contrato con quote disponible.
        """
        snapshots = await self._gateway.fetch_option_snapshots(symbols)
        observed_at = pd.Timestamp.now(tz="UTC")
        rows: list[dict[str, Any]] = []

        for symbol, snap in snapshots.items():
            quote = getattr(snap, "latest_quote", None)
            if quote is None:
                continue
            trade = getattr(snap, "latest_trade", None)
            greeks = getattr(snap, "greeks", None)
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": pd.Timestamp(quote.timestamp).tz_convert("UTC"),
                    "observed_at": observed_at,
                    "bid": _as_float(quote.bid_price),
                    "ask": _as_float(quote.ask_price),
                    "bid_size": _as_float(quote.bid_size),
                    "ask_size": _as_float(quote.ask_size),
                    "last_trade_price": _as_float(getattr(trade, "price", None)),
                    "last_trade_at": (
                        pd.Timestamp(trade.timestamp).tz_convert("UTC")
                        if trade is not None
                        else pd.NaT
                    ),
                    # El snapshot NO trae volumen diario acumulado: sólo el
                    # tamaño del último trade. Guardarlo como "volume" haría que
                    # el filtro de liquidez compare contra la magnitud
                    # equivocada. El volumen diario, si se necesita, sale de
                    # get_option_bars.
                    "last_trade_size": _as_float(getattr(trade, "size", None)),
                    "volume": float("nan"),
                    "iv_vendor": _as_float(
                        getattr(snap, "implied_volatility", None)
                    ),
                    "vendor_delta": _as_float(getattr(greeks, "delta", None)),
                    "spread_is_proxy": False,
                    "feed": self._config.ingestion.feed,
                    "source": "snapshot",
                }
            )

        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["trade_date"] = frame["timestamp"].dt.tz_convert("UTC").dt.normalize()
        _LOG.info("Snapshot: %d contratos con quote de %d pedidos.", len(frame), len(symbols))
        return frame

    async def run_forever(
        self,
        symbols: Sequence[str],
        sink: "Any",
        interval_seconds: int = 300,
        max_iterations: int | None = None,
    ) -> None:
        """Loop de grabación continua.

        Args:
            symbols: Universo a grabar.
            sink: Objeto con método ``write(frame)``, típicamente un
                :class:`storage.ParquetStore`.
            interval_seconds: Período entre capturas. 300 s da ~78 fotos por
                rueda, suficiente para calibrar la superficie sin saturar el
                rate limit.
            max_iterations: Corta el loop tras N pasadas. ``None`` es infinito.
        """
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            started = datetime.now(timezone.utc)
            try:
                frame = await self.capture(symbols)
                if not frame.empty:
                    await asyncio.to_thread(sink.write, frame)
            except Exception:  # noqa: BLE001 - el recorder no debe morir nunca
                _LOG.exception("Fallo en la pasada %d; se continúa.", iteration)
            iteration += 1
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            await asyncio.sleep(max(0.0, interval_seconds - elapsed))


class HistoricalBackfill:
    """Descarga bars históricos por contrato como sustituto de quotes.

    Limitación estructural: una barra de opción no contiene NBBO. Se construye
    un mid proxy con el ``close`` y se estima el spread a partir de la
    volatilidad intrabarra, marcando ``spread_is_proxy=True``. Ese flag debe
    propagarse al backtest: los resultados sobre filas proxy no son comparables
    con los obtenidos sobre NBBO real y deben reportarse por separado.
    """

    def __init__(self, gateway: AsyncAlpacaGateway, config: PipelineConfig) -> None:
        """Inicializa el backfill."""
        self._gateway = gateway
        self._config = config

    async def fetch(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        chunk_size: int = 50,
    ) -> pd.DataFrame:
        """Descarga bars diarios para el universo dado.

        Args:
            symbols: Símbolos OCC.
            start: Inicio de la ventana.
            end: Fin de la ventana.
            chunk_size: Contratos por request.

        Returns:
            DataFrame en el esquema de quotes, con spread proxy.
        """
        chunks = [
            list(symbols[i : i + chunk_size])
            for i in range(0, len(symbols), chunk_size)
        ]
        results = await asyncio.gather(
            *(self._gateway.fetch_option_bars(c, start, end) for c in chunks),
            return_exceptions=True,
        )

        frames: list[pd.DataFrame] = []
        for result in results:
            if isinstance(result, BaseException):
                _LOG.error("Chunk fallido en backfill: %s", result)
                continue
            frame = getattr(result, "df", None)
            if frame is not None and not frame.empty:
                frames.append(frame.reset_index())

        if not frames:
            return pd.DataFrame()

        bars = pd.concat(frames, ignore_index=True)
        return self._bars_to_quotes(bars)

    def _bars_to_quotes(self, bars: pd.DataFrame) -> pd.DataFrame:
        r"""Convierte bars a filas de quote con spread estimado.

        El proxy usado es

        .. math::

            \hat{s} = \max\!\left(0.01,\ \min\!\left(
                \frac{h - l}{2},\ 0.10\, c \right)\right)

        es decir, media amplitud del rango intrabarra, acotada por un tick
        mínimo y por 10% del precio. Es una estimación deliberadamente
        conservadora y **no** un dato: sirve para acotar el costo de ejecución
        por arriba en el backtest, no para calibrar la superficie.

        Args:
            bars: Barras crudas del SDK.

        Returns:
            DataFrame en el esquema de quotes.
        """
        out = bars.rename(columns={"symbol": "symbol", "timestamp": "timestamp"}).copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        close = pd.to_numeric(out["close"], errors="coerce")
        half_spread = (
            ((pd.to_numeric(out["high"]) - pd.to_numeric(out["low"])) / 2.0)
            .clip(lower=0.01)
            .combine(close * 0.10, min)
        )
        out["bid"] = (close - half_spread).clip(lower=0.0)
        out["ask"] = close + half_spread
        out["last_trade_price"] = close
        out["volume"] = pd.to_numeric(out.get("volume"), errors="coerce")
        out["spread_is_proxy"] = True
        out["observed_at"] = pd.Timestamp.now(tz="UTC")
        out["trade_date"] = out["timestamp"].dt.normalize()
        out["feed"] = self._config.ingestion.feed
        out["source"] = "bars_proxy"
        return out
