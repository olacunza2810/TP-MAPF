"""Capa de acceso asincrónico a Alpaca.

Nota de arquitectura
--------------------
Los clientes REST de ``alpaca-py`` son **sincrónicos**: no exponen corrutinas.
Sólo el streaming por websocket (``OptionDataStream``) es nativo async. Por lo
tanto la asincronía real se logra despachando cada llamada bloqueante a un
thread pool con ``asyncio.to_thread``, y coordinando la concurrencia con un
semáforo más un token bucket que respeta el rate limit del plan.

Envolver un cliente sincrónico en ``async def`` sin ``to_thread`` daría una
corrutina que igual bloquea el event loop: sería asincronía cosmética. Acá no se
hace eso.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import date, datetime
from typing import Any, TypeVar

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import (
    OptionBarsRequest,
    OptionSnapshotRequest,
    StockBarsRequest,
    StockLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest

from .config import AlpacaCredentials, IngestionSettings

_LOG = logging.getLogger(__name__)
_T = TypeVar("_T")

__all__ = ["TokenBucket", "AsyncAlpacaGateway"]


class TokenBucket:
    """Limitador de tasa por token bucket, seguro para uso concurrente.

    Se dimensiona en requests por minuto. A diferencia de un ``sleep`` fijo entre
    llamadas, permite ráfagas cortas sin exceder el promedio sostenido, que es
    exactamente el comportamiento que tolera el gateway de Alpaca.
    """

    def __init__(self, rate_per_minute: int, capacity: int | None = None) -> None:
        """Inicializa el bucket.

        Args:
            rate_per_minute: Tokens repuestos por minuto.
            capacity: Tamaño máximo del bucket. Por defecto, un décimo del rate.
        """
        self._rate_per_second: float = rate_per_minute / 60.0
        self._capacity: float = float(capacity or max(1, rate_per_minute // 10))
        self._tokens: float = self._capacity
        self._updated: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Bloquea hasta disponer de ``tokens`` permisos."""
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._capacity,
                    self._tokens + (now - self._updated) * self._rate_per_second,
                )
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait = deficit / self._rate_per_second
            await asyncio.sleep(wait)


class AsyncAlpacaGateway:
    """Fachada asincrónica sobre los tres clientes de Alpaca que necesitamos.

    Agrupa el cliente de trading (metadata de contratos y open interest), el de
    datos de acciones (OHLCV del subyacente) y el de datos de opciones
    (snapshots y bars de contratos).
    """

    def __init__(
        self,
        credentials: AlpacaCredentials,
        settings: IngestionSettings,
    ) -> None:
        """Instancia los clientes y las primitivas de control de concurrencia."""
        self._settings = settings
        self._trading = TradingClient(
            api_key=credentials.api_key,
            secret_key=credentials.secret_key,
            paper=credentials.paper,
        )
        self._stock = StockHistoricalDataClient(
            api_key=credentials.api_key, secret_key=credentials.secret_key
        )
        self._option = OptionHistoricalDataClient(
            api_key=credentials.api_key, secret_key=credentials.secret_key
        )
        self._bucket = TokenBucket(settings.requests_per_minute)
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)

    # ------------------------------------------------------------------ #
    # Infraestructura
    # ------------------------------------------------------------------ #

    async def _call(self, fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
        """Ejecuta una llamada bloqueante con rate limit, semáforo y reintentos.

        Args:
            fn: Método sincrónico del SDK.
            *args: Posicionales de ``fn``.
            **kwargs: Nombrados de ``fn``.

        Returns:
            El resultado de ``fn``.

        Raises:
            Exception: Re-lanza la última excepción si se agotan los reintentos.
        """
        last_error: BaseException | None = None
        for attempt in range(self._settings.max_retries):
            await self._bucket.acquire()
            async with self._semaphore:
                try:
                    return await asyncio.to_thread(fn, *args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - se re-lanza abajo
                    last_error = exc
                    backoff = (2.0**attempt) + random.uniform(0.0, 0.5)
                    _LOG.warning(
                        "Fallo %s (intento %d/%d): %s. Reintento en %.1fs",
                        getattr(fn, "__name__", repr(fn)),
                        attempt + 1,
                        self._settings.max_retries,
                        exc,
                        backoff,
                    )
            await asyncio.sleep(backoff)
        assert last_error is not None
        raise last_error

    @staticmethod
    async def _gather_limited(
        coros: Iterable[Awaitable[_T]], limit: int
    ) -> list[_T]:
        """``asyncio.gather`` con cota de concurrencia."""
        sem = asyncio.Semaphore(limit)

        async def _run(c: Awaitable[_T]) -> _T:
            async with sem:
                return await c

        return list(await asyncio.gather(*(_run(c) for c in coros)))

    # ------------------------------------------------------------------ #
    # Subyacente
    # ------------------------------------------------------------------ #

    async def fetch_underlying_bars(
        self,
        symbols: Sequence[str],
        start: datetime | date,
        end: datetime | date,
        timeframe: TimeFrame | None = None,
        feed: Any | None = None,
    ) -> dict[str, Any]:
        """Descarga OHLCV del subyacente.

        ``feed`` permite pedir IEX explícitamente: con claves gratuitas, SIP no
        entrega los últimos 15 minutos, que es justo lo que necesita el loop en
        vivo.

        Importante: se pide ``adjustment='raw'``. Los precios ajustados por
        splits y dividendos se recalculan **retroactivamente**, de modo que un
        backtest alimentado con series ajustadas incorpora información futura
        (lookahead). Además, los strikes de los contratos de opciones están
        expresados en términos no ajustados, así que alinear una opción contra
        un spot ajustado produce un moneyness incorrecto.

        Args:
            symbols: Tickers del subyacente.
            start: Inicio de la ventana, inclusive.
            end: Fin de la ventana, inclusive.
            timeframe: Granularidad. Por defecto, barras de 1 minuto.

        Returns:
            El objeto ``BarSet`` devuelto por el SDK, indexable por símbolo.
        """
        request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=timeframe or TimeFrame(1, TimeFrameUnit.Minute),
            start=start,
            end=end,
            adjustment="raw",
            feed=feed,
        )
        return await self._call(self._stock.get_stock_bars, request)

    async def fetch_stock_latest_mids(self, symbols: Sequence[str]) -> dict[str, float]:
        """Mid del último quote de cada acción, para el spot en vivo.

        Args:
            symbols: Tickers.

        Returns:
            Diccionario ticker -> mid. Omite los tickers sin bid o ask.
        """
        request = StockLatestQuoteRequest(symbol_or_symbols=list(symbols))
        quotes = await self._call(self._stock.get_stock_latest_quote, request)
        mids: dict[str, float] = {}
        for symbol, quote in quotes.items():
            bid, ask = float(quote.bid_price or 0), float(quote.ask_price or 0)
            if bid > 0 and ask > 0:
                mids[str(symbol)] = (bid + ask) / 2.0
        return mids

    # ------------------------------------------------------------------ #
    # Universo de contratos
    # ------------------------------------------------------------------ #

    async def fetch_option_contracts(
        self,
        underlyings: Sequence[str],
        expiration_gte: date,
        expiration_lte: date,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
        page_limit: int = 10_000,
    ) -> list[Any]:
        """Pagina el endpoint de contratos y devuelve la lista completa.

        Este endpoint es la única fuente de ``open_interest`` en Alpaca, junto
        con su ``open_interest_date``: el OI proviene del cálculo end-of-day de
        OCC y por lo tanto está rezagado un día hábil respecto de la sesión en
        curso. El pipeline nunca debe usar el OI del mismo día para filtrar
        operaciones de ese día.

        Args:
            underlyings: Tickers subyacentes.
            expiration_gte: Vencimiento mínimo.
            expiration_lte: Vencimiento máximo.
            strike_gte: Strike mínimo, opcional.
            strike_lte: Strike máximo, opcional.
            page_limit: Contratos por página.

        Returns:
            Lista de objetos ``OptionContract``.
        """
        collected: list[Any] = []
        page_token: str | None = None
        while True:
            request = GetOptionContractsRequest(
                underlying_symbols=list(underlyings),
                expiration_date_gte=expiration_gte,
                expiration_date_lte=expiration_lte,
                strike_price_gte=str(strike_gte) if strike_gte is not None else None,
                strike_price_lte=str(strike_lte) if strike_lte is not None else None,
                limit=page_limit,
                page_token=page_token,
            )
            response = await self._call(self._trading.get_option_contracts, request)
            batch = getattr(response, "option_contracts", None) or []
            collected.extend(batch)
            page_token = getattr(response, "next_page_token", None)
            if not page_token or not batch:
                break
        _LOG.info("Universo: %d contratos para %s", len(collected), underlyings)
        return collected

    # ------------------------------------------------------------------ #
    # Datos de opciones
    # ------------------------------------------------------------------ #

    async def fetch_option_snapshots(self, symbols: Sequence[str]) -> dict[str, Any]:
        """Descarga snapshots (quote + trade + IV + griegas) en lotes de 100.

        Este es el **único** camino en Alpaca para obtener bid/ask reales de
        opciones. Como el endpoint no acepta parámetro de fecha, la serie
        histórica de quotes debe construirse grabando estos snapshots de forma
        incremental. Ver ``ingest.SnapshotRecorder``.

        Args:
            symbols: Símbolos OCC de los contratos.

        Returns:
            Diccionario símbolo -> ``OptionsSnapshot``.
        """
        batches = [
            list(symbols[i : i + self._settings.snapshot_batch_size])
            for i in range(0, len(symbols), self._settings.snapshot_batch_size)
        ]

        async def _one(batch: list[str]) -> dict[str, Any]:
            request = OptionSnapshotRequest(
                symbol_or_symbols=batch, feed=self._settings.feed
            )
            return await self._call(self._option.get_option_snapshot, request)

        results = await self._gather_limited(
            (_one(b) for b in batches), self._settings.max_concurrency
        )
        merged: dict[str, Any] = {}
        for chunk in results:
            merged.update(chunk)
        return merged

    async def fetch_option_bars(
        self,
        symbols: Sequence[str],
        start: datetime | date,
        end: datetime | date,
        timeframe: TimeFrame | None = None,
    ) -> dict[str, Any]:
        """Descarga OHLCV histórico por contrato de opción.

        Advertencia metodológica: una barra de opción refleja **precios
        negociados**, no el NBBO. No contiene bid ni ask. Usar el ``close`` de
        la barra como si fuera el mid subestima sistemáticamente el costo de
        ejecución, porque el trade pudo haber ocurrido en cualquier punto del
        spread. Cuando el pipeline corre en modo ``backfill``, el spread se
        estima y la columna ``spread_is_proxy`` queda en ``True``.

        Args:
            symbols: Símbolos OCC.
            start: Inicio de la ventana.
            end: Fin de la ventana.
            timeframe: Granularidad. Por defecto, 1 día.

        Returns:
            El ``BarSet`` devuelto por el SDK.
        """
        request = OptionBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=timeframe or TimeFrame(1, TimeFrameUnit.Day),
            start=start,
            end=end,
        )
        return await self._call(self._option.get_option_bars, request)
