r"""Descarga de datos históricos de opciones desde Polygon (Massive).

Alpaca sólo entrega el NBBO del instante, así que no permite reconstruir el
pasado. Polygon sí tiene cotizaciones históricas de opciones. Este script arma,
para RTX, BA y LMT, el **mismo data lake** que construye el pipeline con Alpaca,
pero hacia atrás en el tiempo, para que ``enrich``, los detectores y el backtest
lo consuman sin cambios de formato::

    data/
    ├── option_quotes/underlying=RTX/trade_date=2025-03-03/part-*.parquet
    ├── contract_master/universe_20250303.parquet
    ├── underlying_bars/RTX_1min.parquet, RTX_1day.parquet
    ├── polygon/option_daily_bars.parquet
    ├── polygon/dividends.parquet
    ├── polygon/treasury_yields.parquet
    ├── risk_free_curve.csv                   (formato date,tenor_days,rate)
    └── polygon_manifest_<stamp>.json

Qué se pide y a qué endpoint
----------------------------
1. Tasas del Tesoro ....... ``GET /fed/v1/treasury-yields``
2. Dividendos ............. ``GET /stocks/v1/dividends``
3. Barras de la acción .... ``GET /v2/aggs/ticker/{T}/range/1/{minute|day}/{from}/{to}``
4. Contratos por día ...... ``GET /v3/reference/options/contracts`` con ``as_of``
5. Barras diarias opción .. ``GET /v2/aggs/ticker/O:.../range/1/day/{from}/{to}``
6. NBBO de cada opción .... ``GET /v3/quotes/O:...``

Controles de sesgo
------------------
**Lookahead.** La ventana de strikes del día ``t`` se fija con la *apertura* de
``t``. La columna ``volume`` del día ``t`` es el volumen de la rueda ``t-1``,
porque el de ``t`` recién se conoce al cierre. La cotización en cada marca
horaria es el último quote con ``sip_timestamp <= marca``. La acción se pide sin
ajustar (``adjusted=false``), igual que en Alpaca.

**Survivorship.** El universo de cada día se pide con ``as_of=día``, uniendo
``expired=false`` y ``expired=true``: incluye los contratos que existían ese día
aunque hoy estén vencidos.

Limitación
----------
Polygon no publica **open interest histórico**: sólo el del snapshot actual. Las
columnas ``open_interest`` quedan en ``NaN``. Con los umbrales por defecto
(``min_open_interest=50``) el filtro de liquidez excluiría todo, así que para
correr ``enrich`` sobre estos datos hay que usar el volumen del día anterior como
criterio y bajar ``min_open_interest`` a 0 (o adjuntar OI de otra fuente).

Uso::

    $env:POLYGON_API_KEY="..."
    python scripts/download_polygon.py --start 2025-01-02 --end 2025-12-31 --dry-run
    python scripts/download_polygon.py --start 2025-01-02 --end 2025-12-31

El script es reanudable: si se corta, vuelve a correrlo con los mismos
parámetros y saltea lo que ya está en disco.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ita_options.storage import ParquetStore  # noqa: E402

_LOG = logging.getLogger("download_polygon")
NY = ZoneInfo("America/New_York")

#: Vencimiento en días de cada serie del Tesoro, para la curva ``tenor_days``.
_TREASURY_TENORS: dict[str, int] = {
    "yield_1_month": 30,
    "yield_3_month": 91,
    "yield_6_month": 182,
    "yield_1_year": 365,
    "yield_2_year": 730,
    "yield_3_year": 1095,
    "yield_5_year": 1826,
    "yield_7_year": 2557,
    "yield_10_year": 3652,
    "yield_20_year": 7305,
    "yield_30_year": 10957,
}

ALL_STEPS = ("rates", "dividends", "stock", "contracts", "option_bars", "quotes")


# --------------------------------------------------------------------------- #
# Cliente HTTP
# --------------------------------------------------------------------------- #


class PolygonAccessError(RuntimeError):
    """La clave no tiene permiso para el endpoint (401/403): problema de plan."""


class PolygonNotFound(RuntimeError):
    """El endpoint no existe (404)."""


class RateLimiter:
    """Espaciado uniforme de requests, seguro entre threads."""

    def __init__(self, per_second: float) -> None:
        self._interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next = time.monotonic()

    def wait(self) -> None:
        """Bloquea hasta que toque el próximo request."""
        with self._lock:
            now = time.monotonic()
            self._next = max(self._next, now)
            delay = self._next - now
            self._next += self._interval
        if delay > 0:
            time.sleep(delay)


class PolygonClient:
    """Cliente mínimo de la REST API con reintentos y paginación por ``next_url``.

    La clave viaja como parámetro ``apiKey`` y nunca se escribe en los logs.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        requests_per_second: float,
        max_retries: int = 6,
        timeout: float = 60.0,
    ) -> None:
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._limiter = RateLimiter(requests_per_second)
        self._max_retries = max_retries
        self._timeout = timeout
        self._local = threading.local()
        self.request_count = 0

    def _session(self) -> requests.Session:
        """Una sesión por thread: ``requests.Session`` no es thread-safe."""
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session

    def get(self, path_or_url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET con reintentos ante 429 y 5xx, con backoff exponencial y jitter."""
        url = path_or_url if path_or_url.startswith("http") else self._base + path_or_url
        query = {**(params or {}), "apiKey": self._key}
        for attempt in range(1, self._max_retries + 1):
            self._limiter.wait()
            self.request_count += 1
            try:
                response = self._session().get(url, params=query, timeout=self._timeout)
            except requests.RequestException as exc:
                self._backoff(attempt, url, type(exc).__name__)
                continue
            status = response.status_code
            if status == 200:
                return response.json()
            if status in (401, 403):
                raise PolygonAccessError(
                    f"HTTP {status} en {url.split('?')[0]}: {response.text[:300]}. "
                    "La clave no tiene acceso a este dato: revisar el plan contratado."
                )
            if status == 404:
                raise PolygonNotFound(url.split("?")[0])
            if status == 429 or status >= 500:
                self._backoff(attempt, url, f"HTTP {status}")
                continue
            raise RuntimeError(f"HTTP {status} en {url.split('?')[0]}: {response.text[:300]}")
        raise RuntimeError(f"Se agotaron los reintentos para {url.split('?')[0]}")

    @staticmethod
    def _backoff(attempt: int, url: str, reason: str) -> None:
        delay = min(60.0, 2.0**attempt) + random.uniform(0.0, 1.0)
        _LOG.warning("%s en %s (intento %d). Reintento en %.1fs",
                     reason, url.split("?")[0], attempt, delay)
        time.sleep(delay)

    def pages(self, path: str, params: dict[str, Any]) -> Iterator[list[dict[str, Any]]]:
        """Itera las páginas de resultados siguiendo ``next_url``."""
        payload = self.get(path, params)
        while True:
            yield payload.get("results") or []
            next_url = payload.get("next_url")
            if not next_url:
                return
            payload = self.get(next_url)

    def paginate(self, path: str, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Itera los resultados individuales de todas las páginas."""
        for page in self.pages(path, params):
            yield from page


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def _aggs_frame(rows: list[dict[str, Any]], symbol: str) -> pd.DataFrame:
    """Convierte barras de ``/v2/aggs`` al formato ``timestamp, symbol, open, ...``.

    Polygon, igual que Alpaca, sella la barra con el **inicio** del intervalo:
    ``enrich.align_underlying`` ya corrige eso con ``bar_duration``.
    """
    columns = ["timestamp", "symbol", "open", "high", "low", "close",
               "volume", "vwap", "transactions"]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows).rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close",
                 "v": "volume", "vw": "vwap", "n": "transactions"}
    )
    frame["timestamp"] = pd.to_datetime(frame["t"], unit="ms", utc=True)
    frame["symbol"] = symbol
    return frame.reindex(columns=columns)


def _month_chunks(start: date, end: date) -> Iterator[tuple[date, date]]:
    """Ventanas mensuales: una barra por minuto no entra en un solo request."""
    cursor = start
    while cursor <= end:
        next_month = (cursor.replace(day=1) + timedelta(days=32)).replace(day=1)
        yield cursor, min(end, next_month - timedelta(days=1))
        cursor = next_month


def session_marks(day: date, interval_minutes: int) -> pd.DatetimeIndex:
    """Marcas horarias de la rueda regular (9:30-16:00 ET), en UTC.

    La primera marca es 9:30 + intervalo: a las 9:30 en punto casi no hay quotes.
    El huso de Nueva York resuelve el horario de verano automáticamente.
    """
    open_ny = datetime.combine(day, dtime(9, 30), NY)
    close_ny = datetime.combine(day, dtime(16, 0), NY)
    marks = pd.date_range(
        open_ny + timedelta(minutes=interval_minutes), close_ny,
        freq=f"{interval_minutes}min",
    )
    return marks.tz_convert("UTC")


def _ns(stamp: pd.Timestamp) -> int:
    return int(pd.Timestamp(stamp).as_unit("ns").value)


# --------------------------------------------------------------------------- #
# 1-3. Tasas, dividendos y acción
# --------------------------------------------------------------------------- #


def download_rates(client: PolygonClient, start: date, end: date, out: Path) -> None:
    r"""Rendimientos del Tesoro y curva continua en el formato de ``config.py``.

    Los rendimientos vienen en porcentaje, con capitalización semestral
    equivalente. Se convierten a tasa continua con :math:`r = 2\ln(1 + y/200)`.
    El rendimiento del día ``t`` se publica al cierre de ``t``: quien lo consuma
    debe usar la curva de ``t-1`` para valuar durante la rueda ``t``.
    """
    rows = list(client.paginate("/fed/v1/treasury-yields", {
        "date.gte": (start - timedelta(days=10)).isoformat(),
        "date.lte": end.isoformat(),
        "limit": 50000,
        "sort": "date.asc",
    }))
    if not rows:
        _LOG.warning("Tesoro: sin datos en el rango.")
        return
    raw = pd.DataFrame(rows)
    _write_parquet(raw, out / "polygon" / "treasury_yields.parquet")

    series = [c for c in _TREASURY_TENORS if c in raw.columns]
    curve = raw.melt(id_vars="date", value_vars=series,
                     var_name="serie", value_name="yield_pct").dropna()
    curve["tenor_days"] = curve["serie"].map(_TREASURY_TENORS)
    curve["rate"] = 2.0 * np.log1p(curve["yield_pct"].astype(float) / 200.0)
    curve = curve[["date", "tenor_days", "rate"]].sort_values(["date", "tenor_days"])
    curve.to_csv(out / "risk_free_curve.csv", index=False)
    _LOG.info("Tesoro: %d días, curva en risk_free_curve.csv", raw["date"].nunique())


def download_dividends(
    client: PolygonClient, tickers: list[str], start: date, end: date, out: Path
) -> None:
    """Dividendos con fechas ex, de pago y monto.

    Se pide desde un año antes del inicio para poder calcular el yield trailing
    del primer día. BA puede no traer filas: tiene el dividendo suspendido.
    """
    frames = []
    since = (start - timedelta(days=400)).isoformat()
    for ticker in tickers:
        try:
            rows = list(client.paginate("/stocks/v1/dividends", {
                "ticker": ticker, "ex_dividend_date.gte": since,
                "ex_dividend_date.lte": end.isoformat(), "limit": 5000,
                "sort": "ex_dividend_date.asc",
            }))
        except PolygonNotFound:
            # Endpoint anterior al rebranding, con los mismos campos principales.
            rows = list(client.paginate("/v3/reference/dividends", {
                "ticker": ticker, "ex_dividend_date.gte": since,
                "ex_dividend_date.lte": end.isoformat(), "limit": 1000,
                "sort": "ex_dividend_date", "order": "asc",
            }))
        frames.append(pd.DataFrame(rows).assign(ticker=ticker))
        _LOG.info("Dividendos %s: %d eventos", ticker, len(rows))
    _write_parquet(pd.concat(frames, ignore_index=True), out / "polygon" / "dividends.parquet")


def download_stock_bars(
    client: PolygonClient, ticker: str, start: date, end: date, out: Path
) -> pd.DataFrame:
    """Barras diarias y de 1 minuto de la acción, **sin ajustar**.

    Returns:
        Las barras diarias con ``trade_date`` (fecha en Nueva York), que definen
        el calendario de ruedas y la apertura usada para la ventana de strikes.
    """
    daily_path = out / "underlying_bars" / f"{ticker}_1day.parquet"
    minute_path = out / "underlying_bars" / f"{ticker}_1min.parquet"
    params = {"adjusted": "false", "sort": "asc", "limit": 50000}

    if daily_path.exists():
        daily = pd.read_parquet(daily_path)
    else:
        rows = list(client.paginate(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}", params))
        daily = _aggs_frame(rows, ticker)
        daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True)
        daily["trade_date"] = daily["timestamp"].dt.tz_convert(NY).dt.date
        _write_parquet(daily, daily_path)

    if not minute_path.exists():
        frames = []
        for chunk_start, chunk_end in _month_chunks(start, end):
            rows = list(client.paginate(
                f"/v2/aggs/ticker/{ticker}/range/1/minute/{chunk_start}/{chunk_end}",
                params,
            ))
            frames.append(_aggs_frame(rows, ticker))
        minute = pd.concat(frames, ignore_index=True)
        _write_parquet(minute, minute_path)
        _LOG.info("Acción %s: %d barras de 1 minuto", ticker, len(minute))

    daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.date
    return daily


# --------------------------------------------------------------------------- #
# 4. Universo de contratos por día
# --------------------------------------------------------------------------- #


def contracts_for_day(
    client: PolygonClient, ticker: str, day: date, day_open: float,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Contratos que existían el día ``day``, dentro de la ventana configurada.

    Se unen ``expired=false`` y ``expired=true`` porque la semántica de
    ``expired`` combinada con ``as_of`` no está documentada con precisión;
    pedir ambos es barato y garantiza no perder contratos ya vencidos.
    """
    window = args.strike_window
    params = {
        "underlying_ticker": ticker,
        "as_of": day.isoformat(),
        "expiration_date.gte": (day + timedelta(days=args.min_dte)).isoformat(),
        "expiration_date.lte": (day + timedelta(days=args.max_dte)).isoformat(),
        "strike_price.gte": round(day_open * (1 - window), 2),
        "strike_price.lte": round(day_open * (1 + window), 2),
        "limit": 1000,
    }
    found: dict[str, dict[str, Any]] = {}
    for expired in ("false", "true"):
        for item in client.paginate("/v3/reference/options/contracts",
                                    {**params, "expired": expired}):
            found[item["ticker"]] = item

    if not found:
        return pd.DataFrame()
    raw = pd.DataFrame(list(found.values()))
    master = pd.DataFrame({
        "symbol": raw["ticker"],
        "underlying": raw["underlying_ticker"],
        "expiration": pd.to_datetime(raw["expiration_date"]),
        "strike": raw["strike_price"].astype(float),
        "option_type": raw["contract_type"].astype(str).str.lower().str[0],
        "style": raw.get("exercise_style", pd.Series("american", index=raw.index)),
        "multiplier": raw.get("shares_per_contract",
                              pd.Series(100, index=raw.index)).astype(float),
        # Polygon no tiene OI histórico: ver la limitación en el docstring.
        "open_interest": np.nan,
        "open_interest_date": pd.NaT,
        "close_price": np.nan,
        "tradable": True,
        "status": "active",
    })
    master = master.loc[master["option_type"].isin(["c", "p"])]
    if args.max_expirations:
        keep = sorted(master["expiration"].unique())[: args.max_expirations]
        master = master.loc[master["expiration"].isin(keep)]
    master["observed_at"] = pd.Timestamp(day).tz_localize("UTC")
    master["trade_date"] = pd.Timestamp(day)
    return master.reset_index(drop=True)


def build_universes(
    client: PolygonClient, days: list[date], opens: dict[tuple[str, date], float],
    args: argparse.Namespace, out: Path,
) -> dict[date, pd.DataFrame]:
    """Arma (o relee de disco) el maestro de cada rueda para todos los tickers."""
    universes: dict[date, pd.DataFrame] = {}
    for day in days:
        path = out / "contract_master" / f"universe_{day:%Y%m%d}.parquet"
        if path.exists():
            universes[day] = pd.read_parquet(path)
            continue
        frames = []
        for ticker in args.tickers:
            day_open = opens.get((ticker, day))
            if day_open is None or not np.isfinite(day_open):
                continue
            frames.append(contracts_for_day(client, ticker, day, day_open, args))
        frames = [f for f in frames if not f.empty]
        master = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not master.empty:
            _write_parquet(master, path)
        universes[day] = master
        _LOG.info("Universo %s: %d contratos", day, len(master))
    return universes


# --------------------------------------------------------------------------- #
# 5. Barras diarias de cada opción (volumen)
# --------------------------------------------------------------------------- #


def download_option_daily_bars(
    client: PolygonClient, universes: dict[date, pd.DataFrame],
    args: argparse.Namespace, out: Path,
) -> pd.DataFrame:
    """Una request por contrato con toda su vida dentro del rango.

    Sirve para el volumen diario, que el NBBO no trae. Se arranca una semana
    antes del primer día observado para tener el volumen de ``t-1``.
    """
    path = out / "polygon" / "option_daily_bars.parquet"
    if path.exists():
        return pd.read_parquet(path)

    spans: dict[str, list[date]] = {}
    for day, master in universes.items():
        for symbol, expiration in zip(master.get("symbol", []),
                                      master.get("expiration", []), strict=False):
            first, last = spans.get(symbol, [day, day])
            last_day = min(max(last, day), pd.Timestamp(expiration).date(), args.end)
            spans[symbol] = [min(first, day), last_day]

    def fetch(symbol: str, first: date, last: date) -> pd.DataFrame:
        rows = list(client.paginate(
            f"/v2/aggs/ticker/{symbol}/range/1/day/{first - timedelta(days=7)}/{last}",
            {"adjusted": "false", "sort": "asc", "limit": 50000},
        ))
        return _aggs_frame(rows, symbol)

    frames = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch, s, a, b) for s, (a, b) in spans.items()]
        for index, future in enumerate(as_completed(futures), start=1):
            frames.append(future.result())
            if index % 500 == 0:
                _LOG.info("Barras diarias de opciones: %d/%d", index, len(futures))

    bars = pd.concat(frames, ignore_index=True) if frames else _aggs_frame([], "")
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars["trade_date"] = bars["timestamp"].dt.tz_convert(NY).dt.date
    _write_parquet(bars, path)
    _LOG.info("Barras diarias de opciones: %d filas de %d contratos", len(bars), len(spans))
    return bars


# --------------------------------------------------------------------------- #
# 6. NBBO histórico remuestreado
# --------------------------------------------------------------------------- #


def fetch_contract_quotes(
    client: PolygonClient, symbol: str, marks: pd.DatetimeIndex, max_age: timedelta,
) -> pd.DataFrame:
    """Remuestrea el NBBO tick a tick de un contrato a las marcas de la rueda.

    En cada marca ``m`` toma el último quote con ``sip_timestamp <= m``, que es
    exactamente lo que habría grabado el recorder de Alpaca a esa hora. Los
    quotes se procesan página por página, así que un contrato con millones de
    actualizaciones no se carga entero en memoria. Si el último quote tiene más
    de ``max_age`` de antigüedad, la marca se descarta por stale.
    """
    mark_ns = np.array([_ns(m) for m in marks], dtype=np.int64)
    session_open = _ns(marks[0].tz_convert(NY).replace(hour=9, minute=30))
    latest: dict[int, dict[str, Any]] = {}

    for page in client.pages(f"/v3/quotes/{symbol}", {
        "timestamp.gte": str(session_open),
        "timestamp.lte": str(int(mark_ns[-1])),
        "sort": "timestamp", "order": "asc", "limit": 50000,
    }):
        if not page:
            continue
        frame = pd.DataFrame(page)
        if "sip_timestamp" not in frame:
            continue
        stamps = frame["sip_timestamp"].to_numpy(dtype=np.int64)
        # Primera marca >= quote: el quote vale para esa marca y las siguientes.
        frame["bucket"] = np.searchsorted(mark_ns, stamps, side="left")
        frame = frame.loc[frame["bucket"] < len(mark_ns)]
        for row in frame.drop_duplicates("bucket", keep="last").to_dict("records"):
            latest[int(row["bucket"])] = row

    rows = []
    current: dict[str, Any] | None = None
    max_age_ns = int(max_age.total_seconds() * 1e9)
    for index, mark in enumerate(mark_ns):
        current = latest.get(index, current)
        if current is None or mark - int(current["sip_timestamp"]) > max_age_ns:
            continue
        rows.append({
            "timestamp": int(mark),
            "bid": current.get("bid_price", np.nan),
            "ask": current.get("ask_price", np.nan),
            "bid_size": current.get("bid_size", np.nan),
            "ask_size": current.get("ask_size", np.nan),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["timestamp"] = pd.to_datetime(out["timestamp"], unit="ns", utc=True)
    out["symbol"] = symbol
    return out


def process_quotes_day(
    client: PolygonClient, ticker: str, day: date, prev_day: date | None,
    master: pd.DataFrame, volume: dict[tuple[str, date], float],
    args: argparse.Namespace, store: ParquetStore,
) -> int:
    """Descarga el NBBO de todos los contratos de un ticker en una rueda y lo persiste.

    Escribe una sola vez al final del día: si el proceso se corta a mitad, la
    partición no existe y la próxima corrida la rehace entera.
    """
    partition = store.path / f"underlying={ticker}" / f"trade_date={day.isoformat()}"
    if partition.exists() and any(partition.glob("*.parquet")):
        return 0

    contracts = master.loc[master["underlying"] == ticker]
    if contracts.empty:
        return 0
    marks = session_marks(day, args.interval)
    max_age = timedelta(minutes=args.max_quote_age)

    frames = []
    pool = ThreadPoolExecutor(args.workers)
    try:
        futures = {pool.submit(fetch_contract_quotes, client, s, marks, max_age): s
                   for s in contracts["symbol"]}
        for future in as_completed(futures):
            try:
                frame = future.result()
            except PolygonAccessError:
                raise
            except Exception as exc:  # noqa: BLE001 - un contrato no frena el día
                _LOG.error("Quotes %s %s: %s", futures[future], day, exc)
                continue
            if not frame.empty:
                frames.append(frame)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    if not frames:
        _LOG.warning("%s %s: ningún contrato con quotes.", ticker, day)
        return 0

    quotes = pd.concat(frames, ignore_index=True).merge(
        contracts[["symbol", "underlying", "expiration", "strike", "option_type",
                   "style", "multiplier", "open_interest", "open_interest_date"]],
        on="symbol", how="left",
    )
    # Volumen de la rueda anterior: el del día recién se conoce al cierre.
    if prev_day is None:
        quotes["volume"] = np.nan
    else:
        quotes["volume"] = [volume.get((s, prev_day), 0.0) for s in quotes["symbol"]]
    quotes["observed_at"] = quotes["timestamp"]
    quotes["last_trade_price"] = np.nan
    quotes["spread_is_proxy"] = False
    quotes["feed"] = "polygon"
    quotes["source"] = "polygon_quotes"
    quotes["trade_date"] = pd.Timestamp(day)
    return store.write(quotes)


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    yesterday = date.today() - timedelta(days=1)
    parser = argparse.ArgumentParser(
        description="Descarga histórica de opciones desde Polygon (Massive).")
    parser.add_argument("--tickers", nargs="+", default=["RTX", "BA", "LMT"])
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, default=yesterday)
    parser.add_argument("--out", type=Path, default=ROOT / "data",
                        help="Raíz del data lake (la misma que usa el pipeline).")
    parser.add_argument("--interval", type=int, default=5,
                        help="Minutos entre marcas de la rueda.")
    parser.add_argument("--strike-window", type=float, default=0.25,
                        help="Strikes entre (1-w) y (1+w) veces la apertura.")
    parser.add_argument("--min-dte", type=int, default=7)
    parser.add_argument("--max-dte", type=int, default=180)
    parser.add_argument("--max-expirations", type=int, default=0,
                        help="Sólo los N vencimientos más cercanos (0 = todos).")
    parser.add_argument("--max-quote-age", type=int, default=30,
                        help="Minutos máximos de antigüedad de un quote.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rps", type=float, default=40.0,
                        help="Requests por segundo (Polygon sugiere < 100).")
    parser.add_argument("--base-url", default="https://api.polygon.io",
                        help="api.polygon.io sigue activo; api.massive.com es el nuevo.")
    parser.add_argument("--steps", default=",".join(ALL_STEPS),
                        help=f"Subconjunto de: {', '.join(ALL_STEPS)}")
    parser.add_argument("--dry-run", action="store_true",
                        help="Arma sólo los universos y estima el volumen de requests.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    args.steps = {s.strip() for s in args.steps.split(",") if s.strip()}
    unknown = args.steps - set(ALL_STEPS)
    if unknown:
        parser.error(f"Pasos desconocidos: {sorted(unknown)}")
    if args.start > args.end:
        parser.error("--start no puede ser posterior a --end")
    return args


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    api_key = os.environ.get("POLYGON_API_KEY") or os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        raise SystemExit("Falta POLYGON_API_KEY (o MASSIVE_API_KEY) en el entorno.")

    client = PolygonClient(api_key, args.base_url, args.rps)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    try:
        if "rates" in args.steps and not args.dry_run:
            download_rates(client, args.start, args.end, out)
        if "dividends" in args.steps and not args.dry_run:
            download_dividends(client, args.tickers, args.start, args.end, out)

        # La acción siempre hace falta: define el calendario y las aperturas.
        daily = {t: download_stock_bars(client, t, args.start, args.end, out)
                 for t in args.tickers}
        calendar = sorted({d for frame in daily.values() for d in frame["trade_date"]
                           if args.start <= d <= args.end})
        opens = {(t, d): float(o) for t, frame in daily.items()
                 for d, o in zip(frame["trade_date"], frame["open"], strict=True)}
        _LOG.info("Calendario: %d ruedas entre %s y %s", len(calendar), args.start, args.end)

        if not ({"contracts", "option_bars", "quotes"} & args.steps):
            return
        universes = build_universes(client, calendar, opens, args, out)

        contract_days = sum(len(m) for m in universes.values())
        unique = len({s for m in universes.values() for s in m.get("symbol", [])})
        _LOG.info("Universo total: %d contrato-día, %d contratos únicos", contract_days, unique)
        if args.dry_run:
            print(
                f"\nEstimación: ~{contract_days:,} requests de quotes (una o más por "
                f"contrato-día) + ~{unique:,} de barras diarias.\n"
                f"A {args.rps:g} req/s: al menos "
                f"{(contract_days + unique) / args.rps / 3600:.1f} horas.\n"
            )
            return

        volume: dict[tuple[str, date], float] = {}
        if "option_bars" in args.steps:
            bars = download_option_daily_bars(client, universes, args, out)
            volume = {(s, d): float(v) for s, d, v in
                      zip(bars["symbol"], bars["trade_date"], bars["volume"], strict=True)}

        if "quotes" in args.steps:
            store = ParquetStore(out, "option_quotes")
            for index, day in enumerate(calendar):
                prev_day = calendar[index - 1] if index > 0 else None
                master = universes.get(day, pd.DataFrame())
                if master.empty:
                    continue
                for ticker in args.tickers:
                    tick = time.monotonic()
                    rows = process_quotes_day(client, ticker, day, prev_day, master,
                                              volume, args, store)
                    if rows:
                        _LOG.info("Quotes %s %s: %d filas en %.0fs", ticker, day, rows,
                                  time.monotonic() - tick)
    except PolygonAccessError as exc:
        raise SystemExit(f"\n{exc}\n") from exc
    finally:
        manifest = {
            "source": "polygon",
            "base_url": args.base_url,
            "tickers": args.tickers,
            "start": args.start.isoformat(),
            "end": args.end.isoformat(),
            "interval_minutes": args.interval,
            "strike_window": args.strike_window,
            "min_dte": args.min_dte,
            "max_dte": args.max_dte,
            "max_expirations": args.max_expirations,
            "max_quote_age_minutes": args.max_quote_age,
            "steps": sorted(args.steps),
            "dry_run": args.dry_run,
            "requests": client.request_count,
            "elapsed_s": round(time.monotonic() - started, 1),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (out / f"polygon_manifest_{stamp}.json").write_text(
            json.dumps(manifest, indent=2), "utf-8")


if __name__ == "__main__":
    main()
