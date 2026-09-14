r"""Descarga histórica **diaria** desde Polygon (Massive), pensada para el plan gratuito.

El plan gratuito tiene dos límites que definen todo el diseño:

1. No incluye cotizaciones (``/v3/quotes``) ni snapshots de opciones: sólo velas
   agregadas (OHLCV y VWAP).
2. Admite **5 requests por minuto**.

Por eso el script:

- baja **una sola vez** la serie de velas diarias de cada contrato para toda su
  ventana (``/v2/aggs/.../range/1/day/{desde}/{hasta}``), nunca día por día;
- acota el universo a los **vencimientos mensuales** más próximos y a los
  strikes cercanos al dinero;
- espera 12 segundos entre requests por defecto y reintenta con backoff
  exponencial ante HTTP 429 (respetando ``Retry-After`` si viene);
- guarda cada respuesta en disco apenas llega: si se corta, se vuelve a correr
  con los mismos parámetros y saltea lo que ya está descargado.

Qué se pide
-----------
============  ======================================================  ===================
Paso          Endpoint                                                Requests
============  ======================================================  ===================
stock         ``/v2/aggs/ticker/{T}/range/1/day/{desde}/{hasta}``     1 por ticker
rates         ``/fed/v1/treasury-yields``                             1
dividends     ``/stocks/v1/dividends``                                1 por ticker
contracts     ``/v3/reference/options/contracts``                     1 por ticker y vto.
option_bars   ``/v2/aggs/ticker/O:.../range/1/day/{desde}/{hasta}``   1 por contrato
build         (sin red) arma ``data/option_daily``                    0
============  ======================================================  ===================

Universo
--------
Para cada ticker y cada vencimiento mensual ``E`` (tercer viernes, o el jueves
anterior si el viernes es feriado):

- **Ventana**: desde el día siguiente al vencimiento mensual que está
  ``nearest`` lugares antes que ``E`` (desde ahí ``E`` es uno de los ``nearest``
  mensuales más próximos) hasta ``E``, recortada a ``[start, end]``.
- **Strikes**: dentro de ±``band`` del spot de referencia y, de esos, los
  ``max_strikes_per_side`` más cercanos por debajo y por encima. El spot de
  referencia es el **cierre de la última rueda anterior al inicio de la
  ventana**: la selección no usa precios que todavía no se conocían.

Limitación de esa selección: los strikes que el exchange lista después del
inicio de la ventana, cuando el spot ya se movió, no entran.

Salida
------
::

    data/
    ├── underlying_bars/RTX_1day.parquet
    ├── option_daily/underlying=RTX/trade_date=2025-03-03/part-*.parquet
    ├── polygon/contract_master.parquet
    ├── polygon/option_bars/O_RTX250321C00120000.parquet   (caché por contrato)
    ├── polygon/contracts/*.parquet                         (caché de listados)
    ├── polygon/coverage.json                               (rangos ya descargados)
    ├── polygon/dividends.parquet
    ├── polygon/treasury_yields.parquet
    ├── risk_free_curve.csv                                 (date,tenor_days,rate)
    └── polygon_manifest_<stamp>.json

Uso::

    $env:POLYGON_API_KEY="..."
    python scripts/download_polygon.py --start 2025-01-02 --end 2025-12-31 --dry-run
    python scripts/download_polygon.py --start 2025-01-02 --end 2025-12-31
    python -m ita_options.pipeline --tickers RTX BA LMT enrich-daily
    python -m ita_options.pipeline --tickers RTX BA LMT backtest-daily --lag-sessions 1 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from collections.abc import Iterator, Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ita_options.daily import (  # noqa: E402
    NY,
    RAW_DATASET,
    previous_session_volume,
    session_close,
)
from ita_options.storage import ParquetStore  # noqa: E402

_LOG = logging.getLogger("download_polygon")

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

ALL_STEPS = ("stock", "rates", "dividends", "contracts", "option_bars", "build")

#: Ruedas previas al inicio de cada ventana que se piden de más, para conocer el
#: volumen de la rueda anterior al primer día de la ventana.
_LOOKBACK_DAYS = 10


# --------------------------------------------------------------------------- #
# Cliente HTTP
# --------------------------------------------------------------------------- #


class PolygonAccessError(RuntimeError):
    """La clave no tiene permiso para el dato (401/403): problema de plan."""


class PolygonNotFound(RuntimeError):
    """El endpoint no existe (404)."""


class RateLimiter:
    """Espaciado fijo entre requests: 5 por minuto son 12 segundos."""

    def __init__(self, requests_per_minute: float) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute debe ser positivo.")
        self.interval = 60.0 / requests_per_minute
        self._next: float | None = None

    def wait(self) -> None:
        """Duerme lo necesario para respetar el intervalo desde el request anterior."""
        now = time.monotonic()
        start = now if self._next is None else max(now, self._next)
        if start > now:
            time.sleep(start - now)
        self._next = start + self.interval


class PolygonClient:
    """Cliente mínimo de la REST API: límite de tasa, reintentos y ``next_url``.

    La clave viaja como parámetro ``apiKey`` y nunca se escribe en los logs.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.polygon.io",
        requests_per_minute: float = 5.0,
        max_retries: int = 8,
        timeout: float = 60.0,
        session: requests.Session | None = None,
    ) -> None:
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._limiter = RateLimiter(requests_per_minute)
        self._max_retries = max_retries
        self._timeout = timeout
        self._session = session or requests.Session()
        self.request_count = 0

    @property
    def seconds_per_request(self) -> float:
        """Intervalo mínimo entre requests."""
        return self._limiter.interval

    def get(self, path_or_url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET con límite de tasa y reintentos ante 429, 5xx y errores de red."""
        url = path_or_url if path_or_url.startswith("http") else self._base + path_or_url
        clean = url.split("?")[0]
        query = {**(params or {}), "apiKey": self._key}
        for attempt in range(1, self._max_retries + 1):
            self._limiter.wait()
            self.request_count += 1
            try:
                response = self._session.get(url, params=query, timeout=self._timeout)
            except requests.RequestException as exc:
                self._backoff(attempt, clean, type(exc).__name__, None)
                continue
            status = response.status_code
            if status == 200:
                return response.json()
            if status in (401, 403):
                raise PolygonAccessError(
                    f"HTTP {status} en {clean}: {response.text[:300]}. La clave no "
                    "tiene acceso a este dato (plan o antigüedad de la historia)."
                )
            if status == 404:
                raise PolygonNotFound(clean)
            if status == 429 or status >= 500:
                self._backoff(attempt, clean, f"HTTP {status}",
                              response.headers.get("Retry-After"))
                continue
            raise RuntimeError(f"HTTP {status} en {clean}: {response.text[:300]}")
        raise RuntimeError(f"Se agotaron los reintentos para {clean}")

    def _backoff(self, attempt: int, url: str, reason: str, retry_after: str | None) -> None:
        """Espera exponencial desde el intervalo base; nunca menos que ``Retry-After``."""
        try:
            header = float(retry_after) if retry_after else 0.0
        except ValueError:
            header = 0.0
        delay = min(600.0, max(self._limiter.interval * 2 ** (attempt - 1), header))
        delay += random.uniform(0.0, 1.0)
        _LOG.warning("%s en %s (intento %d/%d). Reintento en %.0fs",
                     reason, url, attempt, self._max_retries, delay)
        time.sleep(delay)

    def pages(self, path: str, params: dict[str, Any]) -> Iterator[list[dict[str, Any]]]:
        """Itera las páginas siguiendo ``next_url``; cada página cuenta como request."""
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
# Calendario de vencimientos y universo
# --------------------------------------------------------------------------- #


def third_friday(year: int, month: int) -> date:
    """Tercer viernes del mes: vencimiento mensual estándar de opciones sobre acciones."""
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7 + 14)


def monthly_expirations(first: date, last: date) -> list[date]:
    """Terceros viernes comprendidos en ``[first, last]``."""
    out: list[date] = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        candidate = third_friday(year, month)
        if first <= candidate <= last:
            out.append(candidate)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def contract_windows(start: date, end: date, nearest: int) -> list[tuple[date, date, date]]:
    """Ventana de cada vencimiento mensual mientras es uno de los ``nearest`` más próximos.

    Returns:
        Lista de ``(vencimiento, inicio_ventana, fin_ventana)`` que se solapan
        con ``[start, end]``.
    """
    if nearest < 1:
        raise ValueError("nearest debe ser al menos 1.")
    span = timedelta(days=31 * (nearest + 1))
    monthlies = monthly_expirations(start - span, end + span)
    windows = []
    for index, expiration in enumerate(monthlies):
        if index < nearest:
            continue
        window_start = max(monthlies[index - nearest] + timedelta(days=1), start)
        window_end = min(expiration, end)
        if window_start <= window_end:
            windows.append((expiration, window_start, window_end))
    return windows


def select_strikes(strikes: Sequence[float], spot: float, band: float, per_side: int) -> list[float]:
    """Strikes dentro de ±``band`` del spot, los ``per_side`` más cercanos de cada lado.

    Args:
        strikes: Strikes listados.
        spot: Spot de referencia.
        band: Distancia relativa máxima al spot.
        per_side: Cuántos strikes tomar por debajo (incluido el ATM) y por
            encima. ``0`` toma todos los de la banda.
    """
    inside = sorted({float(k) for k in strikes if abs(float(k) / spot - 1.0) <= band})
    below = [k for k in inside if k <= spot]
    above = [k for k in inside if k > spot]
    if per_side:
        below, above = below[-per_side:], above[:per_side]
    return below + above


def reference_spot(daily: pd.DataFrame, window_start: date) -> float:
    """Cierre de la última rueda anterior a ``window_start``.

    Si no hay ninguna rueda previa descargada, se usa la apertura de la primera
    rueda de la ventana, que también se conoce antes de operar.
    """
    prior = daily.loc[daily["trade_date"] < window_start]
    if not prior.empty:
        return float(prior.iloc[-1]["close"])
    within = daily.loc[daily["trade_date"] >= window_start]
    if within.empty:
        raise RuntimeError(f"Sin precios del subyacente para {window_start}.")
    return float(within.iloc[0]["open"])


# --------------------------------------------------------------------------- #
# Persistencia auxiliar
# --------------------------------------------------------------------------- #


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


class CoverageIndex:
    """Registro de rangos ya descargados, para reanudar sin repetir requests."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, list[str]] = (
            json.loads(path.read_text("utf-8")) if path.exists() else {}
        )

    def covers(self, key: str, first: date, last: date) -> bool:
        span = self._data.get(key)
        return bool(span) and span[0] <= first.isoformat() and span[1] >= last.isoformat()

    def record(self, key: str, first: date, last: date) -> None:
        self._data[key] = [first.isoformat(), last.isoformat()]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=1), "utf-8")

    def start_of(self, key: str) -> date | None:
        span = self._data.get(key)
        return date.fromisoformat(span[0]) if span else None


def _aggs_frame(rows: list[dict[str, Any]], symbol: str) -> pd.DataFrame:
    """Convierte velas de ``/v2/aggs`` a ``timestamp, symbol, open, ..., trade_date``.

    Polygon sella la vela diaria con el inicio de la rueda (medianoche de Nueva
    York); ``trade_date`` es esa fecha en Nueva York.
    """
    columns = ["timestamp", "symbol", "open", "high", "low", "close",
               "volume", "vwap", "transactions", "trade_date"]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows).rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close",
                 "v": "volume", "vw": "vwap", "n": "transactions"}
    )
    frame["timestamp"] = pd.to_datetime(frame["t"], unit="ms", utc=True)
    frame["symbol"] = symbol
    frame["trade_date"] = frame["timestamp"].dt.tz_convert(NY).dt.date
    return frame.reindex(columns=columns)


def _safe_name(symbol: str) -> str:
    """``O:RTX...`` no es un nombre de archivo válido en Windows."""
    return symbol.replace(":", "_")


# --------------------------------------------------------------------------- #
# Pasos
# --------------------------------------------------------------------------- #


def download_stock_daily(
    client: PolygonClient, ticker: str, first: date, last: date,
    out: Path, coverage: CoverageIndex,
) -> pd.DataFrame:
    """Velas diarias de la acción, **sin ajustar**, en un único request."""
    path = out / "underlying_bars" / f"{ticker}_1day.parquet"
    key = f"stock:{ticker}"
    if path.exists() and coverage.covers(key, first, last):
        daily = pd.read_parquet(path)
    else:
        rows = list(client.paginate(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{first}/{last}",
            {"adjusted": "false", "sort": "asc", "limit": 50000},
        ))
        daily = _aggs_frame(rows, ticker)
        _write_parquet(daily, path)
        coverage.record(key, first, last)
        _LOG.info("Acción %s: %d ruedas", ticker, len(daily))
    daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.date
    return daily.sort_values("trade_date").reset_index(drop=True)


def download_rates(client: PolygonClient, first: date, last: date, out: Path) -> None:
    r"""Rendimientos del Tesoro y curva continua en el formato de ``config.py``.

    Los rendimientos vienen en porcentaje con capitalización semestral
    equivalente; se convierten con :math:`r = 2\ln(1 + y/200)`. El dato del día
    ``t`` se publica al cierre de ``t``: para valuar en ``t`` corresponde la
    curva de ``t-1``.
    """
    rows = list(client.paginate("/fed/v1/treasury-yields", {
        "date.gte": first.isoformat(), "date.lte": last.isoformat(),
        "limit": 50000, "sort": "date.asc",
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
    curve[["date", "tenor_days", "rate"]].sort_values(["date", "tenor_days"]).to_csv(
        out / "risk_free_curve.csv", index=False)
    _LOG.info("Tesoro: %d días", raw["date"].nunique())


def download_dividends(
    client: PolygonClient, tickers: Sequence[str], first: date, last: date, out: Path
) -> None:
    """Dividendos con fechas ex, de pago y monto; desde un año antes para el yield trailing."""
    frames = []
    since = (first - timedelta(days=400)).isoformat()
    for ticker in tickers:
        try:
            rows = list(client.paginate("/stocks/v1/dividends", {
                "ticker": ticker, "ex_dividend_date.gte": since,
                "ex_dividend_date.lte": last.isoformat(), "limit": 5000,
                "sort": "ex_dividend_date.asc",
            }))
        except PolygonNotFound:
            rows = list(client.paginate("/v3/reference/dividends", {
                "ticker": ticker, "ex_dividend_date.gte": since,
                "ex_dividend_date.lte": last.isoformat(), "limit": 1000,
                "sort": "ex_dividend_date", "order": "asc",
            }))
        frames.append(pd.DataFrame(rows).assign(ticker=ticker))
        _LOG.info("Dividendos %s: %d eventos", ticker, len(rows))
    _write_parquet(pd.concat(frames, ignore_index=True), out / "polygon" / "dividends.parquet")


def list_monthly_contracts(
    client: PolygonClient, ticker: str, expiration: date, spot: float,
    args: argparse.Namespace, out: Path,
) -> pd.DataFrame:
    """Contratos del vencimiento mensual, en un request (cacheado en disco).

    Se pide desde el jueves anterior para cubrir los meses en que el tercer
    viernes es feriado; si hay contratos del viernes, se descartan los del jueves.
    """
    lo, hi = round(spot * (1 - args.band), 2), round(spot * (1 + args.band), 2)
    cache = out / "polygon" / "contracts" / f"{ticker}_{expiration:%Y%m%d}_{lo:g}_{hi:g}.parquet"
    if cache.exists():
        raw = pd.read_parquet(cache)
    else:
        rows = list(client.paginate("/v3/reference/options/contracts", {
            "underlying_ticker": ticker,
            "expiration_date.gte": (expiration - timedelta(days=1)).isoformat(),
            "expiration_date.lte": expiration.isoformat(),
            "strike_price.gte": lo,
            "strike_price.lte": hi,
            "expired": "true" if expiration < date.today() else "false",
            "limit": 1000,
        }))
        raw = pd.DataFrame(rows)
        _write_parquet(raw, cache)
    if raw.empty:
        return raw
    expirations = pd.to_datetime(raw["expiration_date"]).dt.date
    target = expiration if (expirations == expiration).any() else expirations.max()
    return raw.loc[expirations == target].reset_index(drop=True)


def build_master(
    client: PolygonClient, windows: list[tuple[date, date, date]],
    stock: dict[str, pd.DataFrame], args: argparse.Namespace, out: Path,
) -> pd.DataFrame:
    """Selecciona los contratos de cada ticker y vencimiento con su ventana."""
    frames = []
    for ticker in args.tickers:
        for expiration, window_start, window_end in windows:
            spot = reference_spot(stock[ticker], window_start)
            raw = list_monthly_contracts(client, ticker, expiration, spot, args, out)
            if raw.empty:
                _LOG.warning("%s %s: sin contratos listados.", ticker, expiration)
                continue
            types = raw["contract_type"].astype(str).str.lower().str[0]
            raw = raw.loc[types.isin(args.option_types)]
            keep = select_strikes(raw["strike_price"], spot, args.band,
                                  args.max_strikes_per_side)
            raw = raw.loc[raw["strike_price"].astype(float).isin(keep)]
            frames.append(pd.DataFrame({
                "symbol": raw["ticker"],
                "underlying": ticker,
                "expiration": pd.to_datetime(raw["expiration_date"]),
                "strike": raw["strike_price"].astype(float),
                "option_type": raw["contract_type"].astype(str).str.lower().str[0],
                "style": raw.get("exercise_style", pd.Series("american", index=raw.index)),
                "multiplier": raw.get("shares_per_contract",
                                      pd.Series(100, index=raw.index)).astype(float),
                "window_start": pd.Timestamp(window_start),
                "window_end": pd.Timestamp(window_end),
                "reference_spot": spot,
            }))
            _LOG.info("%s %s: %d contratos (spot ref. %.2f)", ticker, expiration,
                      len(raw), spot)
    master = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not master.empty:
        master = master.drop_duplicates("symbol").reset_index(drop=True)
        _write_parquet(master, out / "polygon" / "contract_master.parquet")
    return master


def download_option_bars(
    client: PolygonClient, master: pd.DataFrame, out: Path, coverage: CoverageIndex,
) -> None:
    """Una request por contrato con toda su ventana, más unas ruedas previas."""
    pending = []
    for row in master.itertuples(index=False):
        first = row.window_start.date() - timedelta(days=_LOOKBACK_DAYS)
        last = row.window_end.date()
        if not coverage.covers(f"bars:{row.symbol}", first, last):
            pending.append((row.symbol, first, last))

    total = len(pending)
    eta = total * client.seconds_per_request / 60.0
    _LOG.info("Velas de opciones: %d contratos pendientes de %d (~%.0f min).",
              total, len(master), eta)
    for index, (symbol, first, last) in enumerate(pending, start=1):
        rows = list(client.paginate(
            f"/v2/aggs/ticker/{symbol}/range/1/day/{first}/{last}",
            {"adjusted": "false", "sort": "asc", "limit": 50000},
        ))
        _write_parquet(_aggs_frame(rows, symbol),
                       out / "polygon" / "option_bars" / f"{_safe_name(symbol)}.parquet")
        coverage.record(f"bars:{symbol}", first, last)
        if index % 10 == 0 or index == total:
            remaining = (total - index) * client.seconds_per_request / 60.0
            _LOG.info("Velas de opciones: %d/%d (quedan ~%.0f min)", index, total, remaining)


def build_daily_dataset(
    master: pd.DataFrame, stock: dict[str, pd.DataFrame], out: Path,
    coverage: CoverageIndex,
) -> int:
    """Arma ``option_daily`` a partir de la caché de velas, sin usar la red.

    Cada fila queda sellada al cierre de su rueda, con ``volume_prev_day``
    calculado contra el calendario del subyacente. El dataset se regenera entero
    porque sólo lo escribe este script.
    """
    frames = []
    for ticker, contracts in master.groupby("underlying"):
        pieces = []
        for symbol in contracts["symbol"]:
            path = out / "polygon" / "option_bars" / f"{_safe_name(symbol)}.parquet"
            if path.exists():
                bars = pd.read_parquet(path)
                if not bars.empty:
                    pieces.append(bars)
        if not pieces:
            continue
        bars = pd.concat(pieces, ignore_index=True)
        bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.date
        starts = {s: coverage.start_of(f"bars:{s}") for s in bars["symbol"].unique()}
        bars["volume_prev_day"] = previous_session_volume(
            bars, stock[str(ticker)]["trade_date"], {k: v for k, v in starts.items() if v}
        )
        bars = bars.merge(
            contracts[["symbol", "underlying", "expiration", "strike", "option_type",
                       "style", "multiplier", "window_start", "window_end"]],
            on="symbol", how="inner",
        )
        day = pd.to_datetime(bars["trade_date"])
        bars = bars.loc[(day >= bars["window_start"]) & (day <= bars["window_end"])]
        frames.append(bars)

    if not frames:
        _LOG.warning("No hay velas de opciones para armar el dataset.")
        return 0
    data = pd.concat(frames, ignore_index=True)
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data["timestamp"] = session_close(data["trade_date"])
    data["observed_at"] = data["timestamp"]
    data["bid"] = np.nan
    data["ask"] = np.nan
    data["last_trade_price"] = data["close"]
    data["open_interest"] = np.nan
    data["spread_is_proxy"] = True
    data["feed"] = "polygon_daily"
    data["source"] = "polygon_daily_bars"

    shutil.rmtree(out / RAW_DATASET, ignore_errors=True)
    written = ParquetStore(out, RAW_DATASET).write(data)
    _LOG.info("option_daily: %d filas de %d contratos", written, data["symbol"].nunique())
    return written


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #


def estimate_requests(args: argparse.Namespace, n_windows: int) -> dict[str, float]:
    """Cota superior de requests antes de tocar la red."""
    per_expiration = len(args.option_types) * (
        2 * args.max_strikes_per_side if args.max_strikes_per_side else 30
    )
    tickers = len(args.tickers)
    plan = {
        "stock": tickers if "stock" in args.steps else 0,
        "rates": 1 if "rates" in args.steps else 0,
        "dividends": tickers if "dividends" in args.steps else 0,
        "contracts": tickers * n_windows if {"contracts", "option_bars", "build"} & args.steps else 0,
        "option_bars": tickers * n_windows * per_expiration if "option_bars" in args.steps else 0,
    }
    plan["total"] = sum(plan.values())
    plan["minutes"] = plan["total"] / args.requests_per_minute
    return plan


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    yesterday = date.today() - timedelta(days=1)
    parser = argparse.ArgumentParser(
        description="Descarga diaria de opciones desde Polygon (plan gratuito).")
    parser.add_argument("--tickers", nargs="+", default=["RTX", "BA", "LMT"])
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, default=yesterday)
    parser.add_argument("--out", type=Path, default=ROOT / "data",
                        help="Raíz del data lake (la misma que usa el pipeline).")
    parser.add_argument("--nearest-monthlies", type=int, default=2,
                        help="Vencimientos mensuales más próximos a mantener en cada rueda.")
    parser.add_argument("--band", type=float, default=0.15,
                        help="Strikes dentro de ±band del spot de referencia.")
    parser.add_argument("--max-strikes-per-side", type=int, default=5,
                        help="Strikes por debajo y por encima del spot (0 = toda la banda).")
    parser.add_argument("--option-types", nargs="+", choices=["c", "p"], default=["c", "p"])
    parser.add_argument("--requests-per-minute", type=float, default=5.0,
                        help="Plan gratuito: 5. Subirlo sólo con un plan pago.")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--base-url", default="https://api.polygon.io",
                        help="api.polygon.io sigue activo; api.massive.com es el nuevo.")
    parser.add_argument("--steps", default=",".join(ALL_STEPS),
                        help=f"Subconjunto de: {', '.join(ALL_STEPS)}")
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra el plan de requests y el tiempo estimado, sin red.")
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
    windows = contract_windows(args.start, args.end, args.nearest_monthlies)
    plan = estimate_requests(args, len(windows))
    print(
        f"\nPlan: {len(windows)} vencimientos mensuales x {len(args.tickers)} tickers.\n"
        f"Requests (cota superior): acción {plan['stock']:.0f}, Tesoro {plan['rates']:.0f}, "
        f"dividendos {plan['dividends']:.0f}, listados {plan['contracts']:.0f}, "
        f"velas de opciones {plan['option_bars']:.0f}.\n"
        f"Total {plan['total']:.0f} requests a {args.requests_per_minute:g}/min: "
        f"~{plan['minutes'] / 60:.1f} horas (menos si hay caché).\n"
    )
    if args.dry_run:
        return

    api_key = os.environ.get("POLYGON_API_KEY") or os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        raise SystemExit("Falta POLYGON_API_KEY (o MASSIVE_API_KEY) en el entorno.")

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    client = PolygonClient(api_key, args.base_url, args.requests_per_minute, args.max_retries)
    coverage = CoverageIndex(out / "polygon" / "coverage.json")
    first = args.start - timedelta(days=_LOOKBACK_DAYS + 5)
    started = time.monotonic()

    try:
        # La acción siempre hace falta: calendario, spot de referencia y spot
        # de liquidación.
        stock = {t: download_stock_daily(client, t, first, args.end, out, coverage)
                 for t in args.tickers}
        if "rates" in args.steps:
            download_rates(client, first, args.end, out)
        if "dividends" in args.steps:
            download_dividends(client, args.tickers, args.start, args.end, out)
        if {"contracts", "option_bars", "build"} & args.steps:
            master = build_master(client, windows, stock, args, out)
            _LOG.info("Universo: %d contratos", len(master))
            if not master.empty and "option_bars" in args.steps:
                download_option_bars(client, master, out, coverage)
            if not master.empty and "build" in args.steps:
                build_daily_dataset(master, stock, out, coverage)
    except PolygonAccessError as exc:
        raise SystemExit(f"\n{exc}\n") from exc
    finally:
        manifest = {
            "source": "polygon_daily",
            "base_url": args.base_url,
            "tickers": args.tickers,
            "start": args.start.isoformat(),
            "end": args.end.isoformat(),
            "nearest_monthlies": args.nearest_monthlies,
            "band": args.band,
            "max_strikes_per_side": args.max_strikes_per_side,
            "option_types": args.option_types,
            "requests_per_minute": args.requests_per_minute,
            "steps": sorted(args.steps),
            "requests": client.request_count,
            "elapsed_s": round(time.monotonic() - started, 1),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (out / f"polygon_manifest_{stamp}.json").write_text(
            json.dumps(manifest, indent=2), "utf-8")


if __name__ == "__main__":
    main()
