r"""Descarga de cadenas de opciones **con bid y ask observados** desde DoltHub.

Por qué DoltHub
---------------
La entrega parcial ejecutaba el backtest contra un bid/ask sintético construido
alrededor de cierres no simultáneos. La devolución lo marcó como la causa del
97,8% de acierto. Alpaca no tiene histórico de NBBO de opciones y el plan
gratuito de Polygon sólo da velas, así que la fuente de cotizaciones reales es
la base pública ``post-no-preference/options`` de DoltHub: una foto diaria de la
cadena con ``bid``, ``ask``, volatilidad implícita y griegas del proveedor.

Qué NO es
---------
Es una foto por rueda, no un NBBO sincronizado con el cierre de la acción. La
hora exacta de la foto no está documentada. ``ita_options.dolthub`` mide el
desfase contra el spot implícito por paridad y descarta los días en que la foto
y el cierre de la acción no son comparables.

Diseño
------
- **Una consulta por rueda** con los tres subyacentes juntos. Las consultas por
  rango de fechas fallan del lado de DoltHub (timeout), y una cadena diaria de
  un ticker tiene del orden de 200 filas, así que tres tickers entran holgados
  bajo el límite de 1.000 filas por respuesta. Si una respuesta llega justo al
  límite, se repite ticker por ticker.
- **Paralelismo acotado** (``--workers``, 6 por defecto): cada consulta tarda
  entre 30 y 60 segundos del lado del servidor.
- **Caché por rueda** en ``data/dolthub/raw/<fecha>.json``: se puede cortar y
  volver a correr, y sólo pide lo que falta.
- **Calendario y spot** desde Yahoo Finance (velas diarias **sin ajustar por
  dividendos**, que es la convención de los strikes), más dividendos con fecha
  ex y la tasa de la letra a 13 semanas (``^IRX``) como tasa libre de riesgo.

Salida::

    data/dolthub/raw/2026-03-02.json        caché crudo por rueda
    data/dolthub/option_chain.parquet       cadena consolidada
    data/underlying_bars/<T>_1day.parquet   OHLCV diario sin ajustar
    data/dolthub/dividends.parquet          fechas ex y montos
    data/dolthub/rates.parquet              tasa ^IRX diaria (decimal)
    data/dolthub/manifest.json              qué se pidió, qué llegó y cuándo

Uso (desde ``ita_options_pipeline``; sólo librería estándar más pandas)::

    python scripts/download_dolthub.py --start 2026-03-02 --end 2026-11-20
    python scripts/download_dolthub.py --start 2026-03-02 --end 2026-11-20 --steps build
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DOLT_URL = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
ROW_LIMIT = 1000
DEFAULT_TICKERS = ("RTX", "BA", "LMT")
ALL_STEPS = ("yahoo", "chains", "build")

_LOG = logging.getLogger("download_dolthub")
_UA = {"User-Agent": "Mozilla/5.0 (ita-options research; UdeSA MAPF)"}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def _get_json(url: str, timeout: float = 180.0, attempts: int = 4) -> dict[str, Any]:
    """GET con reintentos y backoff exponencial.

    Args:
        url: URL completa.
        timeout: Segundos por intento.
        attempts: Intentos totales.

    Returns:
        El JSON decodificado.

    Raises:
        RuntimeError: si fallan todos los intentos.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            wait = 5 * 2**attempt
            _LOG.warning("Intento %d/%d falló (%s); reintento en %ds", attempt + 1, attempts, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Fallaron {attempts} intentos contra {url[:120]}...: {last}")


def dolt_query(sql: str) -> list[dict[str, Any]]:
    """Ejecuta una consulta SQL de sólo lectura contra la API de DoltHub.

    Raises:
        RuntimeError: si DoltHub responde con un estado distinto de ``Success``.
    """
    last = ""
    for attempt in range(4):
        payload = _get_json(f"{DOLT_URL}?q={urllib.parse.quote(sql)}")
        status = payload.get("query_execution_status")
        if status in ("Success", "RowLimit"):
            return list(payload.get("rows", []))
        last = f"{status} {payload.get('query_execution_message')}"
        # "context deadline exceeded" es un timeout del servidor: se reintenta.
        time.sleep(10 * 2**attempt)
    raise RuntimeError(f"DoltHub: {last}")


# --------------------------------------------------------------------------- #
# Yahoo: calendario, spot, dividendos y tasa
# --------------------------------------------------------------------------- #


def _epoch(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def yahoo_daily(symbol: str, start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Velas diarias sin ajustar por dividendos y eventos de dividendo.

    En la API ``v8/chart`` de Yahoo el campo ``close`` está ajustado por splits
    pero **no** por dividendos (ese es ``adjclose``). Es la convención correcta
    para compararlo con strikes de opciones listadas.

    Returns:
        ``(barras, dividendos)``. Las barras tienen ``date, open, high, low,
        close, volume``; los dividendos, ``ex_dividend_date, cash_amount``.
    """
    params = urllib.parse.urlencode({
        "period1": _epoch(start), "period2": _epoch(end) + 86400,
        "interval": "1d", "events": "div", "includeAdjustedClose": "true",
    })
    payload = _get_json(f"{YAHOO_URL.format(symbol=urllib.parse.quote(symbol))}?{params}", timeout=60)
    result = payload["chart"]["result"][0]
    stamps = result.get("timestamp") or []
    quote = result["indicators"]["quote"][0]
    tz = result["meta"].get("exchangeTimezoneName", "America/New_York")
    days = pd.to_datetime(stamps, unit="s", utc=True).tz_convert(tz).date
    bars = pd.DataFrame({
        "date": days,
        "open": quote.get("open"), "high": quote.get("high"), "low": quote.get("low"),
        "close": quote.get("close"), "volume": quote.get("volume"),
    }).dropna(subset=["close"])
    events = (result.get("events") or {}).get("dividends") or {}
    dividends = pd.DataFrame(
        [{"ex_dividend_date": pd.Timestamp(int(k), unit="s", tz="UTC").tz_convert(tz).date(),
          "cash_amount": float(v["amount"])} for k, v in events.items()],
        columns=["ex_dividend_date", "cash_amount"],
    )
    return bars.reset_index(drop=True), dividends


def download_yahoo(tickers: Sequence[str], start: date, end: date) -> list[date]:
    """Baja spot, dividendos y tasa; devuelve el calendario de ruedas.

    El calendario sale de las ruedas en que operó el primer ticker: así no se
    le pide a DoltHub ningún feriado.
    """
    (DATA / "underlying_bars").mkdir(parents=True, exist_ok=True)
    (DATA / "dolthub").mkdir(parents=True, exist_ok=True)
    calendar: list[date] = []
    dividends = []
    for ticker in tickers:
        # Los dividendos se piden con un año de historia: la proyección
        # point-in-time necesita el último pagado antes del inicio de la muestra.
        bars, divs = yahoo_daily(ticker, start - timedelta(days=370), end)
        bars = bars.loc[bars["date"] >= start].reset_index(drop=True)
        bars.insert(0, "symbol", ticker)
        bars.to_parquet(DATA / "underlying_bars" / f"{ticker}_1day.parquet", index=False)
        divs.insert(0, "ticker", ticker)
        dividends.append(divs)
        if not calendar:
            calendar = sorted(bars["date"].tolist())
        _LOG.info("%s: %d ruedas, %d dividendos", ticker, len(bars), len(divs))
    pd.concat(dividends, ignore_index=True).to_parquet(DATA / "dolthub" / "dividends.parquet", index=False)

    irx, _ = yahoo_daily("^IRX", start, end)
    rates = irx[["date", "close"]].rename(columns={"close": "rate_pct"})
    rates["rate"] = rates["rate_pct"] / 100.0
    rates.to_parquet(DATA / "dolthub" / "rates.parquet", index=False)
    _LOG.info("^IRX: %d observaciones, última %.3f%%", len(rates), rates["rate_pct"].iloc[-1])
    return calendar


# --------------------------------------------------------------------------- #
# DoltHub: una consulta por rueda
# --------------------------------------------------------------------------- #


_COLUMNS = "date, act_symbol, expiration, strike, call_put, bid, ask, vol, delta, gamma, theta, vega, rho"


def _chain_sql(day: date, tickers: Sequence[str], max_dte: int) -> str:
    symbols = ", ".join(f"'{t}'" for t in tickers)
    return (
        f"SELECT {_COLUMNS} FROM option_chain "
        f"WHERE date = '{day.isoformat()}' AND act_symbol IN ({symbols}) "
        f"AND expiration <= DATE_ADD('{day.isoformat()}', INTERVAL {max_dte} DAY)"
    )


def fetch_day(day: date, tickers: Sequence[str], max_dte: int) -> tuple[date, int]:
    """Descarga la cadena de una rueda y la guarda en el caché.

    Si la respuesta toca el límite de filas, se repite por ticker para no
    truncar en silencio.

    Returns:
        ``(fecha, filas)``.
    """
    path = DATA / "dolthub" / "raw" / f"{day.isoformat()}.json"
    if path.exists():
        return day, len(json.loads(path.read_text(encoding="utf-8")))
    rows = dolt_query(_chain_sql(day, tickers, max_dte))
    if len(rows) >= ROW_LIMIT:
        rows = []
        for ticker in tickers:
            part = dolt_query(_chain_sql(day, [ticker], max_dte))
            if len(part) >= ROW_LIMIT:
                raise RuntimeError(f"{ticker} {day}: la cadena supera {ROW_LIMIT} filas; bajar --max-dte.")
            rows.extend(part)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # No se guarda en el caché: una rueda vacía puede ser un hueco de la base
        # o una respuesta transitoria, y la próxima corrida la vuelve a pedir.
        return day, 0
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows), encoding="utf-8")
    tmp.replace(path)
    return day, len(rows)


def download_chains(calendar: Sequence[date], tickers: Sequence[str], max_dte: int, workers: int) -> dict[str, int]:
    """Descarga en paralelo las ruedas que faltan en el caché."""
    done: dict[str, int] = {}
    failed: list[str] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_day, d, tickers, max_dte): d for d in calendar}
        for n, future in enumerate(as_completed(futures), start=1):
            day = futures[future]
            try:
                _, rows = future.result()
                done[day.isoformat()] = rows
                level = logging.WARNING if rows == 0 else logging.INFO
                _LOG.log(level, "[%d/%d %.0fs] %s: %d filas", n, len(futures),
                         time.monotonic() - started, day, rows)
            except Exception as exc:  # noqa: BLE001 - se registra y se sigue
                failed.append(day.isoformat())
                _LOG.error("%s falló: %s", day, exc)
    if failed:
        _LOG.error("Fallaron %d ruedas; volver a correr para completarlas: %s", len(failed), failed)
    return done


# --------------------------------------------------------------------------- #
# Consolidación
# --------------------------------------------------------------------------- #


def build_chain() -> pd.DataFrame:
    """Une el caché en ``option_chain.parquet`` con tipos y nombres del pipeline."""
    files = sorted((DATA / "dolthub" / "raw").glob("*.json"))
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit("El caché está vacío: correr primero el paso 'chains'.")
    frame = pd.DataFrame(rows).rename(columns={"act_symbol": "underlying", "vol": "provider_iv"})
    for column in ("strike", "bid", "ask", "provider_iv", "delta", "gamma", "theta", "vega", "rho"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["trade_date"] = pd.to_datetime(frame.pop("date")).dt.date
    frame["expiration"] = pd.to_datetime(frame["expiration"]).dt.date
    frame["option_type"] = frame.pop("call_put").str[0].str.lower()
    frame = frame.drop_duplicates(["trade_date", "underlying", "expiration", "strike", "option_type"])
    frame.to_parquet(DATA / "dolthub" / "option_chain.parquet", index=False)
    _LOG.info("option_chain.parquet: %d filas, %d ruedas, %s a %s", len(frame),
              frame["trade_date"].nunique(), frame["trade_date"].min(), frame["trade_date"].max())
    return frame


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--max-dte", type=int, default=200, help="Vencimientos hasta N días corridos.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=list(ALL_STEPS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    calendar: list[date] = []
    if "yahoo" in args.steps:
        calendar = download_yahoo(args.tickers, args.start, args.end)
    elif "chains" in args.steps:
        bars = pd.read_parquet(DATA / "underlying_bars" / f"{args.tickers[0]}_1day.parquet")
        calendar = sorted(d for d in bars["date"] if args.start <= d <= args.end)
    done: dict[str, int] = {}
    if "chains" in args.steps:
        done = download_chains(calendar, args.tickers, args.max_dte, args.workers)
    if "build" in args.steps:
        frame = build_chain()
        manifest = {
            "fuente": "dolthub post-no-preference/options (option_chain) + Yahoo v8 chart",
            "generado_utc": datetime.now(timezone.utc).isoformat(),
            "start": args.start.isoformat(), "end": args.end.isoformat(),
            "tickers": args.tickers, "max_dte": args.max_dte,
            "ruedas_calendario": len(calendar) or None,
            "ruedas_con_datos": int(frame["trade_date"].nunique()),
            "ruedas_vacias": sorted(d for d, n in done.items() if n == 0),
            "filas": int(len(frame)),
            "filas_por_subyacente": frame["underlying"].value_counts().to_dict(),
        }
        (DATA / "dolthub" / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
