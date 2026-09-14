"""Modo diario: el pipeline completo sobre velas EOD en lugar de NBBO intradía.

Por qué existe
--------------
El plan gratuito de Polygon no da cotizaciones históricas de opciones: sólo
velas diarias (OHLCV y VWAP) por contrato. Este módulo adapta enriquecimiento,
filtros y backtest a ese insumo sin tocar los detectores.

Qué cambia respecto del modo NBBO
---------------------------------
**Instante de cada fila.** Una vela diaria se conoce recién al cierre, así que
cada fila se sella a las 16:00 de Nueva York de su rueda. El spot del
subyacente se sella igual y se aparea sin corrimiento.

**Bid y ask.** No existen. Se construyen alrededor del precio de referencia
(``close`` o ``vwap``) con un spread relativo *supuesto*
(:class:`~ita_options.config.DailyBarAssumptions`) y quedan marcados con
``spread_is_proxy=True``. Los detectores siguen comprando al ask y vendiendo al
bid, pero ese costo es un supuesto, no una observación.

**Liquidez.** Se mide con el volumen de la rueda anterior (``volume_prev_day``),
sin spread ni open interest.

**Ejecución.** La señal se detecta al cierre de ``t`` y se ejecuta al cierre de
la rueda siguiente como mínimo, re-cotizando las patas con esos precios.
Ejecutar al mismo cierre que generó la señal sería lookahead.

Advertencia metodológica
------------------------
El ``close`` de cada contrato es el precio de su **último trade**, que puede
haber ocurrido a horas distintas en contratos distintos. Comparar cierres no
sincrónicos genera violaciones de no arbitraje aparentes, sobre todo en strikes
poco operados. El filtro de volumen lo mitiga pero no lo elimina: cualquier
oportunidad detectada en este modo debe presentarse con esa salvedad.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .arbitrage import ExecutionCosts
from .backtest import ArbitrageBacktester, BacktestResult, StrategyParams
from .config import DailyBarAssumptions, LiquidityThresholds, PricingAssumptions
from .enrich import (
    align_underlying,
    compute_implied_volatility,
    compute_mid_and_spread,
    compute_time_to_expiry,
    quotes_from_daily_bars,
)
from .filters import FilterReport, apply_liquidity_filters
from .storage import ParquetStore

_LOG = logging.getLogger(__name__)

__all__ = [
    "NY",
    "RAW_DATASET",
    "CURATED_DATASET",
    "session_close",
    "previous_session_volume",
    "load_underlying_daily",
    "enrich_daily",
    "run_enrich_daily",
    "daily_strategy_params",
    "run_backtest_daily",
]

NY = ZoneInfo("America/New_York")

#: Dataset crudo que escribe ``scripts/download_polygon.py``. Es distinto de
#: ``option_quotes`` para no mezclar velas con el NBBO grabado por Alpaca.
RAW_DATASET = "option_daily"
CURATED_DATASET = "curated_daily"


def session_close(trade_dates: pd.Series | Iterable[date]) -> pd.Series:
    """Instante de cierre de cada rueda: 16:00 de Nueva York, expresado en UTC.

    El huso de Nueva York resuelve el horario de verano: el cierre cae a las
    20:00 o a las 21:00 UTC según la época del año.

    Args:
        trade_dates: Fechas de rueda.

    Returns:
        Serie de timestamps UTC, con el índice de la entrada si era una Serie.
    """
    series = trade_dates if isinstance(trade_dates, pd.Series) else pd.Series(list(trade_dates))
    dates = pd.to_datetime(series)
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert(NY).dt.tz_localize(None)
    closes = dates.dt.normalize() + pd.Timedelta(hours=16)
    return closes.dt.tz_localize(NY).dt.tz_convert("UTC")


def previous_session_volume(
    bars: pd.DataFrame,
    calendar: Sequence[date] | pd.Series,
    series_start: Mapping[str, date] | None = None,
) -> pd.Series:
    """Volumen de cada contrato en la rueda anterior del calendario.

    Polygon no emite vela los días en que un contrato no operó, así que la
    ausencia de vela en la rueda anterior significa volumen cero. La excepción
    es cuando esa rueda cae antes del inicio de la descarga: ahí no se sabe si
    operó y el valor queda en ``NaN``.

    Args:
        bars: Velas con ``symbol``, ``trade_date`` y ``volume``.
        calendar: Ruedas del subyacente.
        series_start: Primer día descargado por contrato. Por defecto, la
            fecha de su primera vela.

    Returns:
        Serie alineada al índice de ``bars``.
    """
    sessions = sorted(pd.to_datetime(pd.Series(list(calendar))).dt.date.unique())
    previous = {day: sessions[i - 1] for i, day in enumerate(sessions) if i > 0}
    days = pd.to_datetime(bars["trade_date"]).dt.date
    volumes = pd.to_numeric(bars["volume"], errors="coerce").fillna(0.0)
    lookup = {
        (symbol, day): float(volume)
        for symbol, day, volume in zip(bars["symbol"], days, volumes, strict=True)
    }
    if series_start is None:
        firsts = days.groupby(bars["symbol"]).min().to_dict()
    else:
        firsts = {k: pd.Timestamp(v).date() for k, v in series_start.items()}

    values: list[float] = []
    for symbol, day in zip(bars["symbol"], days, strict=True):
        prev = previous.get(day)
        start = firsts.get(symbol)
        if prev is None or start is None or prev < start:
            values.append(np.nan)
        else:
            values.append(lookup.get((symbol, prev), 0.0))
    return pd.Series(values, index=bars.index, dtype="float64")


def load_underlying_daily(data_root: Path, underlyings: Sequence[str]) -> pd.DataFrame:
    """Lee las velas diarias del subyacente y las sella al cierre de la rueda.

    Raises:
        FileNotFoundError: si falta el archivo de algún ticker.
    """
    frames = []
    for ticker in underlyings:
        path = Path(data_root) / "underlying_bars" / f"{ticker}_1day.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Falta {path}: correr scripts/download_polygon.py para {ticker}."
            )
        bars = pd.read_parquet(path)
        stamps = pd.to_datetime(bars["timestamp"], utc=True)
        bars["trade_date"] = stamps.dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
        bars["timestamp"] = session_close(bars["trade_date"])
        bars["symbol"] = ticker
        frames.append(bars)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )


def enrich_daily(
    raw: pd.DataFrame,
    underlying_daily: pd.DataFrame,
    pricing: PricingAssumptions | None = None,
    liquidity: LiquidityThresholds | None = None,
    assumptions: DailyBarAssumptions | None = None,
) -> tuple[pd.DataFrame, FilterReport]:
    """Encadena la curación de velas diarias: lados sintéticos, spot, IV y filtros.

    Args:
        raw: Velas diarias de opciones con ``timestamp`` al cierre de la rueda.
        underlying_daily: Salida de :func:`load_underlying_daily`.
        pricing: Tasa, dividendos y pasos del árbol.
        liquidity: Umbrales en modo ``daily``.
        assumptions: Fuente de precio y spread supuesto.

    Returns:
        Tupla ``(dataset curado, reporte de filtros)``.

    Raises:
        ValueError: si los umbrales no están en modo ``daily``.
    """
    pricing = pricing or PricingAssumptions()
    liquidity = liquidity or LiquidityThresholds.for_daily_bars()
    assumptions = assumptions or DailyBarAssumptions()
    if liquidity.data_mode != "daily":
        raise ValueError("enrich_daily necesita LiquidityThresholds en modo 'daily'.")

    frame = quotes_from_daily_bars(raw, assumptions)
    frame = compute_mid_and_spread(frame)
    # Ambas series están selladas al cierre de la rueda: el spot vigente es el
    # de la misma rueda, sin corrimiento, y nunca el de una rueda distinta.
    frame = align_underlying(
        frame, underlying_daily, bar_duration=timedelta(0), tolerance=timedelta(hours=1)
    )
    frame = compute_time_to_expiry(frame, pricing.day_count)
    frame = compute_implied_volatility(frame, pricing)
    return apply_liquidity_filters(frame, liquidity, drop=False)


def _has_data(store: ParquetStore) -> bool:
    return any(store.path.rglob("*.parquet"))


def run_enrich_daily(
    data_root: Path,
    underlyings: Sequence[str],
    start: str | None = None,
    end: str | None = None,
    pricing: PricingAssumptions | None = None,
    liquidity: LiquidityThresholds | None = None,
    assumptions: DailyBarAssumptions | None = None,
) -> tuple[pd.DataFrame, FilterReport]:
    """Lee ``option_daily``, cura y reescribe ``curated_daily`` con su manifiesto.

    ``curated_daily`` se regenera entero en cada corrida: ``ParquetStore`` es
    append-only y sumar una segunda curación con otros supuestos mezclaría dos
    datasets incompatibles.
    """
    raw_store = ParquetStore(data_root, RAW_DATASET)
    if not _has_data(raw_store):
        raise RuntimeError(
            f"No hay velas diarias en {raw_store.path}: correr scripts/download_polygon.py."
        )
    raw = raw_store.read(underlyings=list(underlyings), start=start, end=end)
    if raw.empty:
        raise RuntimeError("No hay velas diarias de opciones en el rango pedido.")

    pricing = pricing or PricingAssumptions()
    liquidity = liquidity or LiquidityThresholds.for_daily_bars()
    assumptions = assumptions or DailyBarAssumptions()
    underlying = load_underlying_daily(data_root, underlyings)
    frame, report = enrich_daily(raw, underlying, pricing, liquidity, assumptions)

    shutil.rmtree(Path(data_root) / CURATED_DATASET, ignore_errors=True)
    curated = ParquetStore(data_root, CURATED_DATASET)
    curated.write(frame)
    curated.write_manifest(
        {
            "mode": "daily",
            "underlyings": list(underlyings),
            "start": start,
            "end": end,
            "pricing": asdict(pricing),
            "liquidity": asdict(liquidity),
            "daily_assumptions": asdict(assumptions),
            "rows": len(frame),
            "tradable_rows": int(frame["is_tradable"].sum()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        name="manifest_daily",
    )
    return frame, report


def daily_strategy_params(lag_sessions: int = 1, **overrides: object) -> StrategyParams:
    """Hiperparámetros del backtest para velas diarias.

    El backtest ejecuta en el primer corte posterior a ``señal + lag``. Un lag
    de 12 horas más 24 por cada rueda adicional cae siempre en el cierre de la
    rueda buscada, aunque medien un fin de semana o un cambio de horario.

    Args:
        lag_sessions: Ruedas entre la señal y la ejecución. Mínimo 1.
        **overrides: Cualquier otro campo de :class:`StrategyParams`.

    Raises:
        ValueError: si ``lag_sessions`` es menor que 1.
    """
    if lag_sessions < 1:
        raise ValueError(
            "Con velas diarias la latencia mínima es una rueda: ejecutar al mismo "
            "cierre que generó la señal sería lookahead."
        )
    base: dict[str, object] = {
        "execution_lag": timedelta(hours=12 + 24 * (lag_sessions - 1)),
        "min_open_interest": 0.0,
        "min_volume": 1.0,
    }
    return StrategyParams(**{**base, **overrides})  # type: ignore[arg-type]


def run_backtest_daily(
    data_root: Path,
    underlyings: Sequence[str],
    params: StrategyParams,
    costs: ExecutionCosts | None = None,
    start: str | None = None,
    end: str | None = None,
) -> BacktestResult:
    """Corre el backtest sobre ``curated_daily``.

    Se pasa el dataset completo y no sólo las filas explotables: los detectores
    filtran por ``is_tradable``, pero para marcar y cerrar una posición hace
    falta el precio del contrato aunque ese día haya dejado de ser líquido.
    """
    store = ParquetStore(data_root, CURATED_DATASET)
    if not _has_data(store):
        raise RuntimeError("No hay dataset curado diario: correr 'enrich-daily'.")
    curated = store.read(underlyings=list(underlyings), start=start, end=end)
    if curated.empty:
        raise RuntimeError("El dataset curado diario está vacío en el rango pedido.")
    underlying = load_underlying_daily(data_root, underlyings)
    return ArbitrageBacktester(curated, underlying, costs).run(params)
