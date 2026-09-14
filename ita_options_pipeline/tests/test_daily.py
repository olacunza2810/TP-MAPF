"""Tests del modo diario: descarga acotada, filtros por volumen y backtest EOD."""

from __future__ import annotations

import importlib.util
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ita_options.arbitrage import ExecutionCosts
from ita_options.config import DailyBarAssumptions, LiquidityThresholds, PricingAssumptions
from ita_options.daily import (
    NY,
    daily_strategy_params,
    previous_session_volume,
    run_backtest_daily,
    run_enrich_daily,
    session_close,
)
from ita_options.enrich import quotes_from_daily_bars
from ita_options.filters import apply_liquidity_filters
from ita_options.storage import ParquetStore
from ita_options.volatility import crr_american_price

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_polygon.py"
_spec = importlib.util.spec_from_file_location("download_polygon", _SCRIPT)
dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dp)


# --------------------------------------------------------------------------- #
# Script de descarga
# --------------------------------------------------------------------------- #


def test_vencimientos_mensuales_y_ventanas() -> None:
    """Cada vencimiento se descarga sólo mientras es uno de los N más próximos."""
    assert dp.third_friday(2025, 3) == date(2025, 3, 21)
    assert dp.third_friday(2025, 8) == date(2025, 8, 15)
    windows = {e: (a, b) for e, a, b in
               dp.contract_windows(date(2025, 3, 1), date(2025, 3, 31), nearest=2)}
    assert set(windows) == {date(2025, 3, 21), date(2025, 4, 18), date(2025, 5, 16)}
    assert windows[date(2025, 3, 21)] == (date(2025, 3, 1), date(2025, 3, 21))
    # Mayo recién es uno de los dos más próximos después del vencimiento de marzo.
    assert windows[date(2025, 5, 16)] == (date(2025, 3, 22), date(2025, 3, 31))


def test_estimacion_de_requests_y_tiempo() -> None:
    """3 tickers, 14 vencimientos, 5 strikes por lado: el plan cabe en pocas horas."""
    args = dp._parse_args(["--start", "2025-01-02", "--end", "2025-12-31"])
    windows = dp.contract_windows(args.start, args.end, args.nearest_monthlies)
    plan = dp.estimate_requests(args, len(windows))
    assert plan["option_bars"] == 3 * len(windows) * 2 * 10
    assert plan["minutes"] == plan["total"] / 5.0


def test_seleccion_de_strikes_cerca_del_dinero() -> None:
    strikes = [90, 95, 100, 105, 110, 115, 120, 130]
    assert dp.select_strikes(strikes, spot=103, band=0.15, per_side=2) == [
        95.0, 100.0, 105.0, 110.0
    ]


def test_limitador_por_defecto_y_reintento_ante_429(monkeypatch) -> None:
    """Por defecto 12 s entre requests, y un 429 se reintenta sin perder el dato."""
    sleeps: list[float] = []
    monkeypatch.setattr(dp.time, "sleep", sleeps.append)
    monkeypatch.setattr(dp.random, "uniform", lambda a, b: 0.0)

    class Response:
        def __init__(self, status: int, payload: dict | None = None) -> None:
            self.status_code, self._payload = status, payload or {}
            self.headers: dict[str, str] = {}
            self.text = ""

        def json(self) -> dict:
            return self._payload

    queue = [Response(429), Response(200, {"results": [{"x": 1}]})]

    class Session:
        def get(self, url, params, timeout):
            assert params["apiKey"] == "clave"
            return queue.pop(0)

    client = dp.PolygonClient("clave", session=Session())
    assert client.seconds_per_request == 12.0
    assert list(client.paginate("/v3/algo", {})) == [{"x": 1}]
    assert client.request_count == 2
    assert max(sleeps) >= 12.0


# --------------------------------------------------------------------------- #
# Enriquecimiento y filtros
# --------------------------------------------------------------------------- #


def test_volumen_de_la_rueda_anterior() -> None:
    calendar = [date(2025, 3, 3), date(2025, 3, 4), date(2025, 3, 5), date(2025, 3, 6)]
    bars = pd.DataFrame({
        "symbol": ["A", "A", "A"],
        "trade_date": [date(2025, 3, 3), date(2025, 3, 4), date(2025, 3, 6)],
        "volume": [10.0, 20.0, 5.0],
    })
    prev = previous_session_volume(bars, calendar, {"A": date(2025, 3, 3)})
    assert np.isnan(prev.iloc[0])  # la rueda anterior no se descargó
    assert prev.iloc[1] == 10.0
    assert prev.iloc[2] == 0.0  # el 5/3 no hubo vela: no operó


def test_bid_ask_sintetico_desde_la_vela() -> None:
    bars = pd.DataFrame({"close": [2.0, 0.10], "vwap": [1.9, np.nan]})
    out = quotes_from_daily_bars(
        bars, DailyBarAssumptions(price_source="vwap", assumed_relative_spread=0.10)
    )
    assert out["bid"].round(4).tolist() == [1.805, 0.09]
    assert out["ask"].round(4).tolist() == [1.995, 0.11]
    assert out["spread_is_proxy"].all()


def test_filtro_diario_usa_volumen_anterior_y_no_spread_ni_oi() -> None:
    frame = pd.DataFrame({
        "volume_prev_day": [0.0, 15.0, np.nan, 15.0],
        "mid_price": [1.0, 1.0, 1.0, 0.01],
        "dte": [30] * 4,
        "underlying_price": [100.0] * 4,
        "open_interest": [np.nan] * 4,
        "bid": [0.0] * 4,
        "spread_rel": [0.9] * 4,
        "spread_abs": [0.0] * 4,
        "volume": [np.nan] * 4,
    })
    out, report = apply_liquidity_filters(frame, LiquidityThresholds.for_daily_bars())
    assert out["is_tradable"].tolist() == [False, True, False, False]
    assert "open_interest_insuficiente" not in report.exclusions
    assert "spread_relativo_excesivo" not in report.exclusions


# --------------------------------------------------------------------------- #
# De punta a punta
# --------------------------------------------------------------------------- #


def _write_synthetic_daily_lake(root: Path) -> list[date]:
    """Velas diarias sin arbitraje salvo un call dislocado durante dos ruedas."""
    sessions = [d.date() for d in pd.bdate_range("2025-03-03", "2025-03-21")]
    spot, sigma, rate, div = 100.0, 0.25, 0.0425, 0.016
    expiry = pd.Timestamp("2025-04-17")
    closes = session_close(pd.Series(sessions))

    rows = []
    for index, (day, stamp) in enumerate(zip(sessions, closes, strict=True)):
        expiry_instant = (expiry + pd.Timedelta(hours=20)).tz_localize("UTC")
        tau = (expiry_instant - stamp).total_seconds() / (365 * 86400)
        for strike in (90.0, 95.0, 100.0, 105.0, 110.0):
            for flag in ("c", "p"):
                price = crr_american_price(spot, strike, tau, rate, div, sigma, flag, 64)
                if flag == "c" and strike == 100.0 and index in (5, 6):
                    price += 3.0
                rows.append({
                    "symbol": f"O:RTX250417{flag.upper()}{int(strike * 1000):08d}",
                    "underlying": "RTX", "timestamp": stamp, "observed_at": stamp,
                    "expiration": expiry, "strike": strike, "option_type": flag,
                    "style": "american", "multiplier": 100.0,
                    "open": price, "high": price, "low": price,
                    "close": round(price, 2), "vwap": round(price, 2),
                    "volume": 100.0, "trade_date": pd.Timestamp(day),
                })
    raw = pd.DataFrame(rows)
    raw["volume_prev_day"] = previous_session_volume(raw, sessions)
    ParquetStore(root, "option_daily").write(raw)

    bars = pd.DataFrame({
        "timestamp": [pd.Timestamp(datetime.combine(d, datetime.min.time()), tz=NY)
                      .tz_convert("UTC") for d in sessions],
        "symbol": "RTX", "open": spot, "high": spot, "low": spot, "close": spot,
        "volume": 1e6,
    })
    (root / "underlying_bars").mkdir(parents=True)
    bars.to_parquet(root / "underlying_bars" / "RTX_1day.parquet", index=False)
    return sessions


def test_pipeline_diario_de_punta_a_punta(tmp_path) -> None:
    """Velas -> curación -> backtest, ejecutando recién en la rueda siguiente."""
    sessions = _write_synthetic_daily_lake(tmp_path)

    curated, report = run_enrich_daily(
        tmp_path, ["RTX"], pricing=PricingAssumptions(binomial_steps=64)
    )
    assert curated["underlying_price"].notna().all()
    # La primera rueda no tiene volumen anterior conocido: queda excluida.
    first_day = curated["trade_date"] == pd.Timestamp(sessions[0])
    assert not curated.loc[first_day, "is_tradable"].any()
    assert curated.loc[~first_day, "is_tradable"].mean() > 0.9
    clean = curated.loc[~curated["symbol"].str.contains("C00100000")]
    assert abs(clean["iv_american"].median() - 0.25) < 0.02

    result = run_backtest_daily(
        tmp_path, ["RTX"],
        daily_strategy_params(1, min_net_edge=5.0, stop_loss_usd=None),
        ExecutionCosts(min_net_edge=5.0),
    )
    assert result.diagnostics["signals"] > 0
    assert not result.trades.empty
    opened = pd.to_datetime(result.trades["opened_at"]).dt.tz_convert(NY).dt.date
    # La dislocación aparece al cierre de la rueda 5: nada puede abrirse antes
    # del cierre de la rueda 6.
    assert opened.min() == sessions[6]
    # La señal de la rueda 6 se ejecutaría en la 7, cuando el precio ya volvió:
    # se rechaza en vez de operar sin edge.
    assert result.diagnostics["rejected_edge_gone"] > 0
    assert (result.trades["execution_edge"] >= 5.0).all()
