"""Tests de contrato contra el SDK real de Alpaca.

No hacen llamadas de red: construyen los objetos que devuelve la API y
verifican que nuestros adaptadores leen los campos correctos. Es la clase de
test que hubiera evitado el bug de ``multiplier`` — un nombre de campo
inventado sobrevive a cualquier test que use datos simulados por nosotros
mismos, y sólo falla contra el SDK.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from uuid import uuid4

import pytest

alpaca = pytest.importorskip("alpaca", reason="alpaca-py no instalado")

from alpaca.data.models.snapshots import OptionsSnapshot
from alpaca.data.requests import (
    OptionBarsRequest,
    OptionSnapshotRequest,
    StockBarsRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.enums import AssetStatus, ContractType, ExerciseStyle
from alpaca.trading.models import OptionContract
from alpaca.trading.requests import GetOptionContractsRequest

from ita_options.config import AlpacaCredentials, PipelineConfig
from ita_options.ingest import SnapshotRecorder, UniverseBuilder


def test_los_requests_se_construyen_con_nuestros_parametros() -> None:
    """Cada request que arma el gateway debe validar contra el SDK."""
    OptionSnapshotRequest(symbol_or_symbols=["RTX261218C00150000"], feed="indicative")
    StockBarsRequest(
        symbol_or_symbols=["RTX"],
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=date.today() - timedelta(days=5),
        end=date.today(),
        adjustment="raw",
    )
    OptionBarsRequest(
        symbol_or_symbols=["RTX261218C00150000"],
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=date.today() - timedelta(days=30),
        end=date.today(),
    )
    GetOptionContractsRequest(
        underlying_symbols=["RTX"],
        expiration_date_gte=date.today(),
        expiration_date_lte=date.today() + timedelta(days=180),
        strike_price_gte="112.5",
        strike_price_lte="187.5",
        limit=10_000,
        page_token=None,
    )


def test_no_existe_endpoint_de_quotes_historicos() -> None:
    """Documenta la limitación que define la arquitectura del pipeline.

    Si algún día Alpaca agrega quotes históricos de opciones, este test falla y
    obliga a revisar el modo ``backfill``, que existe sólo por esta ausencia.
    """
    from alpaca.data.historical.option import OptionHistoricalDataClient

    getters = {m for m in dir(OptionHistoricalDataClient) if m.startswith("get_")}
    assert "get_option_latest_quote" in getters
    assert "get_option_quotes" not in getters


def test_universe_builder_lee_los_campos_reales_del_contrato() -> None:
    """``OptionContract`` expone el multiplicador como ``size``, no ``multiplier``."""
    contract = OptionContract(
        id=str(uuid4()),
        symbol="RTX261218C00150000",
        name="RTX Dec 18 2026 150 Call",
        status=AssetStatus.ACTIVE,
        tradable=True,
        expiration_date=date(2026, 12, 18),
        root_symbol="RTX",
        underlying_symbol="RTX",
        underlying_asset_id=str(uuid4()),
        type=ContractType.CALL,
        style=ExerciseStyle.AMERICAN,
        strike_price="150",
        size="100",
        open_interest="1234",
        open_interest_date=date(2026, 9, 4),
        close_price="7.45",
        close_price_date=date(2026, 9, 4),
    )
    frame = UniverseBuilder._to_frame([contract])
    row = frame.loc[0]
    assert row["multiplier"] == 100.0
    assert row["strike"] == 150.0
    assert row["option_type"] == "c"
    assert row["style"] == "american"
    assert row["open_interest"] == 1234.0


def test_recorder_lee_los_campos_reales_del_snapshot() -> None:
    """El recorder debe extraer NBBO y tamaños, y no confundir volumen."""
    snapshot = OptionsSnapshot(
        symbol="RTX261218C00150000",
        raw_data={
            "latestQuote": {
                "t": "2026-09-04T19:59:00Z",
                "bp": 7.40,
                "ap": 7.50,
                "bs": 12,
                "as": 30,
                "bx": "C",
                "ax": "C",
                "c": ["R"],
            },
            "latestTrade": {
                "t": "2026-09-04T19:58:30Z",
                "p": 7.45,
                "s": 3,
                "x": "C",
                "c": ["I"],
            },
            "impliedVolatility": 0.2814,
            "greeks": {"delta": 0.5412, "gamma": 0.0182, "theta": -0.0431,
                       "vega": 0.2913, "rho": 0.1104},
        },
    )
    config = PipelineConfig(
        credentials=AlpacaCredentials(api_key="x", secret_key="y")
    )
    recorder = SnapshotRecorder.__new__(SnapshotRecorder)
    recorder._config = config

    async def fake_fetch(symbols):
        return {snapshot.symbol: snapshot}

    recorder._gateway = type(
        "Gateway", (), {"fetch_option_snapshots": staticmethod(fake_fetch)}
    )()

    frame = asyncio.run(recorder.capture(["RTX261218C00150000"]))
    row = frame.loc[0]
    assert row["bid"] == 7.40 and row["ask"] == 7.50
    assert row["bid_size"] == 12 and row["ask_size"] == 30
    assert row["last_trade_size"] == 3
    # El snapshot no trae volumen diario: debe quedar nulo, no confundirse con
    # el tamaño del último trade.
    assert frame["volume"].isna().all()
