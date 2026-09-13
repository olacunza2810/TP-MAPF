"""Fixtures compartidas: cadenas sintéticas libres de arbitraje por construcción.

La estrategia de testing es generar los precios con el mismo modelo que después
se usa para detectar. Si los detectores encuentran algo sobre una cadena
generada por el modelo, el falso positivo es del detector, no del mercado.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from ita_options.volatility import crr_american_price

SPOT = 150.0
RATE = 0.0425
DIVIDEND = 0.0205
SIGMA = 0.28
STRIKES = np.arange(135.0, 166.0, 5.0)


def build_chain(
    expiries: tuple[tuple[str, float], ...] = (("2026-12-18", 0.25),),
    spread: float = 0.10,
    timestamp: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Construye una cadena consistente con el modelo, sin arbitraje.

    Args:
        expiries: Pares ``(fecha, tau)``.
        spread: Spread absoluto uniforme en USD.
        timestamp: Instante del corte.

    Returns:
        Cadena en el esquema curado.
    """
    stamp = timestamp or pd.Timestamp("2026-09-03 15:00", tz="UTC")
    rows = []
    for expiry, tau in expiries:
        for strike in STRIKES:
            for flag in ("c", "p"):
                fair = crr_american_price(
                    SPOT, strike, tau, RATE, DIVIDEND, SIGMA, flag, 256
                )
                bid = round(max(fair - spread / 2, 0.01), 2)
                ask = round(fair + spread / 2, 2)
                rows.append(
                    {
                        "symbol": f"RTX{expiry}{flag}{strike:.0f}",
                        "underlying": "RTX",
                        "timestamp": stamp,
                        "expiration": pd.Timestamp(expiry),
                        "strike": strike,
                        "option_type": flag,
                        "bid": bid,
                        "ask": ask,
                        "bid_size": 50.0,
                        "ask_size": 50.0,
                        "mid_price": (bid + ask) / 2,
                        "volume": 500.0,
                        "open_interest": 1000.0,
                        "spread_rel": spread / max(fair, 0.01),
                        "underlying_price": SPOT,
                        "tau": tau,
                        "dte": int(tau * 365),
                        "risk_free_rate": RATE,
                        "dividend_yield": DIVIDEND,
                        "vega": 10.0,
                        "is_tradable": True,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def clean_chain() -> pd.DataFrame:
    """Cadena de un vencimiento, sin arbitraje."""
    return build_chain()


@pytest.fixture
def multi_expiry_chain() -> pd.DataFrame:
    """Cadena de dos vencimientos, sin arbitraje."""
    return build_chain((("2026-12-18", 0.25), ("2027-03-19", 0.50)))


@pytest.fixture
def timeseries_chain() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Serie de 12 cortes con una violación de convexidad en los cortes 2 y 3.

    Returns:
        Tupla ``(quotes, barras del subyacente)``.
    """
    start = pd.Timestamp("2026-09-03 14:00", tz="UTC")
    stamps = [start + timedelta(minutes=5 * i) for i in range(12)]
    frames = []
    for index, stamp in enumerate(stamps):
        tau = (pd.Timestamp("2026-10-16") - stamp.tz_localize(None)).days / 365
        frame = build_chain((("2026-10-16", tau),), timestamp=stamp)
        if index in (2, 3):
            mask = (frame["option_type"] == "c") & (frame["strike"] == 150.0)
            frame.loc[mask, ["bid", "ask"]] += 3.0
        frames.append(frame)
    quotes = pd.concat(frames, ignore_index=True)
    underlying = pd.DataFrame(
        {"timestamp": stamps, "symbol": "RTX", "close": SPOT}
    )
    return quotes, underlying
