"""Paridad put-call americana con dividendos discretos, fondeo y banda de spot.

Estos tests fijan las correcciones pedidas en la devolución de la entrega
parcial (punto 2): el dividendo continuo convertía la fecha ex en "arbitraje",
el préstamo de la acción sumaba al crédito de la reversa y el half-spread de la
acción tenía el signo invertido.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ita_options.arbitrage import ExecutionCosts, detect_put_call_parity
from ita_options.volatility import bsm_price

SPOT = 524.0
RATE = 0.042
SIGMA = 0.26
TAU = 10 / 365
DIVIDEND = 3.45  # LMT, ex-dividendo dentro de la vida de la opción
EX_IN = 4 / 365


def _chain(
    spot: float = SPOT,
    dividend: float = DIVIDEND,
    with_pv: bool = True,
    half_spread_pct: float = 0.01,
    uncertainty: float | None = None,
) -> pd.DataFrame:
    """Cadena sin arbitraje por construcción: BSM europeo con dividendo escrowed.

    Con el spot neto del valor presente del dividendo, los precios europeos
    cumplen :math:`C-P = S-D-Ke^{-r\\tau}`, que cae dentro de las cotas
    americanas :math:`[S-D-K,\\ S-Ke^{-r\\tau}]`. Ningún detector correcto
    debería encontrar una violación.
    """
    pv_div = dividend * math.exp(-RATE * EX_IN)
    escrowed = spot - pv_div
    rows = []
    for strike in np.arange(480.0, 571.0, 10.0):
        for flag in ("c", "p"):
            mid = bsm_price(escrowed, strike, TAU, RATE, 0.0, SIGMA, flag)
            half = max(mid * half_spread_pct, 0.05)
            rows.append({
                "underlying": "LMT", "expiration": pd.Timestamp("2026-09-11"),
                "strike": strike, "option_type": flag,
                "symbol": f"LMT_{flag}_{strike:g}",
                "bid": round(mid - half, 2), "ask": round(mid + half, 2),
                "underlying_price": spot, "tau": TAU, "risk_free_rate": RATE,
                "dividend_yield": 0.026, "is_tradable": True,
                "volume": np.nan, "open_interest": np.nan, "spread_rel": 2 * half / mid,
            })
    frame = pd.DataFrame(rows)
    if with_pv:
        frame["pv_dividends"] = pv_div
    if uncertainty is not None:
        frame["spot_uncertainty"] = uncertainty
    return frame


def test_cadena_sin_arbitraje_con_dividendo_discreto_no_dispara() -> None:
    signals = detect_put_call_parity(_chain(), ExecutionCosts(min_net_edge=5.0))
    assert signals.empty


def test_yield_continuo_ve_el_dividendo_como_arbitraje() -> None:
    """Reproduce el error de la entrega parcial: sin D discreto, la reversa dispara."""
    signals = detect_put_call_parity(_chain(with_pv=False), ExecutionCosts(min_net_edge=5.0))
    assert not signals.empty
    assert signals["legs"].str.contains("reversa").all()
    # El "edge" fantasma está acotado por el dividendo que el yield continuo omite
    # (3,45 - 0,37 USD por acción), menos spreads, r·K·tau y comisiones.
    omitted = 100 * (DIVIDEND - SPOT * (1 - math.exp(-0.026 * TAU)))
    assert 0.5 * omitted < signals["net_edge_usd"].max() < omitted


def test_el_prestamo_de_la_accion_resta_en_la_reversa() -> None:
    """Fuerza una reversa real y verifica que más borrow reduce el edge."""
    chain = _chain(dividend=0.0)
    puts = chain["option_type"] == "p"
    chain.loc[puts, ["bid", "ask"]] += 4.0  # put cara: C - P cae bajo S - D - K
    cheap = detect_put_call_parity(chain, ExecutionCosts(stock_borrow_rate=0.0, min_net_edge=0.0))
    costly = detect_put_call_parity(chain, ExecutionCosts(stock_borrow_rate=0.20, min_net_edge=0.0))
    reversas_cheap = cheap.loc[cheap["legs"].str.contains("reversa")].set_index("strikes")
    reversas_costly = costly.loc[costly["legs"].str.contains("reversa")].set_index("strikes")
    common = reversas_cheap.index.intersection(reversas_costly.index)
    assert len(common) > 0
    expected = 100 * (SPOT - 0.005) * 0.20 * TAU
    diff = reversas_cheap.loc[common, "net_edge_usd"] - reversas_costly.loc[common, "net_edge_usd"]
    assert np.allclose(diff, expected, rtol=1e-6)


def test_la_accion_se_compra_al_ask_y_se_vende_al_bid() -> None:
    """El precio de la pata de acción coincide con el lado que usa el edge."""
    chain = _chain(dividend=0.0)
    calls = chain["option_type"] == "c"
    chain.loc[calls, ["bid", "ask"]] += 4.0  # call cara: conversión
    signals = detect_put_call_parity(chain, ExecutionCosts(min_net_edge=0.0))
    conv = signals.loc[signals["legs"].str.startswith("conversión:")]
    assert not conv.empty
    for legs in conv["leg_spec"]:
        stock = [leg for leg in legs if leg["kind"] == "stock"][0]
        assert stock["qty"] > 0 and stock["price"] == pytest.approx(SPOT + 0.005)


def test_banda_de_spot_absorbe_un_desfase_horario() -> None:
    """Un spot de referencia desfasado 1% genera falsas conversiones; la banda las anula."""
    stale = _chain(spot=SPOT * 0.99)  # opciones valuadas con SPOT, acción informada 1% abajo
    stale["pv_dividends"] = DIVIDEND * math.exp(-RATE * EX_IN)
    # Las opciones se valuaron con SPOT pero la columna dice SPOT*0.99: reconstruir.
    fresh = _chain()
    stale[["bid", "ask"]] = fresh[["bid", "ask"]].to_numpy()
    assert not detect_put_call_parity(stale, ExecutionCosts(min_net_edge=5.0)).empty
    stale["spot_uncertainty"] = SPOT * 0.01
    assert detect_put_call_parity(stale, ExecutionCosts(min_net_edge=5.0)).empty


def test_fondeo_hace_mas_exigente_la_conversion() -> None:
    chain = _chain(dividend=0.0)
    calls = chain["option_type"] == "c"
    chain.loc[calls, ["bid", "ask"]] += 1.0
    loose = detect_put_call_parity(chain, ExecutionCosts(funding_spread=0.0, min_net_edge=0.0))
    tight = detect_put_call_parity(chain, ExecutionCosts(funding_spread=0.05, min_net_edge=0.0))
    assert loose["net_edge_usd"].sum() > tight["net_edge_usd"].sum()
