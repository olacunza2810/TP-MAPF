"""Comparación Black-Scholes contra binomial (Parte 2, paso 3)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ita_options.greeks import bs_greeks, crr_price
from ita_options.model_comparison import (
    add_buckets, diebold_mariano, dm_table, implied_vol, model_price, value_slice,
)

S, K, T, R = 100.0, 100.0, 0.25, 0.04


def test_la_prima_de_ejercicio_es_no_negativa_y_nula_en_calls_sin_dividendo() -> None:
    for k in (80, 100, 120):
        am = crr_price(S, k, T, R, 0.3, "c", steps=400)
        eu = crr_price(S, k, T, R, 0.3, "c", steps=400, american=False)
        assert am == pytest.approx(eu, abs=1e-10)
        assert crr_price(S, k, T, R, 0.3, "p", steps=400) >= crr_price(S, k, T, R, 0.3, "p", steps=400, american=False)


def test_el_call_muy_itm_antes_de_la_fecha_ex_se_ejerce() -> None:
    # Dividendo grande en 5 días y poco valor tiempo: el americano vale el intrínseco con dividendo.
    divs = [(5 / 365, 3.0)]
    am = crr_price(S, 70, 30 / 365, R, 0.2, "c", divs, steps=300)
    eu = bs_greeks(S, 70, 30 / 365, R, 0.2, "c", divs).price
    assert am > eu + 2.5
    assert am >= S - 70 - 1e-9


def test_el_arbol_europeo_converge_a_black_scholes_con_dividendos() -> None:
    divs = [(0.1, 1.5)]
    eu = crr_price(S, 105, T, R, 0.3, "p", divs, steps=2000, american=False)
    assert eu == pytest.approx(bs_greeks(S, 105, T, R, 0.3, "p", divs).price, abs=5e-3)


@pytest.mark.parametrize("model", ["crr", "bsm", "crr_q"])
def test_la_iv_recupera_la_volatilidad(model: str) -> None:
    divs = [(0.05, 1.0)]
    price = model_price(model, S, 95, T, R, 0.27, "p", divs)
    assert implied_vol(model, price, S, 95, T, R, "p", divs) == pytest.approx(0.27, abs=1e-4)
    # Por debajo del intrínseco (put K=120 a USD 0,50): no hay volatilidad que lo dé.
    assert np.isnan(implied_vol(model, 0.5, S, 120, T, R, "p", divs))


def test_diebold_mariano_detecta_un_diferencial_persistente_y_no_uno_nulo() -> None:
    rng = np.random.default_rng(0)
    assert diebold_mariano(pd.Series(0.5 + rng.normal(0, 1, 200)))["p_valor"] < 0.01
    assert diebold_mariano(pd.Series(rng.normal(0, 1, 200)))["p_valor"] > 0.05
    assert np.isnan(diebold_mariano(pd.Series([1.0, 2.0]))["dm"])


def _chain(spot: float = 100.0) -> pd.DataFrame:
    rows = []
    divs = [(10 / 365, 1.0)]
    for strike in np.arange(80, 121, 5.0):
        for flag in ("c", "p"):
            fair = crr_price(spot, strike, 40 / 365, R, 0.3, flag, divs)
            rows.append({"trade_date": date(2026, 9, 1), "underlying": "LMT", "expiration": date(2026, 10, 11),
                         "option_type": flag, "strike": strike, "dte": 40, "tau": 40 / 365,
                         "underlying_price": spot, "risk_free_rate": R,
                         "bid": round(fair - 0.05, 2), "ask": round(fair + 0.05, 2)})
    return pd.DataFrame(rows), divs


def test_sobre_una_cadena_del_binomial_el_binomial_reprecia_dentro_del_spread() -> None:
    chain, divs = _chain()
    out = add_buckets(value_slice(chain, divs))
    assert len(out) == len(chain)
    assert out["dentro_crr"].mean() > 0.95
    assert (out["eep"] >= -1e-9).all()
    itm_calls = out.loc[(out["option_type"] == "c") & (out["dinero"] == "ITM profundo")]
    # Calls muy ITM con la fecha ex a 10 días: el europeo queda por debajo del mercado.
    assert (itm_calls["err_bsm"] < itm_calls["err_crr"]).all()
    assert set(out["fecha_ex"]) == {"ex en 8-30 días"}
    table = dm_table(out, by=["option_type"])
    assert set(table["option_type"]) == {"c", "p"}


def test_sin_dividendos_los_dos_arboles_coinciden() -> None:
    for flag in ("c", "p"):
        assert model_price("crr", S, 95, T, R, 0.3, flag, []) == pytest.approx(
            model_price("crr_q", S, 95, T, R, 0.3, flag, []), abs=1e-9)
