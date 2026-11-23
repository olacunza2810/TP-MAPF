"""Black-Scholes y binomial con dividendos discretos y sus griegas (Parte 2).

Validan el módulo contra la teoría (paridad, convergencia, derivadas numéricas)
y contra los números del material de clase.
"""

from __future__ import annotations

import math

import pytest

from ita_options.greeks import bs_greeks, crr_greeks, pv_dividends
from ita_options.volatility import crr_american_price

S, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.20


def test_reproduce_el_call_del_notebook_de_clase() -> None:
    # Black-Scholes.ipynb: BlackScholes(St=100, K=100, r=0.01, sigma=0.2, tau=1).call()
    assert bs_greeks(100, 100, 1, 0.01, 0.2, "c").price == pytest.approx(8.433318690109608, abs=1e-12)


def test_paridad_put_call_europea_con_dividendo_discreto() -> None:
    divs = [(0.25, 1.5), (0.75, 1.5)]
    c = bs_greeks(S, K, T, R, SIG, "c", divs).price
    p = bs_greeks(S, K, T, R, SIG, "p", divs).price
    D = pv_dividends(divs, R, T)
    assert c - p == pytest.approx(S - D - K * math.exp(-R * T), abs=1e-10)


@pytest.mark.parametrize("divs", [(), ((0.25, 1.5), (0.75, 1.5))])
@pytest.mark.parametrize("flag", ["c", "p"])
def test_griegas_bs_coinciden_con_derivadas_numericas(flag: str, divs) -> None:
    """Con dividendos, rho y theta incluyen cómo cambia D con r y con el paso del tiempo."""
    g = bs_greeks(S, K, T, R, SIG, flag, divs)
    h = 1e-3

    def v(**kw) -> float:
        tau = kw.pop("tau", T)
        # Pasar el tiempo acerca las fechas ex: se corren junto con tau.
        shifted = tuple((ti - (T - tau), d) for ti, d in divs)
        args = dict(spot=S, strike=K, tau=tau, rate=R, sigma=SIG, flag=flag, dividends=shifted)
        args.update(kw)
        return bs_greeks(**args).price

    assert g.delta == pytest.approx((v(spot=S + h) - v(spot=S - h)) / (2 * h), abs=1e-6)
    assert g.gamma == pytest.approx((v(spot=S + h) - 2 * g.price + v(spot=S - h)) / h**2, abs=1e-4)
    # Tolerancia relativa: el error de la diferencia central es de segundo orden.
    assert g.vega == pytest.approx((v(sigma=SIG + h) - v(sigma=SIG - h)) / (2 * h), rel=1e-5)
    assert g.rho == pytest.approx((v(rate=R + h) - v(rate=R - h)) / (2 * h), rel=1e-5)
    # Theta es la derivada respecto del paso del tiempo: -dV/dtau.
    assert g.theta == pytest.approx(-(v(tau=T + h) - v(tau=T - h)) / (2 * h), abs=1e-4)


@pytest.mark.parametrize("flag", ["c", "p"])
def test_arbol_europeo_converge_a_black_scholes(flag: str) -> None:
    bs = bs_greeks(S, K, T, R, SIG, flag, [(0.5, 2.0)])
    tree = crr_greeks(S, K, T, R, SIG, flag, [(0.5, 2.0)], steps=2000, american=False)
    assert tree.price == pytest.approx(bs.price, abs=0.01)
    assert tree.delta == pytest.approx(bs.delta, abs=2e-3)
    assert tree.gamma == pytest.approx(bs.gamma, rel=0.02)
    assert tree.vega == pytest.approx(bs.vega, rel=0.01)
    assert tree.theta == pytest.approx(bs.theta, rel=0.02)
    assert tree.rho == pytest.approx(bs.rho, rel=0.01)


def test_convergencia_del_ejemplo_1_de_clase() -> None:
    # codigo.ipynb, Ejemplo #1: call S=52, K=50, r=12%, sigma=30%, tau=3 meses.
    bs = bs_greeks(52, 50, 3 / 12, 0.12, 0.3, "c").price
    errors = [abs(crr_greeks(52, 50, 3 / 12, 0.12, 0.3, "c", steps=n, american=False).price - bs)
              for n in (10, 100, 1000)]
    assert errors[0] > errors[1] > errors[2] and errors[2] < 2e-3


def test_sin_dividendos_coincide_con_el_arbol_del_pipeline() -> None:
    mine = crr_greeks(S, 95, T, R, SIG, "p", steps=256).price
    pipeline = crr_american_price(S, 95, T, R, 0.0, SIG, "p", steps=256)
    assert mine == pytest.approx(pipeline, abs=1e-10)


def test_call_americano_sin_dividendos_no_se_ejerce_antes() -> None:
    american = crr_greeks(S, K, T, R, SIG, "c", steps=500).price
    european = crr_greeks(S, K, T, R, SIG, "c", steps=500, american=False).price
    assert american == pytest.approx(european, abs=1e-10)


def test_put_americana_vale_mas_que_la_europea() -> None:
    american = crr_greeks(S, 110, T, R, SIG, "p", steps=500).price
    european = crr_greeks(S, 110, T, R, SIG, "p", steps=500, american=False).price
    assert american > european + 0.1


def test_call_en_el_dinero_antes_de_la_fecha_ex_tiene_prima_de_ejercicio() -> None:
    """El caso donde el binomial debería ganarle a BS: dividendo grande a punto de caer."""
    divs = [(5 / 365, 3.45)]  # LMT: USD 3,45 en cinco días
    american = crr_greeks(560, 500, 30 / 365, 0.04, 0.25, "c", divs, steps=600).price
    european = bs_greeks(560, 500, 30 / 365, 0.04, 0.25, "c", divs).price
    assert american - european > 1.0
