"""Monte Carlo de estrés (Parte 2, paso 5)."""

from __future__ import annotations

import numpy as np
import pytest

from ita_options.execution.model_rv import Smile
from ita_options.greeks import bs_greeks, crr_greeks
from ita_options.stress import (
    Book, HedgeOption, Leg, RiskRules, ScenarioPaths, Structure, risk_metrics, run_account,
    simulate_gbm, simulate_heston, simulate_merton, vec_bs, vec_crr,
)

DIVS = ((3.5 / 12, 2.06),)
ONE = np.array([[1.0]])


def test_el_arbol_vectorizado_coincide_con_el_escalar() -> None:
    spots, sigmas = np.array([52.0, 60.0, 45.0]), np.array([0.4, 0.3, 0.5])
    price, delta, gamma = vec_crr(spots, 50, 5 / 12, 0.10, sigmas, "p", DIVS)
    for i in range(3):
        g = crr_greeks(spots[i], 50, 5 / 12, 0.10, sigmas[i], "p", DIVS, steps=100)
        assert (price[i], delta[i], gamma[i]) == pytest.approx((g.price, g.delta, g.gamma), abs=1e-10)
    price, delta, gamma = vec_bs(spots, 50, 5 / 12, 0.10, sigmas, "c", DIVS)
    for i in range(3):
        g = bs_greeks(spots[i], 50, 5 / 12, 0.10, sigmas[i], "c", DIVS)
        assert (price[i], delta[i], gamma[i]) == pytest.approx((g.price, g.delta, g.gamma), abs=1e-10)


def test_gbm_tiene_la_volatilidad_pedida_y_la_correlacion() -> None:
    corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    p = simulate_gbm(np.array([100.0, 50.0]), np.array([0.3, 0.2]), corr, 0.0, 20, 20000, np.random.default_rng(1))
    r = np.log(p.spot[:, -1] / p.spot[:, 0])
    assert r.std(axis=0) == pytest.approx(np.array([0.3, 0.2]) * np.sqrt(20 / 252), rel=0.03)
    assert np.corrcoef(r.T)[0, 1] == pytest.approx(0.6, abs=0.03)


def test_en_heston_la_varianza_estocastica_llega_al_precio() -> None:
    """El error del notebook (``Var[-1]``) dejaba al precio con volatilidad constante."""
    rng = np.random.default_rng(2)
    p = simulate_heston(np.array([100.0]), np.array([0.09]), np.array([2.0]), np.array([0.09]), np.array([1.5]),
                        np.array([0.0]), ONE, 0.0, 30, 5000, rng, substeps=8)
    returns = np.diff(np.log(p.spot[:, :, 0]), axis=1)
    realized = (returns ** 2).sum(axis=1)
    mean_var = (p.atm_vol[:, :-1, 0] ** 2).mean(axis=1)
    assert np.corrcoef(realized, mean_var)[0, 1] > 0.5
    # Con xi = 0 y v0 = theta es un GBM.
    flat = simulate_heston(np.array([100.0]), np.array([0.09]), np.array([2.0]), np.array([0.09]), np.array([0.0]),
                           np.array([0.0]), ONE, 0.0, 20, 20000, np.random.default_rng(3))
    r = np.log(flat.spot[:, -1, 0] / 100.0)
    assert r.std() == pytest.approx(0.3 * np.sqrt(20 / 252), rel=0.03)


def test_merton_es_martingala_descontada() -> None:
    rng = np.random.default_rng(4)
    for common in (False, True):
        p = simulate_merton(np.array([100.0, 80.0]), np.array([0.2, 0.2]), np.array([0.3, 0.3]),
                            np.array([4.0, 4.0]), np.array([-0.08, -0.08]), np.array([0.06, 0.06]),
                            np.array([[1.0, 0.5], [0.5, 1.0]]), 0.04, 60, 40000, rng, common=common)
        expected = np.array([100.0, 80.0]) * np.exp(0.04 * 60 / 252)
        assert p.spot[:, -1].mean(axis=0) == pytest.approx(expected, rel=0.01)


def test_la_fecha_ex_baja_el_precio_y_paga_a_la_accion() -> None:
    p = simulate_gbm(np.array([100.0]), np.array([1e-6]), ONE, 0.0, 5, 3, np.random.default_rng(0),
                     ex_days=[[(2, 1.5)]])
    assert p.spot[0, :, 0] == pytest.approx([100, 100, 98.5, 98.5, 98.5, 98.5], abs=1e-3)
    assert p.dividend_paid[2, 0] == 1.5


def _book(model: str, units: float = -1.0) -> Book:
    leg = Leg(u=0, strike=100.0, flag="c", tau0=40 / 365, units=units, sigma={"crr": 0.3, "bsm": 0.3},
              resid=0.0, half=0.05)
    price = float(vec_crr(np.array([100.0]), 100.0, 40 / 365, 0.04, np.array([0.3]), "c")[0][0])
    st = Structure(model, "call vendida", [leg], entry_cash=-units * 100 * price, predicted_edge=10.0)
    hedge = Leg(u=0, strike=100.0, flag="c", tau0=20 / 365, units=0, sigma={"crr": 0.3, "bsm": 0.3}, resid=0.0,
                half=0.05)
    flat = Smile(0.3, 0.0, 0.0, 100.0, 0, 0.0)
    return Book(model, [st], {0: HedgeOption(hedge, {"crr": flat, "bsm": flat})})


def test_la_cobertura_delta_baja_el_riesgo_de_la_cartera() -> None:
    paths = simulate_gbm(np.array([100.0]), np.array([0.3]), ONE, 0.04, 10, 3000, np.random.default_rng(5))
    cal = np.array([i * 7 / 5 / 365 for i in range(11)])
    rules = RiskRules(kill_switch_usd=1e9, daily_loss_usd=1e9, take_profit_fraction=1e9)
    cache: dict = {}
    books = [_book("crr")]
    res = {h: run_account(books, paths, 0.04, cal, h, rules, steps=60, cache=cache)
           for h in ("sin_cobertura", "delta", "delta_gamma")}
    sd = {h: r["pnl"].std() for h, r in res.items()}
    assert sd["delta"] < 0.5 * sd["sin_cobertura"]
    assert sd["delta_gamma"] < sd["delta"]
    assert res["delta"]["costo_cobertura"].mean() > 0
    assert res["sin_cobertura"]["costo_cobertura"].sum() == 0


def test_la_perdida_diaria_corta_la_cuenta_y_congela_el_pnl() -> None:
    spot = np.array([100.0, 100.0, 130.0, 160.0, 190.0, 220.0])[None, :, None]  # suba fuerte: call vendida pierde
    paths = ScenarioPaths(spot, np.full_like(spot, 0.3), np.zeros((6, 1)))
    cal = np.array([i * 7 / 5 / 365 for i in range(6)])
    rules = RiskRules(kill_switch_usd=1e9, daily_loss_usd=1000.0, take_profit_fraction=1e9)
    books = [_book("bsm", units=-1.0)]
    r = run_account(books, paths, 0.0, cal, "sin_cobertura", rules, steps=60)
    assert r["perdida_diaria"][0]
    # Cortada el día 2, el P&L final es el de ese día (la suba posterior no suma pérdidas).
    rules_free = RiskRules(kill_switch_usd=1e9, daily_loss_usd=1e9, take_profit_fraction=1e9)
    free = run_account(books, paths, 0.0, cal, "sin_cobertura", rules_free, steps=60)
    assert r["pnl"][0] > free["pnl"][0] + 5000


def test_metricas_de_riesgo() -> None:
    pnl = np.arange(-50, 50, dtype=float)  # 100 trayectorias
    m = risk_metrics(pnl)
    assert m["var99"] == pytest.approx(49.01, abs=0.01) and m["es99"] == pytest.approx(50.0)
    assert m["peor"] == 50.0


def test_con_spreads_anchos_el_kill_switch_a_liquidacion_salta_sin_que_el_mercado_se_mueva() -> None:
    spot = np.full((1, 4, 1), 100.0)
    paths = ScenarioPaths(spot, np.full_like(spot, 0.3), np.zeros((4, 1)))
    cal = np.array([i / 365 for i in range(4)])
    book = _book("crr", units=-1.0)
    book.structures[0].legs[0].half = 3.0  # medio spread de 3 USD: 300 USD a liquidación
    flat = dict(daily_loss_usd=1e9, take_profit_fraction=1e9)
    liq = run_account([book], paths, 0.0, cal, "sin_cobertura", RiskRules(mark="liquidacion", **flat), steps=60)
    mid = run_account([book], paths, 0.0, cal, "sin_cobertura", RiskRules(mark="mid", **flat), steps=60)
    assert liq["kill"][0] and not mid["kill"][0]


def test_la_opcion_de_gamma_rota_cuando_se_aleja_del_dinero() -> None:
    """Con una suba de 20% la call 100 queda lejos: se vende y se compra la ATM nueva."""
    spot = np.array([100.0, 100.0, 120.0, 120.0])[None, :, None]
    paths = ScenarioPaths(spot, np.full_like(spot, 0.3), np.zeros((4, 1)))
    cal = np.array([i / 365 for i in range(4)])
    rules = RiskRules(kill_switch_usd=1e9, daily_loss_usd=1e9, take_profit_fraction=1e9)
    book = _book("crr", units=-3.0)
    still = run_account([book], ScenarioPaths(np.full_like(spot, 100.0), np.full_like(spot, 0.3),
                                              np.zeros((4, 1))), 0.0, cal, "delta_gamma", rules, steps=60)
    moved = run_account([book], paths, 0.0, cal, "delta_gamma", rules, steps=60)
    # Rotar cuesta dos medios spreads más por contrato que quedarse quieto.
    assert moved["costo_cobertura"][0] > still["costo_cobertura"][0] + 2 * (100 * 0.05 + 0.65)


def test_el_strike_vectorial_coincide_con_el_escalar() -> None:
    spots = np.array([100.0, 100.0])
    a, _, _ = vec_crr(spots, np.array([95.0, 105.0]), 0.2, 0.03, np.array([0.3, 0.3]), "c")
    assert a[0] == pytest.approx(vec_crr(spots[:1], 95.0, 0.2, 0.03, np.array([0.3]), "c")[0][0])
    assert a[1] == pytest.approx(vec_crr(spots[:1], 105.0, 0.2, 0.03, np.array([0.3]), "c")[0][0])
