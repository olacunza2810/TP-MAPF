"""Coberturas con griegas (Parte 2, paso 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ita_options.greeks import bs_greeks
from ita_options.hedging import (
    HedgeCosts, PositionPath, effectiveness, effectiveness_table, hedge_quantities, pick_instrument,
    simulate_position,
)

R, SIG = 0.04, 0.30


def test_delta_gamma_anula_las_dos_griegas() -> None:
    opt = bs_greeks(100, 105, 0.15, R, SIG, "c")
    hed = bs_greeks(100, 100, 0.05, R, SIG, "c")
    n, h = hedge_quantities("delta_gamma", -1.0, opt, hed)
    assert -1.0 * opt.gamma + n * hed.gamma == pytest.approx(0.0, abs=1e-12)
    assert 100 * (-1.0 * opt.delta + n * hed.delta) + h == pytest.approx(0.0, abs=1e-9)
    n_v, _ = hedge_quantities("delta_vega", -1.0, opt, hed)
    assert -opt.vega + n_v * hed.vega == pytest.approx(0.0, abs=1e-12)
    assert hedge_quantities("delta", -1.0, opt, None) == (0.0, pytest.approx(100 * opt.delta))
    with pytest.raises(ValueError):
        hedge_quantities("delta_gamma", -1.0, opt, None)


def test_la_opcion_de_cobertura_excluye_el_vencimiento_propio() -> None:
    cand = pd.DataFrame({"expiration": ["a", "b", "c"], "dte": [10, 30, 90], "symbol": ["A", "B", "C"]})
    assert pick_instrument(cand, "b", "gamma")["symbol"] == "C"  # A queda por debajo del plazo mínimo
    assert pick_instrument(cand, "c", "vega")["symbol"] == "B"
    assert pick_instrument(cand, "c", "gamma", min_dte=60) is None


def _bs_path(seed: int, days: int = 10) -> PositionPath:
    """Camino GBM con la opción y las coberturas valuadas por Black-Scholes, sin spread."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    spots = [100.0]
    for _ in range(days):
        spots.append(spots[-1] * np.exp((R - 0.5 * SIG ** 2) * dt + SIG * np.sqrt(dt) * rng.normal()))
    dates = list(pd.bdate_range("2026-09-01", periods=days + 1))
    taus = [0.20 - i * dt for i in range(days + 1)]
    g = [bs_greeks(s, 100, t, R, SIG, "c") for s, t in zip(spots, taus)]
    hedges = {}
    for kind, (k, t_h) in {"delta_gamma": (100, 0.08), "delta_vega": (100, 0.40)}.items():
        steps = []
        for i in range(days):
            now = bs_greeks(spots[i], k, t_h - i * dt, R, SIG, "c")
            nxt = bs_greeks(spots[i + 1], k, t_h - (i + 1) * dt, R, SIG, "c")
            steps.append({"symbol": kind, "mid": now.price, "half": 0.0, "greeks": {"bsm": now},
                          "mid_next": nxt.price, "half_next": 0.0})
        hedges[kind] = steps
    return PositionPath(dates, [x.price for x in g], [0.0] * (days + 1), spots, [R] * (days + 1),
                        [0.0] * (days + 1), {"bsm": g}, hedges)


def test_en_un_mundo_black_scholes_la_cobertura_reduce_la_varianza() -> None:
    frames = []
    for seed in range(60):
        path = _bs_path(seed)
        for strat in ("sin_cobertura", "delta", "delta_gamma", "delta_vega"):
            frames.append(simulate_position(path, "bsm", strat, costs=HedgeCosts(0.0, 0.0))
                          .assign(modelo="bsm", cobertura=strat, posicion=seed))
    days = pd.concat(frames, ignore_index=True)
    table = effectiveness_table(days).set_index("cobertura")
    assert table.loc["delta", "efectividad"] > 0.85
    # Con la gamma cubierta el error de rebalanceo discreto (1/2 Gamma dS^2) desaparece casi entero.
    assert table.loc["delta_gamma", "efectividad"] > table.loc["delta", "efectividad"]
    assert table.loc["delta_gamma", "efectividad"] > 0.97
    assert (table["costo_medio"] == 0).all()


def test_los_costos_incluyen_rebalanceo_cambio_de_simbolo_y_desarme() -> None:
    path = _bs_path(1, days=3)
    for i, step in enumerate(path.hedges["delta_gamma"]):
        step["half"] = step["half_next"] = 0.05
        step["symbol"] = "H1" if i < 2 else "H2"
    out = simulate_position(path, "bsm", "delta_gamma", costs=HedgeCosts(0.65, 1.0))
    n, h = out["n"].to_numpy(), out["h"].to_numpy()
    per = 100 * 0.05 + 0.65
    stock = [abs(h[0]) * path.spot[0], abs(h[1] - h[0]) * path.spot[1], abs(h[2] - h[1]) * path.spot[2]]
    expected = [stock[0] * 1e-4 + abs(n[0]) * per,
                stock[1] * 1e-4 + abs(n[1] - n[0]) * per,
                stock[2] * 1e-4 + abs(n[2]) * per + abs(n[1]) * per
                + abs(h[2]) * path.spot[3] * 1e-4 + abs(n[2]) * per]
    assert out["costo"].to_numpy() == pytest.approx(expected)


def test_efectividad() -> None:
    assert effectiveness([1, -1, 1, -1], [2, -2, 2, -2]) == pytest.approx(0.75)


def test_la_version_entera_redondea_y_solo_compra() -> None:
    opt = bs_greeks(100, 100, 0.15, R, SIG, "c")
    hed = bs_greeks(100, 100, 0.05, R, SIG, "c")
    n_short, _ = hedge_quantities("delta_gamma_entera", -1.0, opt, hed)   # vendido: gamma negativa
    n_cont, _ = hedge_quantities("delta_gamma", -1.0, opt, hed)
    assert n_short == float(max(round(n_cont), 0)) and n_short >= 0
    n_long, h_long = hedge_quantities("delta_gamma_entera", +1.0, opt, hed)  # comprado: habría que vender
    assert n_long == 0.0 and h_long == pytest.approx(-100 * opt.delta)
