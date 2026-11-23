"""Tests de la simulación histórica del sistema de producción."""

from datetime import date

import pandas as pd
import pytest

from ita_options.replay import ReplayRules, replay, trade_stats

D = [date(2026, 3, 2), date(2026, 3, 4), date(2026, 3, 6), date(2026, 3, 9)]
EXP = date(2026, 4, 17)


def _chain(day, a, b):
    """Dos calls (vertical 100/105) con bid/ask dados por pata."""
    rows = []
    for sym, k, (bid, ask) in (("C100", 100.0, a), ("C105", 105.0, b)):
        rows.append({"symbol": sym, "underlying": "RTX", "strike": k, "option_type": "c", "expiration": EXP,
                     "bid": bid, "ask": ask, "risk_free_rate": 0.0, "trade_date": day})
    return pd.DataFrame(rows)


def _signal(limit=2.0, edge=20.0):
    legs = ({"symbol": "C100", "qty": 1.0, "strike": 100.0, "option_type": "c"},
            {"symbol": "C105", "qty": -1.0, "strike": 105.0, "option_type": "c"})
    return pd.DataFrame([{"detector": "rv_crr_vertical", "underlying": "RTX", "expiration": EXP,
                          "leg_spec": legs, "limit_net": limit, "net_edge_usd": edge, "commission": 1.3}])


def _run(chains, signals, **rules):
    spots = {("RTX", d): 102.0 for d in D}
    divs = pd.DataFrame({"ticker": [], "ex_dividend_date": [], "cash_amount": []})
    return replay(chains, signals, spots, divs, ReplayRules(hedge="sin_cobertura", **rules))


def test_fill_only_when_next_snapshot_crosses_limit():
    # Día 1: ask neto 3,0 - 0,8 = 2,2 > 2,0 -> no se llena.
    chains = {D[0]: _chain(D[0], (2.5, 2.9), (0.9, 1.1)), D[1]: _chain(D[1], (2.6, 3.0), (0.8, 1.0))}
    out = _run(chains, {D[0]: {"crr": _signal()}})
    assert out["estructuras"].loc[0, "motivo"] == "no_llenada"
    assert out["diario"]["valor"].abs().max() == 0


def test_fill_at_limit_and_take_profit_at_liquidation():
    chains = {
        D[0]: _chain(D[0], (2.5, 2.9), (0.9, 1.1)),
        D[1]: _chain(D[1], (2.6, 2.9), (0.9, 1.0)),   # ask neto 2,9 - 0,9 = 2,0 <= 2,0: se llena a 2,0
        D[2]: _chain(D[2], (3.5, 3.6), (1.0, 1.1)),   # liquidación 3,5 - 1,1 = 2,4 -> +40 - 1,3 - 1,3
    }
    out = _run(chains, {D[0]: {"crr": _signal(edge=20.0)}}, take_profit_fraction=0.6)
    t = out["estructuras"].iloc[0]
    assert t["abierta"] == D[1] and t["motivo"] == "convergencia"
    assert t["efectivo_entrada"] == pytest.approx(-200.0 - 1.3)
    assert t["pnl"] == pytest.approx(240.0 - 1.3 - 200.0 - 1.3)
    # La cuenta (sin cobertura) termina con el P&L realizado.
    assert out["diario"]["valor_crr"].iloc[-1] == pytest.approx(t["pnl"])


def test_kill_switch_measured_at_mid():
    chains = {
        D[0]: _chain(D[0], (2.5, 2.9), (0.9, 1.1)),
        D[1]: _chain(D[1], (2.6, 2.9), (0.9, 1.0)),
        D[2]: _chain(D[2], (0.2, 0.4), (1.5, 1.7)),   # mid neto 0,3 - 1,6 = -1,3: -330 contra el mid
    }
    out = _run(chains, {D[0]: {"crr": _signal()}}, kill_switch_usd=250.0)
    t = out["estructuras"].iloc[0]
    assert t["motivo"] == "kill_switch"
    assert t["pnl"] < -250


def test_cap_per_model_and_dedupe():
    chains = {d: _chain(d, (2.6, 2.9), (0.9, 1.0)) for d in D[:2]}
    sig = pd.concat([_signal(), _signal()], ignore_index=True)   # mismas patas dos veces
    out = _run(chains, {D[0]: {"crr": sig}})
    assert len(out["estructuras"]) == 1


def test_trade_stats_counts():
    chains = {
        D[0]: _chain(D[0], (2.5, 2.9), (0.9, 1.1)),
        D[1]: _chain(D[1], (2.6, 2.9), (0.9, 1.0)),
        D[2]: _chain(D[2], (3.5, 3.6), (1.0, 1.1)),
    }
    stats = trade_stats(_run(chains, {D[0]: {"crr": _signal()}})["estructuras"])
    assert stats.loc[0, "llenadas"] == 1 and stats.loc[0, "cerradas"] == 1
