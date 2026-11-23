"""Dividendos point-in-time, spot implícito, carry y asignación anticipada.

Cubren las piezas nuevas de los puntos 1 y 2 de la devolución: la curación de
la foto de DoltHub y el costo de tener abierta una paridad.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ita_options.backtest import ArbitrageBacktester, CarryModel, PointInTimeView, Position
from ita_options.dolthub import implied_spot, merge_dividends, occ_symbol, projected_dividends

DIVS = pd.DataFrame({
    "ticker": ["LMT", "LMT", "RTX"],
    "ex_dividend_date": [date(2026, 3, 2), date(2026, 6, 1), date(2026, 5, 22)],
    "declaration_date": [date(2026, 2, 6), date(2026, 5, 12), date(2026, 4, 30)],
    "cash_amount": [3.45, 3.60, 0.73],
})


def test_dividendo_no_declarado_se_proyecta_con_el_monto_anterior() -> None:
    # El 10/5 el dividendo de junio todavía no fue declarado: se proyecta el de marzo.
    flows = projected_dividends(DIVS, "LMT", date(2026, 5, 10), date(2026, 7, 1))
    assert flows == [(date(2026, 6, 1), 3.45)]


def test_dividendo_declarado_entra_con_su_monto_real() -> None:
    flows = projected_dividends(DIVS, "LMT", date(2026, 5, 13), date(2026, 7, 1))
    assert flows == [(date(2026, 6, 1), 3.60)]


def test_sin_historia_declarada_no_proyecta() -> None:
    assert projected_dividends(DIVS, "BA", date(2026, 5, 13), date(2026, 9, 1)) == []
    assert projected_dividends(DIVS, "RTX", date(2026, 4, 1), date(2026, 9, 1)) == []


def test_merge_prefiere_la_fecha_de_declaracion_de_polygon(tmp_path: Path) -> None:
    poly = pd.DataFrame({
        "ticker": ["RTX"], "ex_dividend_date": ["2026-05-22"], "declaration_date": ["2026-04-30"],
        "cash_amount": [0.73], "distribution_type": ["recurring"],
    })
    poly.to_parquet(tmp_path / "p.parquet")
    yahoo = pd.DataFrame({"ticker": ["RTX", "RTX"], "ex_dividend_date": [date(2026, 5, 22), date(2026, 11, 20)],
                          "cash_amount": [0.73, 0.73]})
    merged = merge_dividends(yahoo, tmp_path / "p.parquet")
    assert len(merged) == 2
    may = merged.loc[merged["ex_dividend_date"] == date(2026, 5, 22)].iloc[0]
    assert may["fuente"] == "polygon" and may["declaration_date"] == date(2026, 4, 30)


def test_simbolo_occ() -> None:
    assert occ_symbol("LMT", date(2026, 9, 18), "c", 600.0) == "LMT260918C00600000"
    assert occ_symbol("BA", date(2026, 12, 18), "p", 217.5) == "BA261218P00217500"


def test_spot_implicito_detecta_una_foto_desfasada() -> None:
    rate, tau, spot_snapshot, spot_close = 0.04, 30 / 365, 100.0, 98.0
    rows = []
    for k in (97.0, 98.0, 99.0, 100.0, 101.0, 102.0):
        diff = spot_snapshot - k * np.exp(-rate * tau)  # C - P de la foto
        call = max(diff, 0) + 2.0
        put = call - diff
        for flag, mid in (("c", call), ("p", put)):
            rows.append({"underlying": "X", "trade_date": date(2026, 6, 1), "expiration": date(2026, 7, 1),
                         "strike": k, "option_type": flag, "bid": mid - 0.05, "ask": mid + 0.05, "mid_price": mid,
                         "underlying_price": spot_close, "tau": tau, "risk_free_rate": rate,
                         "pv_dividends": 0.0, "dte": 30})
    gaps = implied_spot(pd.DataFrame(rows))
    assert gaps["implied_spot"].iloc[0] == pytest.approx(spot_snapshot, abs=1e-9)
    assert gaps["spot_gap_pct"].iloc[0] == pytest.approx(100 * (100 / 98 - 1), abs=1e-6)


# --------------------------------------------------------------------------- #
# Carry y asignación anticipada
# --------------------------------------------------------------------------- #

NY_CLOSE = {d: pd.Timestamp(f"{d} 16:00", tz="America/New_York").tz_convert("UTC")
            for d in ("2026-05-28", "2026-05-29", "2026-06-01")}


def _conversion(opened: str) -> Position:
    legs = (
        {"kind": "option", "symbol": "C", "qty": -1.0, "price": 20.0, "strike": 500.0, "option_type": "c",
         "expiration": pd.Timestamp("2026-06-19")},
        {"kind": "option", "symbol": "P", "qty": 1.0, "price": 1.0, "strike": 500.0, "option_type": "p",
         "expiration": pd.Timestamp("2026-06-19")},
        {"kind": "stock", "symbol": "LMT", "qty": 1.0, "price": 519.0, "strike": np.nan, "option_type": "",
         "expiration": pd.NaT},
    )
    return Position(position_id=0, detector="put_call_parity", underlying="LMT", legs=legs,
                    opened_at=NY_CLOSE[opened], entry_cash=-(519.0 + 1.0 - 20.0) * 100,
                    entry_commission=1.3, predicted_edge=50.0, expiration=pd.Timestamp("2026-06-19"))


def test_carry_cobra_intereses_y_acredita_el_dividendo_a_la_accion_comprada() -> None:
    carry = CarryModel(rate=0.04, funding_spread=0.005, borrow_rate=0.01, dividends=DIVS)
    position = _conversion("2026-05-28")
    interest, borrow, dividends = carry.charges(position, NY_CLOSE["2026-06-01"], 100.0, 1.0)
    assert interest == pytest.approx(-50_000 * 0.045 * 4 / 365)
    assert borrow == 0.0
    assert dividends == pytest.approx(100 * 3.60)


def test_call_corto_en_el_dinero_se_asigna_la_vispera_de_la_fecha_ex() -> None:
    quotes = pd.DataFrame({
        "timestamp": [NY_CLOSE["2026-05-29"]] * 2, "symbol": ["C", "P"],
        "bid": [19.0, 0.9], "ask": [19.5, 1.0], "underlying": ["LMT"] * 2,
    })
    under = pd.DataFrame({"timestamp": [NY_CLOSE["2026-05-29"]], "symbol": ["LMT"], "close": [518.0]})
    view = PointInTimeView(quotes, under)
    view.advance_to(NY_CLOSE["2026-05-29"])
    bt = ArbitrageBacktester(quotes, under, carry=CarryModel(dividends=DIVS))
    position = _conversion("2026-05-28")
    # Valor temporal del call: 19,5 - 18 = 1,5 < dividendo 3,60 -> se ejerce.
    assert bt._assignment_due(position, view, NY_CLOSE["2026-06-01"])
    # Sin fecha ex al día siguiente no hay asignación.
    assert not bt._assignment_due(position, view, NY_CLOSE["2026-05-29"] + pd.Timedelta(hours=1))


def test_cadena_en_vivo_recibe_el_dividendo_discreto() -> None:
    """El loop en vivo usa la misma convención que el backtest, no el yield constante."""
    from ita_options.dolthub import attach_projected_dividends

    chain = pd.DataFrame({
        "underlying": ["LMT", "BA"], "trade_date": [pd.Timestamp("2026-05-20")] * 2,
        "expiration": [pd.Timestamp("2026-06-19")] * 2, "risk_free_rate": [0.04, 0.04],
        "underlying_price": [500.0, 200.0], "tau": [30 / 365] * 2, "dividend_yield": [0.026, 0.0],
    })
    out = attach_projected_dividends(chain, DIVS)
    # LMT: dividendo de junio declarado el 12/5 (3,60), descontado 12 días.
    assert out.loc[0, "pv_dividends"] == pytest.approx(3.60 * np.exp(-0.04 * 12 / 365))
    assert 500 * np.exp(-out.loc[0, "dividend_yield"] * 30 / 365) == pytest.approx(500 - out.loc[0, "pv_dividends"])
    assert out.loc[1, "pv_dividends"] == 0.0 and out.loc[1, "dividend_yield"] == 0.0
