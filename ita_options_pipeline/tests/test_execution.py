"""Tests offline de la capa de ejecución: órdenes mleg, riesgo y cuenta."""

from __future__ import annotations

from datetime import date, datetime, timezone
from uuid import uuid4

import pandas as pd
import pytest

pytest.importorskip("alpaca", reason="alpaca-py no instalado")

from alpaca.trading.enums import AccountStatus, OrderClass, OrderSide, PositionIntent
from alpaca.trading.models import Clock, TradeAccount

from ita_options.arbitrage import ExecutionCosts, execution_edge
from ita_options.doctor import describe_clock, evaluate_account
from ita_options.execution.orders import (
    OrderConstructionError,
    build_mleg_order,
    integer_ratios,
    mleg_limit_price,
    scaled_net_debit,
    strategy_order_id,
)
from ita_options.execution.risk import (
    ex_dividend_in_window,
    insufficient_depth,
    max_loss_at_expiry,
    pre_trade_check,
)

EXPIRY = pd.Timestamp("2026-10-16")


def opt(symbol: str, qty: float, price: float, strike: float, kind: str = "c") -> dict:
    return {"kind": "option", "symbol": symbol, "qty": qty, "price": price,
            "strike": strike, "option_type": kind, "expiration": EXPIRY}


def stock(qty: float, price: float) -> dict:
    return {"kind": "stock", "symbol": "RTX", "qty": qty, "price": price,
            "strike": float("nan"), "option_type": "", "expiration": pd.NaT}


# --------------------------------------------------------------------------- #
# Órdenes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("quantities", "expected"),
    [
        ([0.4, 0.6, -1.0], [2, 3, -5]),
        ([0.5, 0.5, -1.0], [1, 1, -2]),
        ([1.0, -2.0, 1.0], [1, -2, 1]),
        ([2.0, -4.0], [1, -2]),
        ([1 / 3, 2 / 3, -1.0], [1, 2, -3]),
    ],
)
def test_ratios_enteros_con_mcd_uno(quantities, expected) -> None:
    assert integer_ratios(quantities) == expected


def test_ratio_no_representable_se_rechaza() -> None:
    # 0.1234 no tiene una fracción con denominador <= 100 a menos de 1e-6.
    with pytest.raises(OrderConstructionError):
        integer_ratios([0.1234, -1.0])


def test_mleg_de_butterfly_valida_contra_el_sdk() -> None:
    legs = [opt("RTX261016C00095000", 0.4, 7.1, 95),
            opt("RTX261016C00105000", 0.6, 1.9, 105),
            opt("RTX261016C00100000", -1.0, 4.2, 100)]
    request = build_mleg_order(legs, contracts=2, limit_price=-0.31, client_order_id="ita-x")
    assert request.order_class == OrderClass.MLEG
    assert request.qty == 2 and request.limit_price == -0.31
    assert [(leg.ratio_qty, leg.side) for leg in request.legs] == [
        (2, OrderSide.BUY), (3, OrderSide.BUY), (5, OrderSide.SELL)
    ]
    assert request.legs[2].position_intent == PositionIntent.SELL_TO_OPEN
    # Neto escalado: 2·7.1 + 3·1.9 − 5·4.2 = −1.1 (crédito).
    assert scaled_net_debit(legs, [7.1, 1.9, 4.2]) == pytest.approx(-1.1)


def test_cerrar_invierte_lados_e_intenciones() -> None:
    legs = [opt("RTX261016C00100000", 1, 3.0, 100), opt("RTX261016C00105000", -1, 1.0, 105)]
    request = build_mleg_order(legs, 1, 1.5, opening=False)
    assert [(leg.side, leg.position_intent) for leg in request.legs] == [
        (OrderSide.SELL, PositionIntent.SELL_TO_CLOSE),
        (OrderSide.BUY, PositionIntent.BUY_TO_CLOSE),
    ]


def test_patas_de_accion_no_van_en_una_mleg() -> None:
    parity = [opt("RTX261016C00100000", -1, 5.0, 100),
              opt("RTX261016P00100000", 1, 3.0, 100, "p"),
              stock(1, 102.0)]
    with pytest.raises(OrderConstructionError, match="acción"):
        build_mleg_order(parity, 1, 0.5)


def test_limites_de_patas_y_contratos() -> None:
    with pytest.raises(OrderConstructionError):
        build_mleg_order([opt("RTX261016C00100000", 1, 3.0, 100)], 1, 3.0)
    legs = [opt("A", 1, 1, 100), opt("B", -1, 1, 105)]
    with pytest.raises(OrderConstructionError):
        build_mleg_order(legs, 0, 1.0)


def test_signo_del_limite_e_id_idempotente() -> None:
    assert mleg_limit_price(1.234) == 1.23
    assert mleg_limit_price(-0.5) == -0.5  # hipótesis vigente: crédito negativo
    first = strategy_order_id("butterfly", ["B", "A"], "2026-09-14", True)
    assert first == strategy_order_id("butterfly", ["A", "B"], "2026-09-14", True)
    assert first != strategy_order_id("butterfly", ["A", "B"], "2026-09-14", False)
    assert len(first) <= 48


# --------------------------------------------------------------------------- #
# Edge al ejecutar compartido con el backtest
# --------------------------------------------------------------------------- #


def test_execution_edge_coincide_con_el_detector_si_no_se_mueven_precios() -> None:
    legs = (opt("L", 0.5, 7.0, 95), opt("H", 0.5, 2.0, 105), opt("M", -1.0, 5.0, 100))
    gross_credit = 5.0 - 0.5 * 7.0 - 0.5 * 2.0
    signal = {"leg_spec": legs, "gross_credit": gross_credit, "commission": 1.95,
              "net_edge_usd": gross_credit * 100 - 1.95}
    costs = ExecutionCosts()
    assert execution_edge(signal, [7.0, 2.0, 5.0], costs) == pytest.approx(
        signal["net_edge_usd"])
    # Si el cuerpo se vende 0.30 más barato, el edge cae 30 USD por contrato.
    assert execution_edge(signal, [7.0, 2.0, 4.7], costs) == pytest.approx(
        signal["net_edge_usd"] - 30)
    with pytest.raises(ValueError):
        execution_edge(signal, [7.0, 2.0], costs)


# --------------------------------------------------------------------------- #
# Riesgo previo
# --------------------------------------------------------------------------- #


def test_perdida_maxima_por_estructura() -> None:
    debit_vertical = [opt("A", 1, 3.0, 100), opt("B", -1, 1.0, 105)]
    assert max_loss_at_expiry(debit_vertical) == pytest.approx(200.0)
    short_vertical = [opt("A", -1, 4.0, 100), opt("B", 1, 1.0, 105)]
    assert max_loss_at_expiry(short_vertical) == pytest.approx(200.0)
    arbitrage_fly = [opt("L", 0.5, 7.0, 95), opt("H", 0.5, 2.0, 105),
                     opt("M", -1.0, 5.0, 100)]
    assert max_loss_at_expiry(arbitrage_fly) == 0.0
    # Conversión: payoff fijo K − (S − C + P) = 100 − (102 − 5 + 3) = 0.
    conversion = [opt("C", -1, 5.0, 100), opt("P", 1, 3.0, 100, "p"), stock(1, 102.0)]
    assert max_loss_at_expiry(conversion) == pytest.approx(0.0)
    assert max_loss_at_expiry([opt("C", -1, 5.0, 100)]) == float("inf")


def test_calendario_no_admite_perdida_maxima_al_vencimiento() -> None:
    far = dict(opt("F", 1, 3.0, 100), expiration=pd.Timestamp("2026-11-20"))
    with pytest.raises(ValueError):
        max_loss_at_expiry([far, opt("N", -1, 2.0, 100)])


def test_profundidad_y_fecha_ex() -> None:
    legs = [opt("A", 1, 3.0, 100), opt("B", -1, 1.0, 105)]
    assert insufficient_depth(legs, {"A": (5, 10), "B": (8, 1)}, contracts=2) == []
    problems = insufficient_depth(legs, {"A": (5, 1), "B": (8, 1)}, contracts=2)
    assert len(problems) == 1 and problems[0].startswith("A: ask")

    dividends = pd.DataFrame({"ticker": ["LMT", "RTX"],
                              "ex_dividend_date": ["2026-10-01", "2026-12-01"]})
    assert ex_dividend_in_window("LMT", date(2026, 9, 14), date(2026, 10, 16),
                                 dividends) == [date(2026, 10, 1)]
    assert ex_dividend_in_window("RTX", date(2026, 9, 14), date(2026, 10, 16),
                                 dividends) == []


def test_control_previo_rechaza_con_motivos() -> None:
    reverse = {
        "detector": "put_call_parity", "underlying": "LMT",
        "leg_spec": [opt("C", 1, 5.0, 500), opt("P", -1, 3.0, 500, "p"), stock(-1, 502.0)],
    }
    dividends = pd.DataFrame({"ticker": ["LMT"], "ex_dividend_date": ["2026-10-01"]})
    decision = pre_trade_check(
        reverse, contracts=1, options_buying_power=50_000, max_loss_per_trade=2_000,
        sizes={"C": (10, 10), "P": (10, 10)}, today=date(2026, 9, 14),
        dividends=dividends, shorting_enabled=False,
    )
    assert not decision.approved
    assert any("short" in r for r in decision.reasons)
    assert any("ex-dividendo" in r for r in decision.reasons)

    vertical = {"detector": "vertical_bound", "underlying": "RTX",
                "leg_spec": [opt("A", -1, 4.0, 100), opt("B", 1, 1.0, 105)]}
    ok = pre_trade_check(vertical, 1, 10_000, 2_000, {"A": (5, 5), "B": (5, 5)},
                         date(2026, 9, 14))
    assert ok.approved and ok.max_loss_usd == pytest.approx(200.0)


# --------------------------------------------------------------------------- #
# Cuenta y reloj, con los modelos reales del SDK
# --------------------------------------------------------------------------- #


def _account(**fields) -> TradeAccount:
    return TradeAccount(id=uuid4(), account_number="PA123",
                        status=AccountStatus.ACTIVE, **fields)


def test_cuenta_con_nivel_3_y_short_pasa() -> None:
    result = evaluate_account(
        _account(options_trading_level=3, options_approved_level=3,
                 options_buying_power="25000", shorting_enabled=True, multiplier="2"),
        paper=True,
    )
    assert result.passed and not result.blocking
    assert "nivel de opciones 3" in result.detail
    assert "short habilitado" in result.detail


def test_cuenta_sin_nivel_3_advierte_sobre_spreads() -> None:
    result = evaluate_account(
        _account(options_trading_level=1, shorting_enabled=False), paper=False
    )
    assert not result.passed
    assert "requieren nivel 3" in result.detail and "cuenta real" in result.detail


def test_reloj_de_mercado() -> None:
    now = datetime(2026, 9, 14, 15, 0, tzinfo=timezone.utc)
    assert describe_clock(
        Clock(timestamp=now, is_open=True, next_open=now, next_close=now)
    ).startswith("mercado abierto")
    assert describe_clock(
        Clock(timestamp=now, is_open=False, next_open=now, next_close=now)
    ).startswith("mercado cerrado")
