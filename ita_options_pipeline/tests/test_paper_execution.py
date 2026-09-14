"""Secuencia de paridad, ejecución mleg, ledger y loop con un broker falso."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import pandas as pd
import pytest

pytest.importorskip("alpaca", reason="alpaca-py no instalado")

from conftest import SPOT, build_chain
from fake_broker import FakeBroker, SimTime

from ita_options.arbitrage import ExecutionCosts
from ita_options.config import ExecutionSettings
from ita_options.execution.ledger import Ledger, reconcile, structure_key
from ita_options.execution.live_runner import LiveRunner
from ita_options.execution.parity import ParitySequencer
from ita_options.execution.router import Timing, execute_mleg

EXPIRY = pd.Timestamp("2026-10-16")
CALL, PUT = "RTX261016C00100000", "RTX261016P00100000"
TODAY = date(2026, 9, 14)


def opt(symbol, qty, price, strike, kind):
    return {"kind": "option", "symbol": symbol, "qty": qty, "price": price,
            "strike": strike, "option_type": kind, "expiration": EXPIRY}


def conversion_signal(stock_price=102.0, gross_credit=0.5):
    legs = (opt(CALL, -1.0, 5.0, 100.0, "c"), opt(PUT, 1.0, 3.0, 100.0, "p"),
            {"kind": "stock", "symbol": "RTX", "qty": 1.0, "price": stock_price,
             "strike": float("nan"), "option_type": "", "expiration": pd.NaT})
    return {"detector": "put_call_parity", "underlying": "RTX", "timestamp": "t0",
            "leg_spec": legs, "gross_credit": gross_credit, "commission": 1.3,
            "net_edge_usd": gross_credit * 100 - 1.3}


def reverse_signal():
    signal = conversion_signal()
    legs = (opt(CALL, 1.0, 5.2, 100.0, "c"), opt(PUT, -1.0, 2.8, 100.0, "p"),
            {"kind": "stock", "symbol": "RTX", "qty": -1.0, "price": 101.9,
             "strike": float("nan"), "option_type": "", "expiration": pd.NaT})
    return {**signal, "leg_spec": legs}


def setup(broker, **settings):
    ledger = Ledger()
    config = ExecutionSettings(**{"order_poll_s": 1.0, **settings})
    clock = SimTime()
    sequencer = ParitySequencer(broker, ledger, config, ExecutionCosts(),
                                Timing(sleep=clock.sleep, now=clock.now))
    return ledger, sequencer, config, clock


def open_parity(broker, signal=None, **settings):
    ledger, sequencer, config, _ = setup(broker, **settings)
    signal = signal or conversion_signal()
    strategy_id = ledger.create_strategy(signal, "parity", config.contracts, TODAY)
    outcome = sequencer.open(strategy_id, signal)
    ledger.update_strategy(strategy_id, state=outcome.state, contracts_open=outcome.contracts_open,
                           entry_cash=outcome.entry_cash)
    return ledger, outcome


# --------------------------------------------------------------------------- #
# Paridad
# --------------------------------------------------------------------------- #


def test_conversion_con_fill_completo_abre_y_reconcilia() -> None:
    broker = FakeBroker()
    ledger, outcome = open_parity(broker)
    assert outcome.state == "open" and outcome.contracts_open == 1
    # Acciones primero, opciones después.
    assert broker.kinds() == ["limit_buy", "mleg"]
    assert broker.positions() == {"RTX": 100.0, CALL: -1.0, PUT: 1.0}
    assert reconcile(ledger, broker).ok
    assert outcome.residual_edge >= 5.0


def test_la_prueba_en_paper_mleg_antes_del_fill_completo_es_rechazada() -> None:
    """Reproduce el error observado: con acciones parciales, la venta del call se rechaza."""
    broker = FakeBroker(stock_fill_qty=60)
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest

    from ita_options.execution.broker import BrokerRejection
    from ita_options.execution.orders import build_mleg_order

    broker.submit(LimitOrderRequest(symbol="RTX", qty=100, side=OrderSide.BUY,
                                    time_in_force=TimeInForce.DAY, limit_price=102.05))
    legs = list(conversion_signal()["leg_spec"])[:2]
    with pytest.raises(BrokerRejection, match="uncovered"):
        broker.submit(build_mleg_order(legs, 1, -2.0))


def test_acciones_parciales_sin_lote_completo_no_envian_opciones() -> None:
    broker = FakeBroker(stock_fill_qty=60)
    ledger, outcome = open_parity(broker)
    assert outcome.state == "aborted"
    assert "mleg" not in broker.kinds()
    assert broker.kinds() == ["limit_buy", "market_sell"]  # compra parcial y deshacer 60
    assert broker.positions() == {}


def test_acciones_parciales_con_un_lote_operan_ese_lote_y_deshacen_el_resto() -> None:
    broker = FakeBroker(stock_fill_qty=150)
    ledger, outcome = open_parity(broker, contracts=2)
    assert outcome.state == "open" and outcome.contracts_open == 1
    assert broker.positions() == {"RTX": 100.0, CALL: -1.0, PUT: 1.0}
    assert reconcile(ledger, broker).ok


def test_espera_a_ver_la_posicion_antes_de_vender_el_call() -> None:
    broker = FakeBroker(position_lag_calls=3)
    _, outcome = open_parity(broker)
    assert outcome.state == "open"


def test_posicion_no_confirmada_deshace_sin_enviar_opciones() -> None:
    broker = FakeBroker(position_lag_calls=50)
    _, outcome = open_parity(broker, coverage_checks=3)
    assert outcome.state == "aborted"
    assert "mleg" not in broker.kinds()


def test_mleg_rechazada_deshace_las_acciones() -> None:
    broker = FakeBroker(reject_mleg="account not eligible to trade uncovered option contracts")
    ledger, outcome = open_parity(broker)
    assert outcome.state == "hedge_failed" and not outcome.orphan
    assert broker.positions() == {}
    assert "40310000" in outcome.messages[-1]
    assert reconcile(ledger, broker).ok


def test_mleg_sin_fill_cancela_y_deshace() -> None:
    broker = FakeBroker(option_fill_units=0)
    _, outcome = open_parity(broker)
    assert outcome.state == "hedge_failed"
    assert broker.kinds() == ["limit_buy", "mleg", "market_sell"]
    assert broker.positions() == {}


def test_edge_perdido_con_el_precio_real_de_las_acciones() -> None:
    broker = FakeBroker()
    _, outcome = open_parity(broker, stock_limit_tolerance=1.0)  # llena 1 USD peor
    assert outcome.state == "edge_gone"
    assert "mleg" not in broker.kinds() and broker.positions() == {}


def test_deshacer_fallido_marca_huerfano() -> None:
    broker = FakeBroker(stock_fill_qty=60, market_fill=False)
    ledger, outcome = open_parity(broker)
    assert outcome.orphan
    assert ledger.events("orphan_shares")


def test_conversion_reversa() -> None:
    broker = FakeBroker()
    _, outcome = open_parity(broker, reverse_signal())
    assert outcome.state == "open"
    assert broker.kinds() == ["limit_sell", "mleg"]
    assert broker.positions() == {"RTX": -100.0, CALL: 1.0, PUT: -1.0}


def test_cierre_de_paridad_opciones_primero() -> None:
    broker = FakeBroker()
    ledger, sequencer, config, _ = setup(broker)
    signal = conversion_signal()
    strategy_id = ledger.create_strategy(signal, "parity", 1, TODAY)
    outcome = sequencer.open(strategy_id, signal)
    ledger.update_strategy(strategy_id, state="open", contracts_open=outcome.contracts_open,
                           entry_cash=outcome.entry_cash)
    quotes = pd.DataFrame({"bid": [4.9, 2.9], "ask": [5.1, 3.1]}, index=[CALL, PUT])
    units, cash, _, orphan = sequencer.close(ledger.strategy(strategy_id), quotes)
    assert units == 1 and not orphan
    assert broker.kinds()[-2:] == ["mleg", "market_sell"]
    assert broker.positions() == {}


# --------------------------------------------------------------------------- #
# mleg sólo de opciones
# --------------------------------------------------------------------------- #


def butterfly_signal():
    legs = (opt("RTX261016C00095000", 0.5, 7.0, 95.0, "c"),
            opt("RTX261016C00105000", 0.5, 2.0, 105.0, "c"),
            opt("RTX261016C00100000", -1.0, 5.0, 100.0, "c"))
    gross = 5.0 - 0.5 * 7.0 - 0.5 * 2.0
    return {"detector": "butterfly", "underlying": "RTX", "timestamp": "t0", "leg_spec": legs,
            "gross_credit": gross, "commission": 1.95, "net_edge_usd": gross * 100 - 1.95}


def run_mleg(broker, **settings):
    ledger = Ledger()
    config = ExecutionSettings(**{"order_poll_s": 1.0, **settings})
    clock = SimTime()
    signal = butterfly_signal()
    strategy_id = ledger.create_strategy(signal, "mleg", config.contracts, TODAY)
    return ledger, execute_mleg(broker, ledger, strategy_id, signal, config, ExecutionCosts(),
                                Timing(sleep=clock.sleep, now=clock.now))


def test_mleg_llena_con_credito() -> None:
    broker = FakeBroker()
    _, outcome = run_mleg(broker)
    assert outcome.state == "open" and outcome.contracts_open == 1
    request = broker.submitted[0]
    assert request.limit_price == -1.0  # crédito de 1.00 por unidad escalada 1/1/-2
    assert outcome.entry_cash == pytest.approx(100.0)


def test_mleg_sin_fill_re_precia_una_vez_y_no_baja_del_edge_minimo() -> None:
    broker = FakeBroker(option_fill_units=0)
    _, outcome = run_mleg(broker, max_reprices=3, reprice_step=0.20)
    # Edge de la señal: 48.05 USD; cada re-precio de 0.20 escalado (0.10 por unidad
    # de la señal) cuesta 10 USD, así que los cuatro intentos superan el mínimo de 5.
    assert outcome.state == "unfilled"
    assert len(broker.submitted) == 4
    # Con 1.00 escalado cada re-precio cuesta 50 USD: el primero ya quedaría bajo 40.
    broker_tight = FakeBroker(option_fill_units=0)
    _, outcome_tight = run_mleg(broker_tight, max_reprices=3, reprice_step=1.0, min_edge_usd=40.0)
    assert len(broker_tight.submitted) == 1
    assert any("bajo el mínimo" in m for m in outcome_tight.messages)


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #


def test_ledger_posiciones_esperadas_y_diferencias() -> None:
    ledger = Ledger()
    strategy_id = ledger.create_strategy(butterfly_signal(), "mleg", 1, TODAY)
    ledger.update_strategy(strategy_id, state="open", contracts_open=2)
    assert ledger.expected_positions() == {
        "RTX261016C00095000": 2.0, "RTX261016C00105000": 2.0, "RTX261016C00100000": -4.0}
    assert ledger.seen_structure(structure_key(butterfly_signal()), TODAY + timedelta(days=5))
    broker = FakeBroker()
    broker._positions = {"RTX261016C00095000": 2.0, "RTX261016C00105000": 2.0,
                         "RTX261016C00100000": -3.0, "BA": 10.0}
    report = reconcile(ledger, broker, roots=["RTX"])
    assert report.diffs == {"RTX261016C00100000": (-4.0, -3.0)}


# --------------------------------------------------------------------------- #
# Loop
# --------------------------------------------------------------------------- #


class FakeMarket:
    def __init__(self, chains):
        self.chains = list(chains)
        self.calls = 0

    def spots(self):
        return {"RTX": SPOT}

    def universe(self, spots):
        chain = self.chains[0]
        master = chain[["symbol", "underlying", "expiration", "strike", "option_type"]].copy()
        master["style"] = "american"
        master["multiplier"] = 100.0
        master["open_interest"] = 1000.0
        master["open_interest_date"] = pd.Timestamp("2026-09-02")
        master["tradable"] = True
        return master

    def quotes(self, symbols):
        chain = self.chains[min(self.calls, len(self.chains) - 1)]
        self.calls += 1
        return chain[["symbol", "timestamp", "bid", "ask", "bid_size", "ask_size"]].copy()

    def bars(self, now):
        stamp = self.chains[0]["timestamp"].iloc[0]
        # Barra rotulada un minuto antes del quote: su cierre está disponible en el quote.
        return pd.DataFrame({"timestamp": [stamp - pd.Timedelta(minutes=1)],
                             "symbol": ["RTX"], "close": [SPOT]})


def dislocated_chain(shift: float = 3.0):
    chain = build_chain()
    mask = (chain["option_type"] == "c") & (chain["strike"] == 150.0)
    chain.loc[mask, ["bid", "ask"]] += shift
    return chain


def make_runner(tmp_path, broker, chains, dry_run, now="2026-09-03 15:00", **settings):
    base = {"detectors": ("butterfly",), "kill_file": tmp_path / "KILL",
            "order_poll_s": 1.0, "max_open_strategies": 2}
    config = ExecutionSettings(**{**base, **settings})
    clock = SimTime()
    broker.next_close = pd.Timestamp("2026-09-03 20:00", tz="UTC")
    return LiveRunner(broker, FakeMarket(chains), Ledger(), config, dry_run=dry_run,
                      report_dir=tmp_path / "reportes", timing=Timing(sleep=clock.sleep, now=clock.now),
                      now=lambda: pd.Timestamp(now, tz="UTC"))


def test_loop_dry_run_registra_sin_enviar(tmp_path) -> None:
    broker = FakeBroker()
    runner = make_runner(tmp_path, broker, [dislocated_chain()], dry_run=True)
    report = runner.run_cycle()
    assert report.signals > 0 and report.entries_attempted > 0
    assert broker.submitted == []
    states = {s["state"] for s in runner.ledger.strategies()}
    assert "dry_run" in states
    assert runner.ledger.events("dry_run_orders")
    # La misma estructura no se vuelve a intentar en la rueda.
    second = runner.run_cycle()
    assert second.entries_attempted == 0


def test_loop_abre_y_el_kill_switch_cierra(tmp_path) -> None:
    broker = FakeBroker()
    # Se vende el cuerpo caro (+3); en el ciclo siguiente se encarece más (+6):
    # recomprarlo cuesta más de lo cobrado y la marca da pérdida.
    runner = make_runner(tmp_path, broker, [dislocated_chain(3.0), dislocated_chain(6.0)],
                         dry_run=False, kill_switch_usd=1.0)
    first = runner.run_cycle()
    assert first.entries_opened >= 1
    assert reconcile(runner.ledger, broker).ok
    second = runner.run_cycle()
    assert second.closes >= 1
    closed = runner.ledger.strategies(states=["closed"])
    assert closed and all(s["exit_reason"] == "kill_switch" for s in closed)
    assert reconcile(runner.ledger, broker).ok


def test_loop_archivo_de_corte_detiene_y_cierra(tmp_path) -> None:
    broker = FakeBroker()
    runner = make_runner(tmp_path, broker, [dislocated_chain()], dry_run=False)
    runner.run_cycle()
    (tmp_path / "KILL").write_text("stop")
    report = runner.run_cycle()
    assert report.halted == "kill_file"
    assert runner.ledger.open_strategies() == []
    assert report.entries_attempted == 0


def test_loop_no_entra_fuera_de_la_ventana(tmp_path) -> None:
    broker = FakeBroker()
    runner = make_runner(tmp_path, broker, [dislocated_chain()], dry_run=False,
                         now="2026-09-03 13:35")  # 09:35 ET, antes de apertura + 15 min
    report = runner.run_cycle()
    assert report.signals == 0 and broker.submitted == []


def test_spot_alineado_al_timestamp_del_quote_y_no_al_ultimo_quote() -> None:
    """Regresión del dry-run: quotes demorados contra spot en tiempo real fabrican paridades."""
    from ita_options.config import LiquidityThresholds, PricingAssumptions
    from ita_options.execution.live_data import build_live_chain

    market = FakeMarket([build_chain()])
    master = market.universe({})
    quotes = market.quotes([])
    stamp = quotes["timestamp"].iloc[0]
    late = quotes.iloc[:2].copy()
    late["timestamp"] = stamp + pd.Timedelta(minutes=30)
    late["symbol"] = late["symbol"]  # mismos contratos, quote posterior sin barra cercana
    bars = pd.DataFrame({
        "timestamp": [stamp - pd.Timedelta(minutes=10), stamp - pd.Timedelta(minutes=1)],
        "symbol": ["RTX", "RTX"], "close": [140.0, 145.0],
    })
    chain, _ = build_live_chain(quotes, master, {"RTX": 199.0}, PricingAssumptions(),
                                LiquidityThresholds(), stamp, 1200.0, bars=bars)
    assert (chain["underlying_price"] == 145.0).all()
    assert (chain["spot_source"] == "barra_alineada").all()

    stale, _ = build_live_chain(late, master, {"RTX": 199.0}, PricingAssumptions(),
                                LiquidityThresholds(), stamp + pd.Timedelta(minutes=30), 1200.0,
                                bars=bars)
    assert stale["underlying_price"].isna().all()
    assert not stale["is_tradable"].any()


def test_signo_de_posicion_por_side() -> None:
    from types import SimpleNamespace

    from ita_options.execution.broker import AlpacaBroker, signed_quantity

    assert signed_quantity("100", "short") == -100.0
    assert signed_quantity("-100", "short") == -100.0
    assert signed_quantity("2", SimpleNamespace(value="long")) == 2.0
    client = SimpleNamespace(get_all_positions=lambda: [
        SimpleNamespace(symbol="RTX", qty="100", side=SimpleNamespace(value="short")),
        SimpleNamespace(symbol=CALL, qty="1", side=SimpleNamespace(value="long")),
    ])
    assert AlpacaBroker(client).positions() == {"RTX": -100.0, CALL: 1.0}


def test_excepcion_en_ejecucion_no_mata_el_loop(tmp_path) -> None:
    class BrokenBroker(FakeBroker):
        def submit(self, request):
            raise ConnectionError("se cortó la red")

    broker = BrokenBroker()
    runner = make_runner(tmp_path, broker, [dislocated_chain()], dry_run=False)
    report = runner.run_cycle()
    assert report.halted == "excepcion_en_ejecucion"
    assert runner.ledger.events("execution_error")
    second = runner.run_cycle()  # sigue ciclando: gestiona y reconcilia, sin entrar
    assert second.phase == "open" and second.entries_attempted == 0


def test_loop_reporte_de_sesion(tmp_path) -> None:
    broker = FakeBroker()
    broker.is_open = True
    runner = make_runner(tmp_path, broker, [dislocated_chain()], dry_run=True)
    path = runner.run(max_cycles=2)
    summary = (path / "summary.json").read_text("utf-8")
    assert '"dry_run": true' in summary and "indicative" in summary
    assert (path / "strategies.csv").exists() and (path / "cycles.csv").exists()
