"""Órdenes maker de apertura que quedan publicadas sin bloquear el ciclo."""

import pandas as pd

from ita_options.execution.ledger import reconcile

from fake_broker import FakeBroker
from test_paper_execution import dislocated_chain, make_runner

REST = 1800.0


def _runner(tmp_path, broker, **settings):
    runner = make_runner(tmp_path, broker, [dislocated_chain()], dry_run=False,
                         option_rest_s=REST, max_open_strategies=1, **settings)
    clock = {"now": pd.Timestamp("2026-09-03 15:00", tz="UTC")}
    runner._now = lambda: clock["now"]
    return runner, clock


def _working(runner):
    return runner.ledger.strategies(("working",))


def test_la_orden_queda_publicada_sin_bloquear_el_ciclo(tmp_path) -> None:
    broker = FakeBroker(option_fill_units=0)
    runner, _ = _runner(tmp_path, broker)
    start = runner.timing.now()
    report = runner.run_cycle()
    assert report.entries_attempted == 1 and report.entries_opened == 0
    assert len(_working(runner)) == 1
    # No esperó el fill: el reloj del ciclo no avanzó.
    assert runner.timing.now() == start
    assert reconcile(runner.ledger, broker).ok


def test_si_el_mercado_la_cruza_pasa_a_abierta_con_su_caja(tmp_path) -> None:
    broker = FakeBroker(option_fill_units=0)
    runner, clock = _runner(tmp_path, broker)
    runner.run_cycle()
    strategy = _working(runner)[0]
    order = runner.ledger.orders(strategy["id"])[-1]
    broker.fill_resting(order["broker_id"])
    clock["now"] += pd.Timedelta(minutes=10)
    report = runner.run_cycle()
    assert report.entries_opened == 1
    opened = runner.ledger.strategy(strategy["id"])
    assert opened["state"] == "open" and opened["contracts_open"] == 1
    limit = float(broker.get_order(order["broker_id"]).filled_avg_price)
    assert opened["entry_cash"] == -limit * 100.0
    assert reconcile(runner.ledger, broker).ok
    assert runner.ledger.events("maker_fill")


def test_al_vencer_el_plazo_se_cancela_y_queda_sin_fill(tmp_path) -> None:
    broker = FakeBroker(option_fill_units=0)
    runner, clock = _runner(tmp_path, broker)
    runner.run_cycle()
    strategy = _working(runner)[0]
    clock["now"] += pd.Timedelta(minutes=10)
    runner.run_cycle()
    assert runner.ledger.strategy(strategy["id"])["state"] == "working"  # todavía en plazo
    clock["now"] += pd.Timedelta(seconds=REST)
    runner.run_cycle()
    assert runner.ledger.strategy(strategy["id"])["state"] == "unfilled"
    order = runner.ledger.orders(strategy["id"])[0]
    assert broker.get_order(order["broker_id"]).status == "canceled"


def test_la_orden_publicada_ocupa_cupo(tmp_path) -> None:
    broker = FakeBroker(option_fill_units=0)
    runner, clock = _runner(tmp_path, broker)
    runner.run_cycle()
    sent = len(broker.submitted)
    clock["now"] += pd.Timedelta(minutes=2)
    runner.run_cycle()
    assert len(broker.submitted) == sent  # con el cupo de 1 lleno no publica otra


def test_el_archivo_de_corte_cancela_las_publicadas(tmp_path) -> None:
    broker = FakeBroker(option_fill_units=0)
    runner, _ = _runner(tmp_path, broker)
    runner.run_cycle()
    (tmp_path / "KILL").write_text("stop")
    report = runner.run_cycle()
    assert report.halted == "kill_file"
    assert _working(runner) == []
    assert all(o.status == "canceled" for o in broker._orders.values())


def test_al_terminar_la_rueda_no_quedan_ordenes_publicadas(tmp_path) -> None:
    broker = FakeBroker(option_fill_units=0)
    runner, _ = _runner(tmp_path, broker)
    runner.run(max_cycles=1)
    assert _working(runner) == []
    assert all(o.status == "canceled" for o in broker._orders.values())


def test_sin_plazo_maker_se_mantiene_la_espera_bloqueante(tmp_path) -> None:
    broker = FakeBroker(option_fill_units=0)
    runner = make_runner(tmp_path, broker, [dislocated_chain()], dry_run=False, max_open_strategies=1)
    start = runner.timing.now()
    runner.run_cycle()
    assert _working(runner) == []
    assert runner.timing.now() > start  # esperó el fill y canceló
