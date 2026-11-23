"""Cobertura delta en vivo por modelo (Parte 2, paso 4)."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("alpaca", reason="alpaca-py no instalado")

from conftest import build_chain  # noqa: E402
from fake_broker import FakeBroker, SimTime  # noqa: E402
from test_paper_execution import FakeMarket  # noqa: E402

from ita_options.arbitrage import ExecutionCosts  # noqa: E402
from ita_options.config import ExecutionSettings  # noqa: E402
from ita_options.execution.hedger import (  # noqa: E402
    DeltaHedger, HedgeSettings, model_of_detector, portfolio_greeks,
)
from ita_options.execution.ledger import Ledger, reconcile  # noqa: E402
from ita_options.execution.live_runner import LiveRunner  # noqa: E402
from ita_options.execution.model_rv import CompetingModels, RVSettings  # noqa: E402
from ita_options.execution.router import Timing  # noqa: E402

NOW = pd.Timestamp("2026-09-03 15:00", tz="UTC")


def _chain() -> pd.DataFrame:
    return build_chain(expiries=(("2026-10-16", 43 / 365),), spread=0.20)


def _strategy(chain: pd.DataFrame, detector: str, k_long: float, k_short: float, contracts: int = 1) -> dict:
    calls = chain.loc[chain["option_type"] == "c"].set_index("strike")
    return {"detector": detector, "underlying": "RTX", "contracts_open": contracts, "legs": [
        {"kind": "option", "symbol": calls.loc[k_long, "symbol"], "units": 1},
        {"kind": "option", "symbol": calls.loc[k_short, "symbol"], "units": -1},
    ]}


def test_el_detector_dice_el_modelo() -> None:
    assert model_of_detector("rv_crr_vertical") == "crr"
    assert model_of_detector("rv_bsm_butterfly") == "bsm"
    assert model_of_detector("put_call_parity") is None


def test_griegas_del_portafolio_por_modelo() -> None:
    chain = _chain()
    strategies = [_strategy(chain, "rv_crr_vertical", 145, 150), _strategy(chain, "rv_bsm_vertical", 150, 155, 2)]
    greeks, missing = portfolio_greeks(strategies, chain)
    assert not missing
    g = greeks.set_index("modelo")
    # Call spread comprado: delta positiva, menor que 100 acciones por contrato.
    assert 0 < g.loc["crr", "delta"] < 100
    assert 0 < g.loc["bsm", "delta"] < 200
    assert g.loc["crr", "patas"] == 2 and g.loc["bsm", "patas"] == 2


def test_un_libro_sin_cotizacion_no_se_rebalancea() -> None:
    chain = _chain()
    strategy = _strategy(chain, "rv_crr_vertical", 145, 150)
    ledger = Ledger()
    hedger = DeltaHedger(FakeBroker(), ledger, HedgeSettings(mode="delta"), ExecutionSettings(), dry_run=False)
    table = hedger.run([strategy], chain.loc[chain["strike"] != 150], {"RTX": 150.0})
    assert table.iloc[0]["accion"] == "faltan_patas" and table.iloc[0]["operado"] == 0


def test_el_cruce_por_cero_se_parte_en_dos_ordenes() -> None:
    broker = FakeBroker()
    ledger = Ledger()
    hedger = DeltaHedger(broker, ledger, HedgeSettings(mode="delta"), ExecutionSettings(order_poll_s=1.0),
                         timing=Timing(sleep=SimTime().sleep, now=SimTime().now), dry_run=False)
    assert hedger._trade("crr", "RTX", 50, 150.0, shorting=True) == 50
    assert hedger._trade("crr", "RTX", -80, 150.0, shorting=True) == -80
    sides = [r.side.value for r in broker.submitted]
    qtys = [float(r.qty) for r in broker.submitted]
    assert sides == ["buy", "sell", "sell"] and qtys == [50, 50, 30]
    assert ledger.hedge_book("crr", "RTX")["shares"] == -30
    # Sin venta en corto: sólo se puede bajar el largo hasta cero.
    broker2, ledger2 = FakeBroker(), Ledger()
    h2 = DeltaHedger(broker2, ledger2, HedgeSettings(mode="delta"), ExecutionSettings(order_poll_s=1.0),
                     timing=Timing(sleep=SimTime().sleep, now=SimTime().now), dry_run=False)
    h2._trade("bsm", "RTX", 20, 150.0, shorting=False)
    assert h2._trade("bsm", "RTX", -50, 150.0, shorting=False) == -20
    assert ledger2.events("cobertura_limitada")


def test_loop_en_vivo_cubre_cada_libro_y_lo_cierra_con_el_kill_switch(tmp_path) -> None:
    chain = _chain()
    broker = FakeBroker(market_price=150.0)
    broker.next_close = pd.Timestamp("2026-09-03 20:00", tz="UTC")
    clock = SimTime()
    timing = Timing(sleep=clock.sleep, now=clock.now)
    settings = ExecutionSettings(kill_file=tmp_path / "KILL", order_poll_s=1.0, max_open_strategies=6,
                                 min_edge_usd=5.0, max_dte=75)
    ledger = Ledger()
    source = CompetingModels(
        ledger, ExecutionCosts(min_net_edge=5.0),
        {m: RVSettings(model=m, min_edge_usd=5.0, moneyness_band=0.2, max_signals=3) for m in ("crr", "bsm")},
        cap_per_model=2,
    )
    hedger = DeltaHedger(broker, ledger, HedgeSettings(mode="delta", band_shares=1.0), settings,
                         timing=timing, dry_run=False)
    runner = LiveRunner(broker, FakeMarket([chain]), ledger, settings, dry_run=False,
                        report_dir=tmp_path / "rep", timing=timing, now=lambda: NOW,
                        signal_source=source, hedger=hedger)
    report = runner.run_cycle()
    assert report.entries_opened > 0
    greeks = ledger.events("griegas_portafolio")
    assert greeks
    books = {b["book"]: b["shares"] for b in ledger.hedge_books()}
    table, _ = portfolio_greeks(ledger.open_strategies(), runner_chain(runner, chain))
    for row in table.itertuples():
        target = -round(row.delta)
        if abs(target) >= 1:
            assert books[f"{row.modelo}:RTX"] == target
    assert reconcile(ledger, broker).ok
    # Kill-switch manual: se cierran las estructuras y los libros quedan en cero.
    (tmp_path / "KILL").write_text("")
    runner.run_cycle()
    assert all(b["shares"] == 0 for b in ledger.hedge_books())
    assert broker.positions().get("RTX", 0.0) == 0


def runner_chain(runner: LiveRunner, chain: pd.DataFrame) -> pd.DataFrame:
    """La cadena del ciclo tal como la arma el runner (para recalcular las griegas en el test)."""
    from ita_options.execution.live_data import build_live_chain

    market = runner.market
    quotes = market.quotes([])
    live, _ = build_live_chain(quotes, market.universe(market.spots()), market.spots(), runner.pricing,
                               runner.liquidity, NOW, runner.settings.max_quote_age_s, bars=market.bars(NOW))
    return live


# --------------------------------------------------------------------------- #
# Delta-gamma: opción ATM corta comprada + acciones
# --------------------------------------------------------------------------- #


def _two_expiries() -> pd.DataFrame:
    return build_chain(expiries=(("2026-09-23", 20 / 365), ("2026-10-16", 43 / 365)), spread=0.20)


def _single(chain: pd.DataFrame, strike: float, units: int, contracts: int) -> dict:
    long_exp = max(chain["expiration"].unique())
    calls = chain.loc[(chain["option_type"] == "c") & (chain["expiration"] == long_exp)]
    calls = calls.set_index("strike")
    return {"detector": "rv_crr_vertical", "underlying": "RTX", "contracts_open": contracts,
            "legs": [{"kind": "option", "symbol": calls.loc[strike, "symbol"], "units": units}]}


def test_la_opcion_de_gamma_es_la_atm_corta_y_se_mantiene_cerca_del_dinero() -> None:
    from ita_options.execution.hedger import pick_gamma_option

    chain = _two_expiries()
    short_exp, long_exp = sorted(chain["expiration"].unique())
    pick = pick_gamma_option(chain, "RTX", exclude={long_exp})
    assert pick["expiration"] == short_exp and pick["strike"] == 150 and pick["option_type"] == "c"
    # El vencimiento de la propia posición se excluye.
    assert pick_gamma_option(chain, "RTX", exclude={short_exp})["expiration"] == long_exp
    # La que ya se tiene se mantiene si sigue a menos de 5% del dinero.
    held = chain.loc[(chain["expiration"] == short_exp) & (chain["strike"] == 155) & (chain["option_type"] == "c"),
                     "symbol"].iloc[0]
    assert pick_gamma_option(chain, "RTX", exclude={long_exp}, keep=held)["symbol"] == held


def _hedger(broker: FakeBroker, ledger: Ledger, mode: str = "delta_gamma") -> DeltaHedger:
    clock = SimTime()
    return DeltaHedger(broker, ledger, HedgeSettings(mode=mode, band_shares=1.0), ExecutionSettings(order_poll_s=1.0),
                       timing=Timing(sleep=clock.sleep, now=clock.now), dry_run=False)


def test_un_libro_corto_en_gamma_compra_la_atm_y_cubre_la_delta_restante() -> None:
    chain = _two_expiries()
    broker, ledger = FakeBroker(market_price=150.0), Ledger()
    hedger = _hedger(broker, ledger)
    strategy = _single(chain, 150, -1, 3)  # 3 calls ATM vendidas: gamma negativa
    table = hedger.run([strategy], chain, {"RTX": 150.0})
    row = table.iloc[0]
    assert row["gamma"] < 0 and row["gamma_n_objetivo_continuo"] > 0.5
    options = ledger.hedge_options("crr", "RTX")
    assert len(options) == 1 and options[0]["contracts"] == row["gamma_n_objetivo"] >= 1
    # La delta que queda se cubre con acciones, contando la delta de la opción comprada.
    assert ledger.hedge_book("crr", "RTX")["shares"] == row["objetivo"]
    assert row["objetivo"] == -round(row["delta"] + row["delta_opcion_cobertura"])
    # La cuenta tiene exactamente lo que dice el ledger (acciones y opciones).
    assert reconcile(ledger, broker).ok
    # Cierre: se venden las opciones compradas y se cierran las acciones.
    hedger.run([], chain, {"RTX": 150.0}, flatten=True)
    assert not ledger.hedge_options()
    assert all(b["shares"] == 0 for b in ledger.hedge_books())
    assert reconcile(ledger, broker).ok and not broker.positions()


def test_un_libro_largo_en_gamma_no_vende_opciones_descubiertas() -> None:
    chain = _two_expiries()
    broker, ledger = FakeBroker(market_price=150.0), Ledger()
    table = _hedger(broker, ledger).run([_single(chain, 150, +1, 3)], chain, {"RTX": 150.0})
    row = table.iloc[0]
    assert row["gamma_n_objetivo_continuo"] < -0.5 and row["gamma_n_objetivo"] == 0
    assert "descubiertas" in row["gamma_limitacion"]
    assert not ledger.hedge_options()
    # La delta igual se cubre con acciones.
    assert ledger.hedge_book("crr", "RTX")["shares"] == -round(row["delta"])
