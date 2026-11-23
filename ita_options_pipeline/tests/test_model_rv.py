"""Estrategia de valor relativo contra el modelo calibrado (paper trading).

Fija tres propiedades: sobre una cadena generada por el propio modelo no hay
edge después de cruzar el spread; en modo ``maker`` el límite queda dentro del
spread y el edge contra el modelo es exactamente el margen pedido; y el loop en
vivo abre la estructura y la cierra por convergencia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import build_chain
from ita_options.arbitrage import ExecutionCosts, execution_edge
from ita_options.execution.model_rv import RVSettings, detect_model_rv, fit_smile


def _chain() -> pd.DataFrame:
    chain = build_chain(expiries=(("2026-12-18", 0.25),), spread=0.40)
    chain["underlying_price"] = 150.0
    chain["dte"] = 45  # dentro de la ventana operable [21, 75]
    chain["is_tradable"] = True
    return chain


def test_la_sonrisa_recupera_la_volatilidad_plana_del_generador() -> None:
    chain = _chain()
    smile = fit_smile(chain, "crr", RVSettings())
    assert smile is not None
    # El generador usa una sigma única: la sonrisa ajustada es casi plana.
    assert smile.rmse_iv < 0.02
    assert abs(smile.b) < 0.1


def test_sin_edge_al_cruzar_el_spread_en_una_cadena_del_modelo() -> None:
    signals = detect_model_rv(_chain(), ExecutionCosts(), RVSettings(execution="taker", min_edge_usd=1.0))
    assert signals.empty


def test_maker_publica_dentro_del_spread_con_el_margen_exacto() -> None:
    settings = RVSettings(execution="maker", min_edge_usd=5.0, max_signals=10, moneyness_band=0.2)
    signals = detect_model_rv(_chain(), ExecutionCosts(), settings)
    assert not signals.empty
    for _, signal in signals.iterrows():
        assert signal["net_edge_usd"] == pytest.approx(5.0, abs=0.05)
        assert 0.0 < signal["aggressiveness"] < 1.0
        fills = [leg["price"] for leg in signal["leg_spec"]]
        assert execution_edge(signal, fills, ExecutionCosts()) == pytest.approx(signal["net_edge_usd"], abs=1e-6)


pytest.importorskip("alpaca", reason="alpaca-py no instalado")

from fake_broker import FakeBroker, SimTime  # noqa: E402

from ita_options.config import ExecutionSettings  # noqa: E402
from ita_options.execution.ledger import Ledger, reconcile  # noqa: E402
from ita_options.execution.live_runner import LiveRunner  # noqa: E402
from ita_options.execution.router import Timing  # noqa: E402
from test_paper_execution import FakeMarket  # noqa: E402


def test_loop_con_fuente_rv_abre_y_cierra_por_convergencia(tmp_path) -> None:
    base = build_chain()
    improved = base.copy()
    # El mercado converge a favor de la posición: sube sólo la pata comprada.
    calls = improved.loc[improved["option_type"] == "c"].sort_values("strike")
    improved.loc[calls.index[2], ["bid", "ask"]] += 0.50

    def source(chain: pd.DataFrame) -> pd.DataFrame:
        calls = chain.loc[chain["option_type"] == "c"].sort_values("strike")
        lo = calls.iloc[2]
        hi = calls.iloc[3]
        legs = (
            {"kind": "option", "symbol": lo["symbol"], "qty": 1.0, "price": float(lo["ask"]),
             "strike": float(lo["strike"]), "option_type": "c", "expiration": lo["expiration"]},
            {"kind": "option", "symbol": hi["symbol"], "qty": -1.0, "price": float(hi["bid"]),
             "strike": float(hi["strike"]), "option_type": "c", "expiration": hi["expiration"]},
        )
        return pd.DataFrame([{
            "detector": "rv_crr_vertical", "underlying": "RTX", "timestamp": lo["timestamp"],
            "option_type": "c", "expiration": lo["expiration"], "strikes": "a/b", "legs": "compra",
            "gross_credit": 0.12, "commission": 1.3, "net_edge": 0.107, "net_edge_usd": 10.7,
            "leg_spec": legs, "max_spread_rel": 0.1, "symbols": "x", "min_volume": np.nan,
            "min_open_interest": np.nan,
        }])

    broker = FakeBroker()
    broker.next_close = pd.Timestamp("2026-09-03 20:00", tz="UTC")
    clock = SimTime()
    settings = ExecutionSettings(kill_file=tmp_path / "KILL", order_poll_s=1.0,
                                 max_open_strategies=2, take_profit_fraction=0.6,
                                 min_edge_usd=5.0)
    runner = LiveRunner(broker, FakeMarket([base, improved]), Ledger(), settings, dry_run=False,
                        report_dir=tmp_path / "rep", timing=Timing(sleep=clock.sleep, now=clock.now),
                        now=lambda: pd.Timestamp("2026-09-03 15:00", tz="UTC"),
                        signal_source=source)
    first = runner.run_cycle()
    assert first.entries_opened == 1
    second = runner.run_cycle()
    assert second.closes == 1
    closed = runner.ledger.strategies(states=["closed"])
    assert closed and closed[0]["exit_reason"] == "convergencia"
    assert reconcile(runner.ledger, broker).ok


# --------------------------------------------------------------------------- #
# Competencia binomial contra Black-Scholes (Parte 2)
# --------------------------------------------------------------------------- #


def _competencia(cap: int = 3):
    from ita_options.execution.ledger import Ledger
    from ita_options.execution.model_rv import CompetingModels

    ledger = Ledger()
    settings = {m: RVSettings(model=m, execution="maker", min_edge_usd=5.0, max_signals=6, moneyness_band=0.2)
                for m in ("crr", "bsm")}
    return ledger, CompetingModels(ledger, ExecutionCosts(), settings, cap_per_model=cap)


def test_competencia_atribuye_cada_senal_a_su_modelo_y_respeta_el_cupo() -> None:
    _, source = _competencia(cap=2)
    signals = source(_chain())
    models = signals["detector"].map(source.model_of)
    assert set(models) <= {"crr", "bsm"}
    assert (models.value_counts() <= 2).all()


def test_competencia_no_ejecuta_dos_veces_las_mismas_patas() -> None:
    ledger, source = _competencia(cap=6)
    signals = source(_chain())
    keys = [source.legs_key(legs) for legs in signals["leg_spec"]]
    assert len(keys) == len(set(keys))
    # Sobre una cadena del modelo, CRR y BSM eligen estructuras parecidas: hay duplicados descartados.
    assert ledger.events("senal_duplicada")


def test_competencia_alterna_la_prioridad_entre_modelos() -> None:
    ledger, source = _competencia()
    chain = _chain()
    source(chain)
    source(chain)
    import json

    priorities = [json.loads(e["payload_json"])["prioridad"] for e in ledger.events("senales_por_modelo")]
    assert priorities == ["crr", "bsm"]


def test_loop_en_vivo_con_los_dos_modelos_compitiendo(tmp_path) -> None:
    """De punta a punta: la competencia abre estructuras atribuidas a cada modelo y reconcilia."""
    from ita_options.execution.model_rv import CompetingModels

    # Vencimiento a 43 días (dentro de la ventana 21-75) y spread angosto: el filtro en
    # vivo deja operables las alas OTM, que son las que ajustan la sonrisa.
    chain = build_chain(expiries=(("2026-10-16", 43 / 365),), spread=0.20)
    broker = FakeBroker()
    broker.next_close = pd.Timestamp("2026-09-03 20:00", tz="UTC")
    clock = SimTime()
    settings = ExecutionSettings(kill_file=tmp_path / "KILL", order_poll_s=1.0, max_open_strategies=6,
                                 min_edge_usd=5.0, max_dte=75)
    ledger = Ledger()
    source = CompetingModels(
        ledger, ExecutionCosts(min_net_edge=5.0),
        {m: RVSettings(model=m, min_edge_usd=5.0, moneyness_band=0.2, max_signals=3) for m in ("crr", "bsm")},
        cap_per_model=2,
    )
    runner = LiveRunner(broker, FakeMarket([chain]), ledger, settings, dry_run=False,
                        report_dir=tmp_path / "rep", timing=Timing(sleep=clock.sleep, now=clock.now),
                        now=lambda: pd.Timestamp("2026-09-03 15:00", tz="UTC"), signal_source=source)
    report = runner.run_cycle()
    opened = runner.ledger.strategies(states=["open"])
    assert report.entries_opened == len(opened) > 0
    models = {source.model_of(s["detector"]) for s in opened}
    assert models <= {"crr", "bsm"} and len(models) == 2
    assert all(sum(1 for s in opened if source.model_of(s["detector"]) == m) <= 2 for m in models)
    assert reconcile(runner.ledger, broker).ok


# --------------------------------------------------------------------------- #
# Dividendo discreto (escrowed) en la estrategia: paso 3 de la Parte 2
# --------------------------------------------------------------------------- #


def _escrowed_chain() -> pd.DataFrame:
    """Cadena generada por el árbol escrowed con un dividendo de 2 USD a 10 días."""
    from datetime import date

    from ita_options.greeks import crr_price

    divs = ((10 / 365, 2.0),)
    rows = []
    for strike in np.arange(130.0, 171.0, 5.0):
        for flag in ("c", "p"):
            fair = crr_price(150.0, strike, 45 / 365, 0.04, 0.3, flag, divs, steps=100)
            bid, ask = round(max(fair - 0.15, 0.01), 2), round(fair + 0.15, 2)
            rows.append({"symbol": f"X{flag}{strike:g}", "underlying": "LMT", "trade_date": date(2026, 9, 1),
                         "timestamp": pd.Timestamp("2026-09-01 20:00", tz="UTC"),
                         "expiration": date(2026, 10, 16), "strike": strike, "option_type": flag,
                         "bid": bid, "ask": ask, "spread_rel": (ask - bid) / ((ask + bid) / 2),
                         "underlying_price": 150.0, "tau": 45 / 365, "dte": 45, "risk_free_rate": 0.04,
                         "dividend_yield": 0.0, "is_tradable": True, "dividend_schedule": divs})
    return pd.DataFrame(rows)


def test_con_calendario_de_dividendos_la_sonrisa_es_la_del_arbol_escrowed() -> None:
    chain = _escrowed_chain()
    smile = fit_smile(chain, "crr", RVSettings())
    assert smile is not None
    # Generada por el árbol escrowed con sigma 0,30: la sonrisa recupera la sigma plana.
    assert smile.a == pytest.approx(0.30, abs=0.01) and smile.rmse_iv < 0.01
    assert smile.forward == pytest.approx((150 - 2 * np.exp(-0.04 * 10 / 365)) * np.exp(0.04 * 45 / 365))
    assert detect_model_rv(chain, ExecutionCosts(), RVSettings(execution="taker", min_edge_usd=1.0)).empty


@pytest.mark.parametrize("model", ["crr", "bsm"])
def test_vivo_y_backtest_valuan_igual_con_dividendo_discreto(model: str) -> None:
    from ita_options.rv_backtest import precompute_valuations

    chain = _escrowed_chain()
    dividends = pd.DataFrame({"ticker": ["LMT"], "ex_dividend_date": [pd.Timestamp("2026-09-11")],
                              "declaration_date": [pd.Timestamp("2026-08-01")], "cash_amount": [2.0]})
    vals, _ = precompute_valuations(chain.drop(columns="dividend_schedule"), model,
                                    RVSettings(model=model), dividends=dividends, band=0.10)
    settings = RVSettings(model=model, execution="maker", min_edge_usd=0.01, max_signals=200, moneyness_band=0.10)
    live = detect_model_rv(chain, ExecutionCosts(min_net_edge=0.0), settings)
    assert not live.empty
    by_strikes = {(r.option_type, r.strikes, r.kind): r.model_value for r in vals.itertuples()}
    for sig in live.itertuples():
        kind = sig.detector.split("_")[-1]
        direction = 1.0 if sig.legs.startswith("compra") else -1.0
        assert direction * sig.model_value == pytest.approx(by_strikes[(sig.option_type, sig.strikes, kind)], abs=1e-9)
