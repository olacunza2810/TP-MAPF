"""Tests del motor de backtest y de la evaluación estadística."""

from __future__ import annotations

from datetime import timedelta

import pytest

from ita_options.arbitrage import ExecutionCosts
from ita_options.backtest import (
    ArbitrageBacktester,
    LookaheadError,
    PointInTimeView,
    StrategyParams,
)
from ita_options.evaluation import (
    compute_metrics,
    deflated_sharpe_ratio,
    split_in_sample,
)

COSTS = ExecutionCosts(min_net_edge=1.0)


def test_la_vista_no_deja_mirar_el_futuro(timeseries_chain) -> None:
    """Pedir un corte posterior al reloj levanta excepción."""
    quotes, underlying = timeseries_chain
    stamps = sorted(quotes["timestamp"].unique())
    view = PointInTimeView(quotes, underlying)
    view.advance_to(stamps[1])
    with pytest.raises(LookaheadError):
        view.chain_at(stamps[5])


def test_el_reloj_no_retrocede(timeseries_chain) -> None:
    """El backtest sólo avanza en el tiempo."""
    quotes, underlying = timeseries_chain
    stamps = sorted(quotes["timestamp"].unique())
    view = PointInTimeView(quotes, underlying)
    view.advance_to(stamps[3])
    with pytest.raises(LookaheadError):
        view.advance_to(stamps[0])


def test_la_latencia_cambia_el_resultado(timeseries_chain) -> None:
    """Con más demora entre señal y orden, la oportunidad se evapora.

    Es el resultado más importante del backtest: la misma señal, sobre los
    mismos datos, deja de ser rentable sólo por llegar tarde. Cualquier
    backtest de arbitraje que no muestre esta sensibilidad está asumiendo
    latencia cero.
    """
    quotes, underlying = timeseries_chain
    engine = ArbitrageBacktester(quotes, underlying, COSTS)

    fast = compute_metrics(
        engine.run(
            StrategyParams(
                min_net_edge=1.0,
                execution_lag=timedelta(minutes=5),
                stop_loss_usd=None,
            )
        )
    )
    slow = compute_metrics(
        engine.run(
            StrategyParams(
                min_net_edge=1.0,
                execution_lag=timedelta(minutes=20),
                stop_loss_usd=None,
            )
        )
    )
    assert fast.total_pnl > slow.total_pnl


def test_el_split_no_se_solapa(timeseries_chain) -> None:
    """El out-of-sample empieza estrictamente después del in-sample."""
    quotes, _ = timeseries_chain
    in_sample, out_sample = split_in_sample(quotes, 0.5)
    assert in_sample["timestamp"].max() < out_sample["timestamp"].min()


def test_el_sharpe_deflactado_castiga_la_busqueda() -> None:
    """A igual Sharpe, más configuraciones probadas implican menos evidencia."""
    scores = [
        deflated_sharpe_ratio(
            sharpe=0.113, n_obs=252, n_trials=trials, sharpe_variance=0.02
        )
        for trials in (1, 20, 200)
    ]
    assert scores[0] > scores[1] > scores[2]
    assert scores[0] > 0.95 and scores[2] < 0.05
