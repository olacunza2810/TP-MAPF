"""La línea de comandos acepta ``--tickers`` en cualquier posición."""

from __future__ import annotations

import pytest

from ita_options.pipeline import parse_args


@pytest.mark.parametrize(
    "argv",
    [
        ["--tickers", "RTX", "BA", "LMT", "enrich-daily", "--price-source", "close"],
        ["--tickers", "RTX", "BA", "LMT", "--", "enrich-daily", "--price-source", "close"],
        ["--tickers", "RTX,BA,LMT", "enrich-daily", "--price-source", "close"],
        ["enrich-daily", "--tickers", "RTX", "BA", "LMT", "--price-source", "close"],
        ["--tickers", "rtx", "ba", "lmt", "--data-root", "data", "enrich-daily",
         "--price-source", "close"],
    ],
)
def test_tickers_no_se_come_el_subcomando(argv: list[str]) -> None:
    args = parse_args(argv)
    assert args.command == "enrich-daily"
    assert args.tickers == ["RTX", "BA", "LMT"]
    assert args.price_source == "close"


def test_opciones_del_backtest_diario() -> None:
    args = parse_args(["--tickers", "RTX", "BA", "LMT", "backtest-daily",
                       "--lag-sessions", "1", "2", "--no-edge-check",
                       "--report-dir", "reportes"])
    assert args.tickers == ["RTX", "BA", "LMT"]
    assert args.lag_sessions == [1, 2]
    assert args.edge_check is False
    assert args.report_dir == "reportes"
    assert args.min_edge == 5.0 and args.stop_loss == 500.0
    assert args.execute_at == "open"
    assert parse_args(["backtest-daily", "--execute-at", "close"]).execute_at == "close"


def test_tickers_por_defecto() -> None:
    assert parse_args(["backtest-daily"]).tickers == ["RTX", "BA"]
