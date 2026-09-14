r"""Grilla factorial definitiva del modo diario sobre los datos reales de Polygon.

Factores (2 × 2 × 2 × 2 = 16 configuraciones):

- fuente de precio: ``close`` y ``vwap``;
- spread relativo supuesto: 2% y 5%;
- latencia: apertura de t+1 y apertura de t+2 (señal al cierre de t);
- kill-switch por posición: 1.500 y 2.000 USD de pérdida no realizada.

Configuración base, fijada antes de correr: ``close``, 2%, t+1, 2.000 USD. Los
niveles del kill-switch salen de una regla de exposición (≈4% y ≈5% del
nocional medio de una conversión, ~38.000 USD) y no de optimizar el P&L.

Además de las métricas, calcula:

- **Sharpe deflactado** con N = 16 (la grilla) y con N = ``--trials-total``
  (todas las configuraciones evaluadas en el proyecto, incluidas las previas).
- **Ajuste estimado de la paridad put-call** por dos costos que el backtest no
  contabiliza: el fondeo del nocional (intereses pagados en conversiones,
  cobrados en conversiones reversas) y los dividendos con fecha ex dentro de la
  tenencia (cobrados con la acción comprada, pagados con la vendida). Es una
  estimación con tasa plana; la dirección de cada operación se infiere del
  signo de la caja de entrada.

Uso::

    python scripts/run_daily_grid.py --out reportes_final
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ita_options.arbitrage import ExecutionCosts  # noqa: E402
from ita_options.config import DailyBarAssumptions, LiquidityThresholds  # noqa: E402
from ita_options.daily import (  # noqa: E402
    daily_strategy_params,
    run_backtest_daily,
    run_enrich_daily,
)
from ita_options.evaluation import (  # noqa: E402
    compute_metrics,
    deflated_sharpe_ratio,
    signal_funnel,
    trade_breakdown,
)

SOURCES = ("close", "vwap")
SPREADS = (0.02, 0.05)
LAGS = (1, 2)
KILL_SWITCHES = (1500.0, 2000.0)
BASE = ("close", 0.02, 1, 2000.0)
PERIODS_PER_YEAR = 252.0


def parity_cost_estimate(
    trades: pd.DataFrame, dividends: pd.DataFrame, rate: float
) -> tuple[float, float]:
    """Fondeo y dividendos no contabilizados en las operaciones de paridad.

    Returns:
        ``(ajuste_fondeo, ajuste_dividendos)`` en USD, con signo: negativo
        reduce el P&L.
    """
    parity = trades.loc[trades["detector"] == "put_call_parity"]
    if parity.empty:
        return 0.0, 0.0
    opened = pd.to_datetime(parity["opened_at"], utc=True)
    closed = pd.to_datetime(parity["closed_at"], utc=True)
    days = (closed - opened).dt.total_seconds() / 86400.0
    conversion = parity["entry_cash"] < 0  # compra la acción
    direction = np.where(conversion, 1.0, -1.0)
    funding = float(np.sum(-parity["entry_cash"].abs() * rate * days / 365.0 * direction))

    dividend_total = 0.0
    if not dividends.empty:
        ex_dates = pd.to_datetime(dividends["ex_dividend_date"]).dt.tz_localize("UTC")
        for underlying, start, end, sign in zip(
            parity["underlying"], opened, closed, direction, strict=True
        ):
            mask = (dividends["ticker"] == underlying) & (ex_dates > start) & (ex_dates <= end)
            dividend_total += 100.0 * float(dividends.loc[mask, "cash_amount"].sum()) * sign
    return funding, dividend_total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--tickers", nargs="+", default=["RTX", "BA", "LMT"])
    parser.add_argument("--out", type=Path, default=ROOT / "reportes_final")
    parser.add_argument("--min-edge", type=float, default=5.0)
    parser.add_argument("--rate", type=float, default=0.0425,
                        help="Tasa plana para estimar el fondeo de la paridad.")
    parser.add_argument("--trials-total", type=int, default=35,
                        help="Configuraciones evaluadas en todo el proyecto, para el DSR.")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    dividends_path = args.data_root / "polygon" / "dividends.parquet"
    dividends = pd.read_parquet(dividends_path) if dividends_path.exists() else pd.DataFrame()
    rows: list[dict[str, object]] = []
    data_summary: dict[str, object] = {}

    for source, spread in itertools.product(SOURCES, SPREADS):
        tick = time.monotonic()
        frame, report = run_enrich_daily(
            args.data_root, args.tickers,
            liquidity=LiquidityThresholds.for_daily_bars(),
            assumptions=DailyBarAssumptions(price_source=source,
                                            assumed_relative_spread=spread),
        )
        key = f"{source}_spread{spread:g}"
        iv = frame["iv_american"].dropna()
        data_summary[key] = {
            "filas": len(frame),
            "contratos": int(frame["symbol"].nunique()),
            "filas_por_subyacente": frame["underlying"].value_counts().to_dict(),
            "desde": str(pd.to_datetime(frame["trade_date"]).min().date()),
            "hasta": str(pd.to_datetime(frame["trade_date"]).max().date()),
            "operables": report.surviving_rows,
            "supervivencia": report.survival_rate,
            "iv_invertidas": int(len(iv)),
            "iv_mediana": float(iv.median()) if len(iv) else None,
            "exclusiones": report.exclusions,
        }
        report.to_frame().to_csv(args.out / f"filters_{key}.csv", index=False)
        print(f"[enrich {key}] {len(frame)} filas, {report.surviving_rows} operables "
              f"({report.survival_rate:.1%}) en {time.monotonic() - tick:.0f}s", flush=True)

        for lag, kill in itertools.product(LAGS, KILL_SWITCHES):
            tick = time.monotonic()
            params = daily_strategy_params(
                lag, min_net_edge=args.min_edge, stop_loss_usd=kill,
                execution_price="open",
            )
            result = run_backtest_daily(args.data_root, args.tickers, params,
                                        ExecutionCosts(min_net_edge=args.min_edge))
            metrics = compute_metrics(result, PERIODS_PER_YEAR)
            tag = f"{source}_spread{spread:g}_open_t{lag}_ks{kill:g}"
            funnel = signal_funnel(result)
            result.trades.to_csv(args.out / f"trades_{tag}.csv", index=False)
            result.equity.to_csv(args.out / f"equity_{tag}.csv", index=False)
            funnel.to_csv(args.out / f"funnel_{tag}.csv", index=False)
            trade_breakdown(result, "detector").to_csv(args.out / f"by_detector_{tag}.csv", index=False)
            trade_breakdown(result, "exit_reason").to_csv(args.out / f"by_exit_{tag}.csv", index=False)

            exits = (result.trades["exit_reason"].value_counts().to_dict()
                     if not result.trades.empty else {})
            funding, dividend = parity_cost_estimate(result.trades, dividends, args.rate)
            diagnostics = result.diagnostics
            rows.append({
                "config": tag,
                "base": (source, spread, lag, kill) == BASE,
                "price_source": source, "assumed_spread": spread,
                "execution": f"open t+{lag}", "kill_switch_usd": kill,
                "n_trades": metrics.n_trades, "total_pnl": metrics.total_pnl,
                "hit_rate": metrics.hit_rate, "edge_capture": metrics.edge_capture,
                "sharpe": metrics.sharpe, "max_drawdown": metrics.max_drawdown,
                "sharpe_period": metrics.sharpe / np.sqrt(PERIODS_PER_YEAR),
                "n_periods": metrics.n_periods, "skew": metrics.skew,
                "kurtosis": metrics.kurtosis,
                "exits_expiry": exits.get("expiry", 0),
                "exits_kill_switch": exits.get("stop_loss", 0),
                "exits_end_of_data": exits.get("end_of_data", 0),
                **{k: diagnostics.get(k, 0) for k in (
                    "signals", "queued", "filled", "rejected_edge_gone",
                    "rejected_no_quote", "rejected_capacity", "pending_at_end")},
                "parity_funding_adj": funding,
                "parity_dividend_adj": dividend,
                "pnl_adjusted_estimate": metrics.total_pnl + funding + dividend,
            })
            print(f"  [{tag}] {metrics.n_trades} ops | P&L {metrics.total_pnl:,.0f} | "
                  f"hit {metrics.hit_rate:.2f} | Sharpe {metrics.sharpe:.2f} | "
                  f"kill-switch {exits.get('stop_loss', 0)} | {time.monotonic() - tick:.0f}s",
                  flush=True)

    summary = pd.DataFrame(rows)
    finite = summary["sharpe_period"].replace([np.inf, -np.inf], np.nan).dropna()
    variance = float(finite.var(ddof=1)) if len(finite) > 1 else 0.0
    for trials in sorted({len(summary), args.trials_total}):
        summary[f"dsr_n{trials}"] = [
            deflated_sharpe_ratio(
                sharpe=float(r.sharpe_period), n_obs=int(r.n_periods), n_trials=trials,
                sharpe_variance=variance,
                skew=float(r.skew) if np.isfinite(r.skew) else 0.0,
                kurtosis=float(r.kurtosis) if np.isfinite(r.kurtosis) else 3.0,
            )
            for r in summary.itertuples()
        ]
    summary.to_csv(args.out / "summary.csv", index=False)
    data_summary["grid"] = {
        "factores": {"price_source": SOURCES, "assumed_spread": SPREADS,
                     "lag_sessions": LAGS, "kill_switch_usd": KILL_SWITCHES},
        "base": dict(zip(("price_source", "assumed_spread", "lag_sessions",
                          "kill_switch_usd"), BASE)),
        "min_edge_usd": args.min_edge, "trials_total": args.trials_total,
        "sharpe_period_variance": variance,
    }
    (args.out / "data_summary.json").write_text(
        json.dumps(data_summary, indent=2, default=str), "utf-8")

    columns = ["config", "n_trades", "total_pnl", "hit_rate", "edge_capture", "sharpe",
               "max_drawdown", "exits_expiry", "exits_kill_switch", "rejected_edge_gone",
               "pnl_adjusted_estimate", *[c for c in summary.columns if c.startswith("dsr_")]]
    print("\n" + summary[columns].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    main()
