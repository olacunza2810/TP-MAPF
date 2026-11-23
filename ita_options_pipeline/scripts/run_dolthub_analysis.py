r"""Análisis completo sobre bid/ask observado (DoltHub), marzo a noviembre de 2026.

Responde a los puntos 1, 2 y 4 de la devolución de la entrega parcial:

1. **Bid/ask real.** Cura la foto diaria de DoltHub y corre el backtest de
   arbitraje ejecutando cada pata contra el bid o el ask observados.
2. **Paridad con dividendos discretos y fondeo.** Mide cuántas violaciones de
   paridad aporta cada supuesto (yield continuo, dividendo discreto, banda de
   spot) y corre el backtest con carry: intereses, préstamo, dividendos
   realizados y asignación anticipada.
4. **Walk-forward IS/OOS.** Calibra los hiperparámetros de la estrategia de
   valor relativo en una ventana creciente que arranca con marzo-junio y la
   evalúa mes a mes fuera de muestra, para el binomial (CRR) y para
   Black-Scholes.

Etapas (``--steps``), cada una guarda su salida y la siguiente la reutiliza:

=============  ============================================================
``curate``     ``data/dolthub/curated.parquet``, desfase de spot, filtros
``arbitrage``  embudo de detectores y backtest de arbitraje con carry
``valuations`` sonrisas y valores de modelo (CRR y BSM), cacheados
``walkforward`` grilla in-sample, selección y evaluación fuera de muestra
=============  ============================================================

Uso (desde ``ita_options_pipeline``)::

    python scripts/run_dolthub_analysis.py --out reportes_dolthub
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ita_options.arbitrage import ExecutionCosts, run_all_detectors  # noqa: E402
from ita_options.backtest import ArbitrageBacktester  # noqa: E402
from ita_options.daily import daily_strategy_params  # noqa: E402
from ita_options.dolthub import DoltHubData, curate  # noqa: E402
from ita_options.evaluation import compute_metrics, deflated_sharpe_ratio, signal_funnel, trade_breakdown  # noqa: E402
from ita_options.rv_backtest import RVParams, default_grid, precompute_valuations, simulate, summarize  # noqa: E402

_LOG = logging.getLogger("dolthub_analysis")
TICKERS = ("RTX", "BA", "LMT")
STEPS = ("curate", "arbitrage", "valuations", "walkforward")


# --------------------------------------------------------------------------- #
# Curación
# --------------------------------------------------------------------------- #


def step_curate(data_root: Path, out: Path) -> None:
    data = DoltHubData.load(data_root, TICKERS)
    curated, under, report, gaps = curate(data)
    curated.to_parquet(data_root / "dolthub" / "curated.parquet", index=False)
    under.to_parquet(data_root / "dolthub" / "underlying_bidask.parquet", index=False)
    gaps.to_csv(out / "desfase_spot.csv", index=False)
    report.to_frame().to_csv(out / "filtros_liquidez.csv", index=False)
    summary = {
        "filas": int(len(curated)),
        "explotables": int(curated["is_tradable"].sum()),
        "supervivencia": float(curated["is_tradable"].mean()),
        "ruedas": int(curated["trade_date"].nunique()),
        "desde": str(curated["trade_date"].min()), "hasta": str(curated["trade_date"].max()),
        "filas_por_subyacente": curated["underlying"].value_counts().to_dict(),
        "spread_rel_mediano": float(curated["spread_rel"].median()),
        "exclusiones": dict(report.exclusions),
        "desfase_spot_abs_mediano_pct": float(gaps["spot_gap_pct"].abs().median()),
        "desfase_spot_abs_p90_pct": float(gaps["spot_gap_pct"].abs().quantile(0.9)),
        "dividendos": data.dividends.astype(str).to_dict(orient="records"),
    }
    (out / "resumen_datos.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _LOG.info("Curado: %s", {k: summary[k] for k in ("filas", "explotables", "ruedas")})


def _load_curated(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, DoltHubData]:
    curated = pd.read_parquet(data_root / "dolthub" / "curated.parquet")
    curated["trade_date"] = pd.to_datetime(curated["trade_date"]).dt.date
    under = pd.read_parquet(data_root / "dolthub" / "underlying_bidask.parquet")
    return curated, under, DoltHubData.load(data_root, TICKERS)


# --------------------------------------------------------------------------- #
# Arbitraje: embudo y backtest con carry
# --------------------------------------------------------------------------- #


def parity_variants(curated: pd.DataFrame) -> pd.DataFrame:
    """Violaciones por día: los detectores sin acción una vez, la paridad bajo tres supuestos.

    Las variantes aíslan cada corrección de la paridad. Los otros detectores no
    dependen de dividendos ni del spot, así que se corren una sola vez.
    """
    from ita_options.arbitrage import (
        detect_butterfly_convexity, detect_calendar, detect_put_call_parity,
        detect_vertical_monotonicity, detect_vertical_spread_bound,
    )

    costs = ExecutionCosts()
    tradable = curated.loc[curated["is_tradable"]]
    variants = {
        "yield_continuo_sin_banda": lambda c: c.drop(columns=["pv_dividends", "spot_uncertainty"]).assign(
            dividend_yield=c["underlying"].map({"RTX": 0.016, "LMT": 0.026}).fillna(0.0)),
        "dividendo_discreto_sin_banda": lambda c: c.drop(columns=["spot_uncertainty"]),
        "dividendo_discreto_con_banda": lambda c: c,
    }
    rows = []

    def record(name: str, day, found: pd.DataFrame) -> None:
        for det, grp in (found.groupby("detector") if not found.empty else []):
            rows.append({"variante": name, "trade_date": day, "detector": det, "n": len(grp),
                         "edge_mediano": float(grp["net_edge_usd"].median()),
                         "edge_total": float(grp["net_edge_usd"].sum())})

    for day, chain in tradable.groupby("trade_date"):
        others = [f(chain, costs) for f in (detect_vertical_monotonicity, detect_vertical_spread_bound,
                                            detect_butterfly_convexity, detect_calendar)]
        others = pd.concat([o for o in others if not o.empty]) if any(not o.empty for o in others) else pd.DataFrame()
        record("sin_paridad", day, others)
        for name, build in variants.items():
            record(name, day, detect_put_call_parity(build(chain), costs))
    return pd.DataFrame(rows)


def _arbitrage_backtest(args: tuple[str, int, str]) -> dict[str, object]:
    """Una corrida del backtest de arbitraje (se lanza en un proceso aparte)."""
    data_root, lag, out = Path(args[0]), args[1], Path(args[2])
    curated, under, data = _load_curated(data_root)
    params = daily_strategy_params(
        lag_sessions=lag, execution_price="quote", fresh_quotes_only=True,
        min_volume=0.0, min_open_interest=0.0, max_spread_rel=0.25,
        stop_loss_usd=2000.0, max_dte=180,
    )
    result = ArbitrageBacktester(curated, under.rename(columns={"trade_date": "date"}),
                                 ExecutionCosts(), data.carry_model()).run(params)
    tag = f"t{lag}"
    result.trades.to_csv(out / f"arbitraje_operaciones_{tag}.csv", index=False)
    signal_funnel(result).to_csv(out / f"arbitraje_embudo_{tag}.csv")
    if not result.trades.empty:
        trade_breakdown(result, "detector").to_csv(out / f"arbitraje_por_detector_{tag}.csv")
        trade_breakdown(result, "exit_reason").to_csv(out / f"arbitraje_por_salida_{tag}.csv")
    diag = {k: v for k, v in result.diagnostics.items() if k != "by_detector"}
    return {"ejecucion": f"foto de t+{lag}", "operaciones": len(result.trades),
            "pnl": result.total_pnl, "hit_rate": result.hit_rate, **diag}


def step_arbitrage(data_root: Path, out: Path) -> None:
    from concurrent.futures import ProcessPoolExecutor

    curated, _, _ = _load_curated(data_root)
    with ProcessPoolExecutor(max_workers=2) as pool:
        backtests = [pool.submit(_arbitrage_backtest, (str(data_root), lag, str(out))) for lag in (1, 2)]
        started = time.monotonic()
        variants = parity_variants(curated)
        variants.to_csv(out / "arbitraje_violaciones_por_dia.csv", index=False)
        table = (variants.groupby(["variante", "detector"])["n"].sum().unstack(fill_value=0)
                 if not variants.empty else pd.DataFrame())
        table.to_csv(out / "arbitraje_violaciones_por_variante.csv")
        _LOG.info("Variantes (%.0fs):\n%s", time.monotonic() - started, table)
        rows = [f.result() for f in backtests]
    pd.DataFrame(rows).to_csv(out / "arbitraje_resumen.csv", index=False)
    _LOG.info("Backtests de arbitraje:\n%s", pd.DataFrame(rows).to_string(index=False))


# --------------------------------------------------------------------------- #
# Valor relativo: valuaciones y walk-forward
# --------------------------------------------------------------------------- #


def step_valuations(data_root: Path, out: Path) -> None:
    curated, _, _ = _load_curated(data_root)
    tradable = curated.loc[curated["is_tradable"]]
    for model in ("crr", "bsm"):
        started = time.monotonic()
        vals, smiles = precompute_valuations(tradable, model)
        vals.to_pickle(data_root / "dolthub" / f"valuaciones_{model}.pkl")
        smiles.to_csv(out / f"sonrisas_{model}.csv", index=False)
        _LOG.info("Valuaciones %s: %d filas en %.0fs", model, len(vals), time.monotonic() - started)


def _months(days: list[date]) -> list[tuple[date, date]]:
    frame = pd.Series(days)
    out = []
    for _, grp in frame.groupby([frame.map(lambda d: d.year), frame.map(lambda d: d.month)]):
        out.append((grp.min(), grp.max()))
    return out


def _objective(metrics: dict[str, float]) -> tuple[float, float]:
    """Criterio de selección in-sample: P&L total, con un mínimo de 5 operaciones."""
    if metrics["trades"] < 5:
        return (-np.inf, -np.inf)
    return (metrics["pnl"], metrics.get("sharpe") if np.isfinite(metrics.get("sharpe", np.nan)) else -np.inf)


def step_walkforward(data_root: Path, out: Path, first_oos: str) -> None:
    curated, _, data = _load_curated(data_root)
    spots = data.bars.copy()
    spots["date"] = pd.to_datetime(spots["date"]).dt.date
    days = sorted(curated["trade_date"].unique())
    months = _months(days)
    oos_months = [(a, b) for a, b in months if a >= date.fromisoformat(first_oos)]
    grid = default_grid()
    fold_rows, oos_trades, full_rows = [], [], []
    for model in ("crr", "bsm"):
        vals = pd.read_pickle(data_root / "dolthub" / f"valuaciones_{model}.pkl")
        # Muestra completa: todas las configuraciones, para el DSR.
        for i, params in enumerate(grid):
            trades, daily, funnel = simulate(vals, curated, spots, params)
            full_rows.append({"modelo": model, "config": i, **asdict(params), **summarize(trades, daily), **funnel})
        for oos_start, oos_end in oos_months:
            is_start, is_end = days[0], max(d for d in days if d < oos_start)
            scored = []
            for i, params in enumerate(grid):
                trades, daily, _ = simulate(vals, curated, spots, params, start=is_start, end=is_end, data_end=is_end)
                scored.append((_objective(summarize(trades, daily)), i, summarize(trades, daily)))
            scored.sort(key=lambda x: x[0], reverse=True)
            (_, best, is_metrics) = scored[0]
            params = grid[best]
            trades, daily, funnel = simulate(vals, curated, spots, params, start=oos_start, end=oos_end)
            oos = summarize(trades, daily.loc[daily["trade_date"] >= oos_start])
            if not trades.empty:
                oos_trades.append(trades.assign(modelo=model, mes_oos=f"{oos_start:%Y-%m}", config=best))
            fold_rows.append({
                "modelo": model, "is_desde": is_start, "is_hasta": is_end, "oos_desde": oos_start, "oos_hasta": oos_end,
                "config_elegida": best, **{f"param_{k}": v for k, v in asdict(params).items()},
                **{f"is_{k}": v for k, v in is_metrics.items()},
                **{f"oos_{k}": v for k, v in oos.items()}, **{f"oos_{k}": v for k, v in funnel.items()},
            })
            _LOG.info("%s OOS %s: config %d, IS pnl %.0f (%d op.), OOS pnl %.0f (%d op.)", model,
                      f"{oos_start:%Y-%m}", best, is_metrics["pnl"], is_metrics["trades"], oos["pnl"], oos["trades"])
    full = pd.DataFrame(full_rows)
    for model, grp in full.groupby("modelo"):
        variance = float(np.nanvar(grp["sharpe_daily"].to_numpy(), ddof=1)) if grp["sharpe_daily"].notna().sum() > 1 else 0.0
        full.loc[grp.index, "dsr_n18"] = [
            deflated_sharpe_ratio(r.sharpe_daily, int(r.n_days), len(grid), variance,
                                  r.skew if np.isfinite(r.skew) else 0.0, r.kurt if np.isfinite(r.kurt) else 3.0)
            if np.isfinite(r.sharpe_daily) else np.nan
            for r in grp.itertuples()
        ]
    full.to_csv(out / "rv_grilla_muestra_completa.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "rv_walkforward.csv", index=False)
    if oos_trades:
        pd.concat(oos_trades, ignore_index=True).to_csv(out / "rv_operaciones_oos.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(ROOT / "reportes_dolthub"))
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--first-oos", default="2026-07-01",
                        help="Primer mes fuera de muestra; antes, cuatro meses de calibración.")
    args = parser.parse_args(argv)
    data_root, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if "curate" in args.steps:
        step_curate(data_root, out)
    if "arbitrage" in args.steps:
        step_arbitrage(data_root, out)
    if "valuations" in args.steps:
        step_valuations(data_root, out)
    if "walkforward" in args.steps:
        step_walkforward(data_root, out, args.first_oos)


if __name__ == "__main__":
    main()
