r"""Simulación histórica del sistema de producción sobre toda la historia de DoltHub.

Uso::

    python scripts/run_replay.py --data-root data_hist --out reportes_replay --workers 8

Dos etapas:

1. **Señales** (``senales.pkl``, cacheadas): para cada foto y cada modelo,
   :func:`detect_model_rv` con la configuración de ``paper_daily.ps1``
   (``min_dte`` 21, ``max_dte`` 75, edge neto mínimo USD 10, dividendo discreto
   escrowed, spot implícito por paridad put-call).
2. **Cuenta** (:func:`ita_options.replay.replay`), una vez por modo de cobertura.

Salidas en ``--out``: ``replay_diario_<modo>.csv``, ``replay_estructuras_<modo>.csv``,
``replay_resumen.csv`` (por modo y modelo), ``replay_por_anio.csv``,
``replay_m4.csv`` (diferencia de P&L medio entre modelos con su error estándar)
y ``replay_equity.png``.
"""

from __future__ import annotations

import argparse
import logging
import math
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ita_options.arbitrage import ExecutionCosts  # noqa: E402
from ita_options.dolthub import attach_dividend_schedule, merge_dividends  # noqa: E402
from ita_options.execution.model_rv import RVSettings, detect_model_rv  # noqa: E402
from ita_options.replay import ReplayRules, replay, trade_stats  # noqa: E402

_LOG = logging.getLogger("replay")
MODELS = ("crr", "bsm")
MODES = ("sin_cobertura", "delta", "delta_gamma")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#ecebe6"
COLORS = {"crr": "#2a6fdb", "bsm": "#d9822b", "total": "#0b0b0b"}


def load(data_root: Path) -> tuple[dict, pd.DataFrame, dict]:
    """Cadenas por foto (spot implícito y dividendos escrowed), dividendos y spots."""
    curated = pd.read_parquet(data_root / "dolthub" / "curated.parquet")
    curated["trade_date"] = pd.to_datetime(curated["trade_date"]).dt.date
    curated["expiration"] = pd.to_datetime(curated["expiration"]).dt.date
    spot = curated.groupby(["underlying", "trade_date"])["implied_spot"].first().astype(float)
    close = curated.groupby(["underlying", "trade_date"])["underlying_price"].first().astype(float)
    spot = spot.fillna(close)
    curated["underlying_price"] = curated.set_index(["underlying", "trade_date"]).index.map(spot)
    curated = curated.loc[curated["underlying_price"].notna()]
    dividends = merge_dividends(pd.read_parquet(data_root / "dolthub" / "dividends.parquet"),
                                data_root / "dolthub" / "dividends_polygon.parquet")
    curated = attach_dividend_schedule(curated, dividends)
    chains = {d: g.reset_index(drop=True) for d, g in curated.groupby("trade_date")}
    spots = spot.to_dict()
    bars = data_root.parent / "data" / "underlying_bars" / "historia_10y.csv"
    for path in (bars, Path("data/underlying_bars/historia_10y.csv")):
        if path.exists():
            hist = pd.read_csv(path, parse_dates=["date"])
            for sym, d, c in zip(hist["symbol"], hist["date"].dt.date, hist["close"]):
                spots.setdefault((sym, d), float(c))  # sólo para liquidar vencimientos sin foto
            break
    return chains, dividends, spots


def _signals_day(chain: pd.DataFrame) -> dict[str, pd.DataFrame]:
    costs = ExecutionCosts(min_net_edge=10.0)
    out = {}
    for m in MODELS:
        try:
            out[m] = detect_model_rv(chain, costs, RVSettings(model=m, min_dte=21, max_dte=75, min_edge_usd=10.0))
        except Exception as exc:  # una foto rota no debe tirar la corrida
            _LOG.warning("%s %s: %s", chain["trade_date"].iloc[0], m, exc)
            out[m] = pd.DataFrame()
    return out


def signals(chains: dict, workers: int, cache: Path) -> dict:
    done: dict = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    todo = [d for d in sorted(chains) if d not in done]
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (d, sig) in enumerate(zip(todo, pool.map(_signals_day, [chains[d] for d in todo], chunksize=2)), 1):
            done[d] = sig
            if i % 25 == 0 or i == len(todo):
                _LOG.info("señales %d/%d (%.0fs)", i, len(todo), time.time() - start)
                cache.write_bytes(pickle.dumps(done))
    return done


def m4_test(trades: pd.DataFrame) -> dict:
    """Diferencia de P&L medio por estructura cerrada (CRR − BSM) y su error estándar."""
    closed = trades.loc[trades["cerrada"].notna()]
    a, b = (closed.loc[closed["modelo"] == m, "pnl"] for m in MODELS)
    if len(a) < 2 or len(b) < 2:
        return {"n_crr": len(a), "n_bsm": len(b)}
    diff = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return {"n_crr": len(a), "n_bsm": len(b), "media_crr": a.mean(), "media_bsm": b.mean(),
            "diferencia": diff, "error_estandar": se, "z": diff / se if se else float("nan"),
            "supera_2_se": abs(diff) > 2 * se, "cumple_minimo_10": min(len(a), len(b)) >= 10}


def plot(daily: dict[str, pd.DataFrame], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(daily), figsize=(5.2 * len(daily), 3.8), sharey=True, facecolor=SURFACE)
    for ax, (mode, frame) in zip(axes, daily.items()):
        ax.set_facecolor(SURFACE)
        x = pd.to_datetime(frame["fecha"])
        for m in MODELS:
            ax.plot(x, frame[f"valor_{m}"], color=COLORS[m], lw=2, label=m.upper())
        ax.plot(x, frame["valor"], color=COLORS["total"], lw=1.2, ls="--", label="Cuenta")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(mode.replace("_", " "), color=INK, fontsize=11, loc="left")
        ax.grid(color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0].set_ylabel("P&L acumulado al mid (USD)", color=INK2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default="data_hist")
    parser.add_argument("--out", default="reportes_replay")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chains, dividends, spots = load(Path(args.data_root))
    if args.start or args.end:
        lo = pd.Timestamp(args.start or "1900-01-01").date()
        hi = pd.Timestamp(args.end or "2100-01-01").date()
        chains = {d: c for d, c in chains.items() if lo <= d <= hi}
    _LOG.info("%d fotos, %s a %s", len(chains), min(chains), max(chains))
    sigs = signals(chains, args.workers, out / "senales.pkl")
    summary, years, m4, daily_all = [], [], [], {}
    for mode in args.modes:
        t0 = time.time()
        res = replay(chains, sigs, spots, dividends, ReplayRules(hedge=mode))
        _LOG.info("cuenta %s (%.0fs)", mode, time.time() - t0)
        daily, trades = res["diario"], res["estructuras"]
        daily.to_csv(out / f"replay_diario_{mode}.csv", index=False)
        trades.to_csv(out / f"replay_estructuras_{mode}.csv", index=False)
        res["eventos"].to_csv(out / f"replay_eventos_{mode}.csv", index=False)
        daily_all[mode] = daily
        last = daily.iloc[-1]
        stats = trade_stats(trades)
        for row in stats.to_dict("records"):
            m = row["modelo"]
            curve = daily[f"valor_{m}"]
            row.update({"modo": mode, "valor_final_mid": last[f"valor_{m}"],
                        "costo_cobertura": last[f"costo_cobertura_{m}"],
                        "max_drawdown": float((curve - curve.cummax()).min()),
                        "cortes_perdida_diaria": int((res["eventos"].get("evento", pd.Series()) == "perdida_diaria").sum())})
            summary.append(row)
        closed = trades.loc[trades["cerrada"].notna()].copy()
        closed["anio"] = pd.to_datetime(closed["cerrada"]).dt.year
        for (y, m), g in closed.groupby(["anio", "modelo"]):
            years.append({"modo": mode, "anio": y, "modelo": m, "cerradas": len(g), "pnl": g["pnl"].sum(),
                          "acierto": (g["pnl"] > 0).mean()})
        m4.append({"modo": mode, **m4_test(trades)})
    pd.DataFrame(summary).to_csv(out / "replay_resumen.csv", index=False)
    pd.DataFrame(years).to_csv(out / "replay_por_anio.csv", index=False)
    pd.DataFrame(m4).to_csv(out / "replay_m4.csv", index=False)
    plot(daily_all, out / "replay_equity.png")
    print(pd.DataFrame(summary).to_string())
    print(pd.DataFrame(m4).to_string())


if __name__ == "__main__":
    main()
