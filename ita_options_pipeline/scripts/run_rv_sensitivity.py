r"""Sensibilidad exploratoria: más órdenes por rueda (declarada como post hoc).

La grilla declarada (``default_grid``) publica 2 órdenes por rueda, y con una
tasa de llenado del 5% eso deja unas 20 operaciones en nueve meses: muy poca
potencia estadística. Esta corrida se agregó **después** de ver ese resultado,
con 10 órdenes por rueda, para tener más llenados y estudiar la selección
adversa. Por eso no se usa para elegir hiperparámetros y sus 36 configuraciones
se suman al N del Sharpe deflactado (N = 18 + 18 + 36).

Uso::

    python scripts/run_rv_sensitivity.py --out reportes_dolthub
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ita_options.dolthub import DoltHubData  # noqa: E402
from ita_options.evaluation import deflated_sharpe_ratio  # noqa: E402
from ita_options.rv_backtest import default_grid, simulate, summarize  # noqa: E402

N_TRIALS_TOTAL = 18 * 2 + 18 * 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(ROOT / "reportes_dolthub"))
    parser.add_argument("--max-new", type=int, default=10)
    args = parser.parse_args()
    root, out = Path(args.data_root), Path(args.out)
    curated = pd.read_parquet(root / "dolthub" / "curated.parquet")
    curated["trade_date"] = pd.to_datetime(curated["trade_date"]).dt.date
    spots = DoltHubData.load(root, ("RTX", "BA", "LMT")).bars.copy()
    spots["date"] = pd.to_datetime(spots["date"]).dt.date
    rows, trades_all = [], []
    for model in ("crr", "bsm"):
        vals = pd.read_pickle(root / "dolthub" / f"valuaciones_{model}.pkl")
        for i, base in enumerate(default_grid()):
            params = replace(base, max_new_per_day=args.max_new, max_open=30)
            trades, daily, funnel = simulate(vals, curated, spots, params)
            rows.append({"modelo": model, "config": i, **asdict(params), **summarize(trades, daily), **funnel})
            if not trades.empty:
                trades_all.append(trades.assign(modelo=model, config=i))
    table = pd.DataFrame(rows)
    variance = float(np.nanvar(table["sharpe_daily"], ddof=1))
    table["dsr_n72"] = [
        deflated_sharpe_ratio(r.sharpe_daily, int(r.n_days), N_TRIALS_TOTAL, variance,
                              r.skew if np.isfinite(r.skew) else 0.0, r.kurt if np.isfinite(r.kurt) else 3.0)
        if np.isfinite(r.sharpe_daily) else np.nan for r in table.itertuples()
    ]
    table.to_csv(out / "rv_sensibilidad_10_ordenes.csv", index=False)
    pd.concat(trades_all, ignore_index=True).to_csv(out / "rv_sensibilidad_operaciones.csv", index=False)
    print(table[["modelo", "config", "margin_usd", "take_profit", "stop_fraction", "trades", "pnl",
                 "hit_rate", "sharpe", "dsr_n72", "ordenes", "llenadas"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
