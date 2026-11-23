"""Descarga 10 años de precios diarios (cierre y cierre ajustado) de Yahoo para el estrés.

``run_stress.py`` calibra con esto la correlación entre subyacentes y los
saltos de Merton, y toma las repeticiones históricas (COVID, aranceles).
Necesita red: se corre en la máquina local, no en el contenedor.

Uso::

    python scripts/download_history.py --out data/underlying_bars/historia_10y.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def fetch(ticker: str, years: int = 10) -> pd.DataFrame:
    """Serie diaria de ``ticker``: ``date``, ``close``, ``adjclose``, ``symbol``."""
    response = requests.get(URL.format(ticker=ticker),
                            params={"range": f"{years}y", "interval": "1d", "events": "div,split"},
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    result = response.json()["chart"]["result"][0]
    quotes = result["indicators"]
    frame = pd.DataFrame({
        "date": pd.to_datetime(result["timestamp"], unit="s").date,
        "close": quotes["quote"][0]["close"],
        "adjclose": quotes["adjclose"][0]["adjclose"],
    })
    frame["symbol"] = ticker
    return frame.dropna()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=["RTX", "BA", "LMT"])
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--out", default=str(ROOT / "data" / "underlying_bars" / "historia_10y.csv"))
    args = parser.parse_args(argv)
    data = pd.concat([fetch(t, args.years) for t in args.tickers], ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)
    print(data.groupby("symbol")["date"].agg(["min", "max", "count"]))


if __name__ == "__main__":
    main()
