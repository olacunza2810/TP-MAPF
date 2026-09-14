"""La alineación tiene que funcionar sobre datos releídos del data lake.

``enrich`` no recibe los quotes recién bajados sino los que ``ParquetStore``
devuelve, que ya pasaron por ``enforce_schema``: columnas derivadas vacías,
``underlying`` como StringDtype y timestamps en otra resolución. Este test cubre
ese camino, que la demo no ejercita porque alinea antes de escribir.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from ita_options.enrich import align_underlying, compute_mid_and_spread
from ita_options.storage import ParquetStore


def test_alineacion_sobre_quotes_releidos_del_parquet(tmp_path) -> None:
    stamps = pd.date_range("2025-03-03 14:35", periods=3, freq="5min", tz="UTC")
    quotes = pd.DataFrame({
        "symbol": "O:RTX250417C00120000",
        "underlying": "RTX",
        "timestamp": stamps,
        "observed_at": stamps,
        "expiration": pd.Timestamp("2025-04-17"),
        "strike": 120.0,
        "option_type": "c",
        "bid": [4.0, 4.1, 4.2],
        "ask": [4.2, 4.3, 4.4],
        "trade_date": pd.Timestamp("2025-03-03"),
    })
    store = ParquetStore(tmp_path, "option_quotes")
    store.write(quotes)
    reread = store.read(underlyings=["RTX"])

    bars = pd.DataFrame({
        "timestamp": pd.date_range("2025-03-03 14:30", periods=20, freq="1min", tz="UTC"),
        "symbol": "RTX",
        "close": [120.0 + i for i in range(20)],
    })
    aligned = align_underlying(compute_mid_and_spread(reread), bars,
                               bar_duration=timedelta(minutes=1))

    assert aligned.columns.is_unique
    assert aligned["underlying_price"].notna().all()
    # Marca 14:35: la última barra disponible es la de 14:34, que cierra 14:35.
    first = aligned.sort_values("timestamp").iloc[0]
    assert first["underlying_price"] == 124.0
    assert (aligned["alignment_lag_s"] >= 0).all()
