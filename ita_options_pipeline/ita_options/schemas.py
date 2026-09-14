"""Contrato de esquema del dataset.

Fijar el esquema de forma explícita evita que ``pyarrow`` infiera tipos distintos
entre corridas parciales, lo que rompe la lectura del dataset particionado.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

__all__ = ["OPTION_QUOTE_SCHEMA", "REQUIRED_COLUMNS", "enforce_schema"]

#: Esquema canónico. La columna ``spread_is_proxy`` distingue registros con NBBO
#: real (snapshots grabados) de registros con spread estimado (bars históricos).
OPTION_QUOTE_SCHEMA: Final[dict[str, str]] = {
    # Identificación
    "timestamp": "datetime64[ns, UTC]",
    "observed_at": "datetime64[ns, UTC]",
    "symbol": "string",
    "underlying": "string",
    "expiration": "datetime64[ns]",
    "strike": "float64",
    "option_type": "string",
    "style": "string",
    "multiplier": "float64",
    # Cotización
    "bid": "float64",
    "ask": "float64",
    "bid_size": "float64",
    "ask_size": "float64",
    "mid_price": "float64",
    "spread_abs": "float64",
    "spread_rel": "float64",
    "spread_is_proxy": "bool",
    "last_trade_price": "float64",
    "last_trade_at": "datetime64[ns, UTC]",
    # Vela diaria (modo daily: el bid/ask se sintetiza a partir de estos campos)
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "vwap": "float64",
    "bid_open": "float64",
    "ask_open": "float64",
    # Actividad
    "volume": "float64",
    "volume_prev_day": "float64",
    "open_interest": "float64",
    "open_interest_date": "datetime64[ns]",
    # Subyacente alineado
    "underlying_price": "float64",
    "underlying_bar_ts": "datetime64[ns, UTC]",
    "underlying_available_at": "datetime64[ns, UTC]",
    "alignment_lag_s": "float64",
    # Derivados
    "dte": "int32",
    "tau": "float64",
    "moneyness": "float64",
    "log_moneyness": "float64",
    "risk_free_rate": "float64",
    "dividend_yield": "float64",
    "iv_american": "float64",
    "iv_european": "float64",
    "iv_vendor": "float64",
    "vega": "float64",
    "violates_bounds": "bool",
    "is_tradable": "bool",
    # Trazabilidad
    "feed": "string",
    "source": "string",
    "trade_date": "datetime64[ns]",
}

REQUIRED_COLUMNS: Final[tuple[str, ...]] = tuple(OPTION_QUOTE_SCHEMA)


def enforce_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Reordena, completa y castea un DataFrame al esquema canónico.

    Las columnas faltantes se crean con nulos del tipo correcto; las sobrantes se
    descartan. Esto hace que los dos modos de ingesta produzcan artefactos
    intercambiables aunque no llenen exactamente los mismos campos.

    Args:
        frame: DataFrame crudo.

    Returns:
        DataFrame con exactamente las columnas de :data:`OPTION_QUOTE_SCHEMA`.
    """
    out = frame.copy()
    # Las columnas de auditoría del filtrado no están en el contrato fijo porque
    # su nombre depende de los criterios configurados, pero deben sobrevivir al
    # round-trip: son la traza que justifica cada exclusión.
    audit_columns = [c for c in out.columns if c.startswith("excl_")]
    for column, dtype in OPTION_QUOTE_SCHEMA.items():
        if column not in out.columns:
            out[column] = pd.Series(pd.NA, index=out.index, dtype="object")
        try:
            out[column] = out[column].astype(dtype)
        except (TypeError, ValueError):
            if dtype.startswith("datetime"):
                utc = "UTC" in dtype
                out[column] = pd.to_datetime(out[column], utc=utc, errors="coerce")
                if not utc and getattr(out[column].dtype, "tz", None) is not None:
                    out[column] = out[column].dt.tz_localize(None)
            elif dtype == "bool":
                out[column] = out[column].fillna(False).astype("bool")
            else:
                out[column] = pd.to_numeric(out[column], errors="coerce")
    return out[list(OPTION_QUOTE_SCHEMA) + audit_columns]
