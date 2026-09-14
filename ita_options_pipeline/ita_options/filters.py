"""Filtros de liquidez y explotabilidad.

Los filtros no borran filas en silencio: marcan cada causa de exclusión en una
columna booleana y devuelven un reporte con el conteo por causa. Poder mostrar
esa tabla en la presentación es la diferencia entre "aplicamos filtros de
liquidez" y una justificación auditable.

Hay dos juegos de criterios, según el insumo (``LiquidityThresholds.data_mode``):

``nbbo``
    Cotizaciones con bid y ask reales (recorder de Alpaca). Se evalúan volumen,
    open interest, bid, spread y quotes cruzados.

``daily``
    Velas diarias (Polygon gratuito). No hay bid/ask observado ni open interest
    histórico, así que la liquidez se juzga por el **volumen de la rueda
    anterior** y el precio de referencia. El spread sintético no se filtra: es
    un supuesto, no una medición, y filtrar por él sería filtrar por el supuesto.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .config import LiquidityThresholds

_LOG = logging.getLogger(__name__)

__all__ = ["FilterReport", "apply_liquidity_filters"]


@dataclass(frozen=True, slots=True)
class FilterReport:
    """Resumen de la aplicación de filtros.

    Attributes:
        total_rows: Filas evaluadas.
        surviving_rows: Filas que pasan todos los criterios.
        exclusions: Conteo de filas excluidas por cada criterio, no exclusivo
            entre sí (una fila puede violar varios).
    """

    total_rows: int
    surviving_rows: int
    exclusions: dict[str, int]

    @property
    def survival_rate(self) -> float:
        """Fracción de filas que sobreviven al filtrado."""
        return self.surviving_rows / self.total_rows if self.total_rows else 0.0

    def to_frame(self) -> pd.DataFrame:
        """Devuelve el reporte como DataFrame, listo para la presentación."""
        return (
            pd.DataFrame(
                {
                    "criterio": list(self.exclusions),
                    "filas_excluidas": list(self.exclusions.values()),
                }
            )
            .assign(
                pct_del_total=lambda d: d["filas_excluidas"]
                / max(self.total_rows, 1)
                * 100.0
            )
            .sort_values("filas_excluidas", ascending=False, ignore_index=True)
        )


def _nbbo_checks(
    out: pd.DataFrame, thresholds: LiquidityThresholds
) -> tuple[dict[str, pd.Series], set[str]]:
    """Criterios sobre cotizaciones con bid y ask reales."""
    checks: dict[str, pd.Series] = {
        # Sólo se excluye cuando el volumen se conoce y es bajo. Tratar el
        # volumen desconocido como cero descartaría toda la cadena grabada por
        # snapshots, donde el dato no existe; tratarlo como infinito dejaría
        # pasar contratos ilíquidos. Se cuenta aparte para que quede a la vista.
        "volume_insuficiente": out["volume"].notna()
        & (out["volume"] <= thresholds.min_volume),
        "volumen_desconocido": out["volume"].isna(),
        "open_interest_insuficiente": (
            out["open_interest"].fillna(0) <= thresholds.min_open_interest
        ),
        "bid_nulo": out["bid"].fillna(0) <= thresholds.min_bid,
        "spread_relativo_excesivo": (
            out["spread_rel"].isna()
            | (out["spread_rel"] >= thresholds.max_relative_spread)
        ),
        "quote_cruzado": out["spread_abs"].fillna(-1.0) <= 0.0,
        "mid_por_debajo_del_minimo": (
            out["mid_price"].fillna(0) < thresholds.min_mid_price
        ),
        "dte_fuera_de_rango": (
            (out["dte"] < thresholds.min_dte) | (out["dte"] > thresholds.max_dte)
        ),
        "spot_no_alineado": out["underlying_price"].isna(),
    }
    # ``volumen_desconocido`` se reporta pero no excluye: es un diagnóstico de
    # cobertura del dato, no un criterio de explotabilidad.
    return checks, {"volumen_desconocido"}


def _daily_checks(
    out: pd.DataFrame, thresholds: LiquidityThresholds
) -> tuple[dict[str, pd.Series], set[str]]:
    """Criterios sobre velas diarias: volumen de la rueda anterior y precio.

    Se usa el volumen de ``t-1`` y no el de ``t`` para que el filtro sea
    conocible antes de operar en la rueda ``t``, igual que el open interest
    rezagado del modo ``nbbo``.
    """
    if "volume_prev_day" not in out.columns:
        raise KeyError(
            "El modo 'daily' necesita la columna 'volume_prev_day' (volumen de la "
            "rueda anterior). Ver daily.previous_session_volume."
        )
    prev = pd.to_numeric(out["volume_prev_day"], errors="coerce")
    mid = pd.to_numeric(out["mid_price"], errors="coerce")
    checks: dict[str, pd.Series] = {
        "volumen_rueda_anterior_insuficiente": prev.notna()
        & (prev <= thresholds.min_prev_day_volume),
        # A diferencia del modo nbbo, el volumen desconocido excluye: en datos
        # diarios sólo falta en la primera rueda de cada serie, y ahí no hay
        # ninguna otra evidencia de liquidez.
        "volumen_rueda_anterior_desconocido": prev.isna(),
        "precio_invalido": mid.isna() | (mid <= 0.0),
        "mid_por_debajo_del_minimo": mid.fillna(0) < thresholds.min_mid_price,
        "dte_fuera_de_rango": (
            (out["dte"] < thresholds.min_dte) | (out["dte"] > thresholds.max_dte)
        ),
        "spot_no_alineado": out["underlying_price"].isna(),
    }
    return checks, set()


def apply_liquidity_filters(
    frame: pd.DataFrame,
    thresholds: LiquidityThresholds,
    drop: bool = False,
) -> tuple[pd.DataFrame, FilterReport]:
    """Marca y opcionalmente descarta contratos no explotables.

    Criterios del modo ``nbbo``, en el orden en que aparecen en la consigna más
    los que la práctica exige:

    - ``volume > min_volume``: sin volumen negociado no hay evidencia de que el
      contrato se pueda operar.
    - ``open_interest > min_open_interest``: usando el OI rezagado, ver
      :func:`enrich.attach_lagged_open_interest`.
    - ``bid > min_bid``: un bid de cero deja la pata vendedora sin contraparte.
      Es la exclusión más importante: los contratos con bid cero son los que más
      "arbitrajes" falsos generan, porque el mid colapsa a la mitad del ask.
    - ``spread_rel < max_relative_spread``: el spread es el costo de cruzar. Una
      señal de 5% de desvío sobre un contrato con 30% de spread no es
      explotable.
    - ``spread_abs > 0``: descarta quotes cruzados o bloqueados.
    - ``min_mid_price``, ``min_dte``, ``max_dte``: higiene del universo.
    - ``underlying_price`` no nulo: sin spot alineado no hay valuación posible.

    En modo ``daily`` se reemplazan volumen, open interest, bid y spread por el
    volumen de la rueda anterior (``volume_prev_day > min_prev_day_volume``) y
    un precio de referencia válido.

    Args:
        frame: DataFrame enriquecido.
        thresholds: Umbrales configurados.
        drop: Si ``True`` devuelve sólo las filas que pasan; si ``False``
            devuelve todo el frame con la columna ``is_tradable``.

    Returns:
        Tupla ``(frame, reporte)``.
    """
    out = frame.copy()
    build = _daily_checks if thresholds.data_mode == "daily" else _nbbo_checks
    checks, informational = build(out, thresholds)

    excluded = pd.Series(False, index=out.index)
    for name, mask in checks.items():
        out[f"excl_{name}"] = mask.fillna(True)
        if name not in informational:
            excluded |= out[f"excl_{name}"]

    out["is_tradable"] = ~excluded
    report = FilterReport(
        total_rows=len(out),
        surviving_rows=int(out["is_tradable"].sum()),
        exclusions={name: int(mask.fillna(True).sum()) for name, mask in checks.items()},
    )
    _LOG.info(
        "Filtros de liquidez (%s): %d/%d filas explotables (%.1f%%).",
        thresholds.data_mode,
        report.surviving_rows,
        report.total_rows,
        report.survival_rate * 100.0,
    )
    return (out.loc[out["is_tradable"]].copy() if drop else out), report
