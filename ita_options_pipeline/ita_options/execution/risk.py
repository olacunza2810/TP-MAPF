"""Controles de riesgo previos al envío de una orden.

Ninguno reemplaza al margen que calcula Alpaca. Sirven para no enviar órdenes
que el broker rechazaría o que el piloto no debería tomar:

- **pérdida máxima al vencimiento** por payoff lineal por tramos, comparada con
  el buying power de opciones y con un tope por operación;
- **profundidad**: el tamaño publicado en el NBBO debe cubrir la cantidad;
- **fecha ex-dividendo**: una paridad que atraviesa una fecha ex queda expuesta
  al dividendo discreto que el detector no modela (sección 14 del informe);
- **short**: la conversión reversa necesita vender la acción en corto.

La pérdida máxima supone que todas las patas vencen juntas y no modela la
asignación anticipada; por eso no se calcula para calendarios.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

__all__ = [
    "PreTradeDecision",
    "expiry_pnl",
    "max_loss_at_expiry",
    "insufficient_depth",
    "ex_dividend_in_window",
    "pre_trade_check",
]


def _leg_value(leg: Mapping[str, object], spot: float) -> float:
    """Valor al vencimiento de una pata por unidad de subyacente."""
    if leg.get("kind") == "stock":
        return spot
    strike = float(leg["strike"])  # type: ignore[arg-type]
    if str(leg.get("option_type", "")).lower().startswith("c"):
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def expiry_pnl(
    leg_spec: Sequence[Mapping[str, object]], spot: float, multiplier: float = 100.0
) -> float:
    """P&L al vencimiento en USD para un spot dado, a los precios de entrada."""
    return float(
        sum(
            float(leg["qty"]) * (_leg_value(leg, spot) - float(leg["price"]))  # type: ignore[arg-type]
            for leg in leg_spec
        )
        * multiplier
    )


def max_loss_at_expiry(
    leg_spec: Sequence[Mapping[str, object]], multiplier: float = 100.0
) -> float:
    """Pérdida máxima al vencimiento, en USD y como número positivo.

    El payoff es lineal por tramos con quiebres en los strikes, así que el
    mínimo está en ``S = 0``, en algún strike o cuando ``S`` tiende a infinito.
    Si la pendiente final es negativa (por ejemplo, un call vendido
    descubierto), la pérdida es ilimitada y se devuelve ``inf``.

    Raises:
        ValueError: si las patas de opciones tienen vencimientos distintos.
    """
    expirations = {
        str(pd.Timestamp(leg["expiration"]).date())  # type: ignore[arg-type]
        for leg in leg_spec
        if leg.get("kind") == "option"
    }
    if len(expirations) > 1:
        raise ValueError(
            "La pérdida máxima por payoff al vencimiento no aplica a calendarios."
        )
    strikes = sorted(
        {float(leg["strike"]) for leg in leg_spec if leg.get("kind") == "option"}  # type: ignore[arg-type]
    )
    top = (strikes[-1] if strikes else 1.0) * 10.0 + 1.0
    points = [0.0, *strikes, top]
    values = [expiry_pnl(leg_spec, s, multiplier) for s in points]
    slope = expiry_pnl(leg_spec, top + 1.0, multiplier) - values[-1]
    if slope < -1e-9:
        return float("inf")
    return max(0.0, -min(values))


def insufficient_depth(
    leg_spec: Sequence[Mapping[str, object]],
    sizes: Mapping[str, tuple[float, float]],
    contracts: int,
    ratios: Sequence[int] | None = None,
) -> list[str]:
    """Patas cuyo tamaño publicado no cubre la cantidad a operar.

    Args:
        leg_spec: Patas de la señal.
        sizes: ``símbolo -> (bid_size, ask_size)`` del NBBO, en contratos.
        contracts: Unidades de la estructura.
        ratios: Ratios enteros por pata; por defecto, el valor absoluto de
            ``qty``.

    Returns:
        Descripción de cada pata insuficiente; vacía si todas alcanzan.
    """
    problems = []
    option_legs = [leg for leg in leg_spec if leg.get("kind") == "option"]
    if ratios is None:
        ratios = [abs(float(leg["qty"])) for leg in option_legs]  # type: ignore[arg-type]
    for leg, ratio in zip(option_legs, ratios, strict=True):
        symbol = str(leg["symbol"])
        bid_size, ask_size = sizes.get(symbol, (0.0, 0.0))
        needed = abs(ratio) * contracts
        buying = float(leg["qty"]) > 0  # type: ignore[arg-type]
        available = float(ask_size if buying else bid_size)
        if not np.isfinite(available) or available < needed:
            side = "ask" if buying else "bid"
            problems.append(
                f"{symbol}: {side} publica {available:g}, se necesitan {needed:g}"
            )
    return problems


def ex_dividend_in_window(
    underlying: str, start: date, end: date, dividends: pd.DataFrame | None
) -> list[date]:
    """Fechas ex-dividendo del subyacente dentro de ``(start, end]``."""
    if dividends is None or dividends.empty:
        return []
    ex_dates = pd.to_datetime(dividends["ex_dividend_date"]).dt.date
    mask = (dividends["ticker"] == underlying) & (ex_dates > start) & (ex_dates <= end)
    return sorted(ex_dates[mask])


@dataclass(frozen=True)
class PreTradeDecision:
    """Resultado de los controles previos al envío."""

    approved: bool
    max_loss_usd: float
    reasons: list[str] = field(default_factory=list)


def pre_trade_check(
    signal: Mapping[str, object],
    contracts: int,
    options_buying_power: float,
    max_loss_per_trade: float,
    sizes: Mapping[str, tuple[float, float]],
    today: date,
    dividends: pd.DataFrame | None = None,
    shorting_enabled: bool = True,
    multiplier: float = 100.0,
) -> PreTradeDecision:
    """Aplica todos los controles y devuelve si la orden puede enviarse.

    Args:
        signal: Oportunidad con ``detector``, ``underlying`` y ``leg_spec``.
        contracts: Unidades de la estructura.
        options_buying_power: Buying power de opciones de la cuenta, en USD.
        max_loss_per_trade: Tope de pérdida máxima por operación, en USD.
        sizes: Tamaños del NBBO por símbolo.
        today: Fecha de la rueda.
        dividends: Dividendos con ``ticker`` y ``ex_dividend_date``.
        shorting_enabled: Si la cuenta puede vender en corto.
        multiplier: Acciones por contrato.
    """
    legs = list(signal["leg_spec"])  # type: ignore[arg-type]
    reasons: list[str] = []
    detector = str(signal.get("detector", ""))

    try:
        loss = max_loss_at_expiry(legs, multiplier) * contracts
    except ValueError as exc:
        loss = float("nan")
        reasons.append(str(exc))
    if loss == float("inf"):
        reasons.append("pérdida ilimitada: hay una pata corta descubierta")
    elif np.isfinite(loss):
        if loss > max_loss_per_trade:
            reasons.append(
                f"pérdida máxima {loss:,.0f} USD supera el tope {max_loss_per_trade:,.0f}"
            )
        if loss > options_buying_power:
            reasons.append(
                f"pérdida máxima {loss:,.0f} USD supera el buying power "
                f"{options_buying_power:,.0f}"
            )

    reasons.extend(insufficient_depth(legs, sizes, contracts))

    stock_legs = [leg for leg in legs if leg.get("kind") == "stock"]
    if not shorting_enabled and any(float(leg["qty"]) < 0 for leg in stock_legs):  # type: ignore[arg-type]
        reasons.append(
            "la estructura vende la acción en corto y la cuenta no tiene short habilitado"
        )

    if detector == "put_call_parity":
        expirations = [
            pd.Timestamp(leg["expiration"]).date()  # type: ignore[arg-type]
            for leg in legs
            if leg.get("kind") == "option"
        ]
        ex_dates = ex_dividend_in_window(
            str(signal.get("underlying", "")), today, max(expirations), dividends
        )
        if ex_dates:
            reasons.append(
                "fecha ex-dividendo dentro de la tenencia: "
                + ", ".join(d.isoformat() for d in ex_dates)
            )

    return PreTradeDecision(approved=not reasons, max_loss_usd=loss, reasons=reasons)
