"""Construcción de órdenes multi-leg de Alpaca a partir de las señales.

Traduce el ``leg_spec`` que producen los detectores a un ``LimitOrderRequest``
con ``order_class=mleg``. No envía nada: arma y valida el request.

Reglas de Alpaca que se aplican acá (documentación de options level 3):

- las patas de una ``mleg`` llenan juntas o no llenan;
- ``ratio_qty`` es entero y el máximo común divisor de las patas debe ser 1;
- **no se admiten patas de acción**: la paridad put-call necesita una orden de
  acción separada;
- contratos enteros, ``time_in_force`` ``day`` o ``gtc``, sin extended hours.

Convención de signo confirmada en paper (``reportes/paper_probe_20260914T175808Z.json``):
un vertical de débito con ``limit_price=+2.5`` llenó de inmediato, así que en
una ``mleg`` el precio positivo es débito y el negativo es crédito.

La misma prueba mostró que Alpaca rechaza una ``mleg`` que vende un call si la
compra de acciones que debía cubrirlo todavía está ``partially_filled``
(``40310000: account not eligible to trade uncovered option contracts``). Por
eso la paridad nunca envía la pata de opciones antes de confirmar la posición
completa en acciones (ver :mod:`ita_options.execution.parity`).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from functools import reduce

from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest

__all__ = [
    "OrderConstructionError",
    "MAX_MLEG_LEGS",
    "CREDIT_LIMIT_SIGN",
    "integer_ratios",
    "scaled_net_debit",
    "mleg_limit_price",
    "strategy_order_id",
    "build_mleg_order",
]

#: Cantidad máxima de patas que se envían en una ``mleg``. La documentación de
#: Alpaca muestra estructuras de hasta cuatro patas; no se prueba más allá.
MAX_MLEG_LEGS = 4

#: Signo del ``limit_price`` de una ``mleg`` de crédito. Confirmado en paper el
#: 14/09/2026: positivo es débito (se paga) y negativo es crédito (se cobra).
CREDIT_LIMIT_SIGN = -1


class OrderConstructionError(ValueError):
    """La señal no se puede expresar como una orden ``mleg`` válida."""


def integer_ratios(quantities: Sequence[float], max_denominator: int = 100) -> list[int]:
    """Convierte cantidades con signo a enteros primos entre sí.

    Una butterfly con strikes no equidistantes tiene pesos como
    ``+0.4, +0.6, -1``. Alpaca exige ``ratio_qty`` enteros con MCD 1, así que
    se escalan a ``+2, +3, -5`` conservando la proporción exacta.

    Args:
        quantities: Cantidades por pata, positivas para comprar.
        max_denominator: Denominador máximo admitido al racionalizar.

    Returns:
        Ratios enteros con signo, con MCD 1.

    Raises:
        OrderConstructionError: si alguna cantidad es cero o no se puede
            representar con el denominador pedido sin error.
    """
    fractions = []
    for quantity in quantities:
        fraction = Fraction(float(quantity)).limit_denominator(max_denominator)
        if fraction == 0:
            raise OrderConstructionError("Una pata con cantidad cero no es una pata.")
        if abs(float(fraction) - float(quantity)) > 1e-6:
            raise OrderConstructionError(
                f"La cantidad {quantity} no admite una proporción entera con "
                f"denominador <= {max_denominator}."
            )
        fractions.append(fraction)
    common = reduce(
        lambda a, b: a * b // math.gcd(a, b), (f.denominator for f in fractions)
    )
    integers = [int(f * common) for f in fractions]
    divisor = reduce(math.gcd, (abs(i) for i in integers))
    return [i // divisor for i in integers]


def _option_legs(leg_spec: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    """Valida que la estructura sea expresable como ``mleg``."""
    legs = list(leg_spec)
    if any(leg.get("kind") != "option" for leg in legs):
        raise OrderConstructionError(
            "Alpaca no admite patas de acción en una orden mleg: la paridad "
            "put-call se ejecuta con la acción en una orden separada."
        )
    if not 2 <= len(legs) <= MAX_MLEG_LEGS:
        raise OrderConstructionError(
            f"Una mleg necesita entre 2 y {MAX_MLEG_LEGS} patas; hay {len(legs)}."
        )
    return legs


def scaled_net_debit(
    leg_spec: Sequence[Mapping[str, object]], prices: Sequence[float]
) -> float:
    """Precio neto por unidad de la ``mleg`` escalada a ratios enteros.

    Positivo es débito (se paga), negativo es crédito (se cobra). Para una
    butterfly ``+2/+3/-5`` es ``2·p1 + 3·p3 − 5·p2``.
    """
    legs = _option_legs(leg_spec)
    if len(prices) != len(legs):
        raise OrderConstructionError("Hace falta un precio por pata.")
    ratios = integer_ratios([float(leg["qty"]) for leg in legs])  # type: ignore[arg-type]
    return float(sum(r * float(p) for r, p in zip(ratios, prices, strict=True)))


def mleg_limit_price(net_debit: float) -> float:
    """Traduce un precio neto (positivo = débito) a la convención de Alpaca.

    Usa :data:`CREDIT_LIMIT_SIGN`, confirmado en paper.
    """
    sign = 1.0 if net_debit >= 0 else float(CREDIT_LIMIT_SIGN)
    return round(abs(net_debit) * sign, 2)


def strategy_order_id(
    detector: str, symbols: Sequence[str], signal_at: object, opening: bool
) -> str:
    """``client_order_id`` determinístico: la misma señal no se envía dos veces."""
    payload = "|".join(
        [detector, *sorted(map(str, symbols)), str(signal_at),
         "open" if opening else "close"]
    )
    return "ita-" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]


def build_mleg_order(
    leg_spec: Sequence[Mapping[str, object]],
    contracts: int,
    limit_price: float,
    opening: bool = True,
    client_order_id: str | None = None,
    time_in_force: TimeInForce = TimeInForce.DAY,
) -> LimitOrderRequest:
    """Arma el request ``mleg`` de una señal de opciones.

    Args:
        leg_spec: Patas de la señal, en el formato de ``arbitrage._opt_leg``.
        contracts: Unidades de la estructura (``qty`` de la orden padre).
        limit_price: Precio límite ya expresado en la convención de Alpaca
            (ver :func:`mleg_limit_price`).
        opening: ``True`` para abrir la estructura; ``False`` invierte los
            lados para cerrarla.
        client_order_id: Identificador idempotente, típicamente de
            :func:`strategy_order_id`.
        time_in_force: ``day`` o ``gtc``.

    Returns:
        El ``LimitOrderRequest`` validado por el SDK.

    Raises:
        OrderConstructionError: si la señal no es expresable como ``mleg``.
    """
    if int(contracts) != contracts or contracts < 1:
        raise OrderConstructionError("Alpaca opera contratos enteros y positivos.")
    legs = _option_legs(leg_spec)
    ratios = integer_ratios([float(leg["qty"]) for leg in legs])  # type: ignore[arg-type]

    requests = []
    for leg, ratio in zip(legs, ratios, strict=True):
        buy = (ratio > 0) == opening
        if opening:
            intent = PositionIntent.BUY_TO_OPEN if buy else PositionIntent.SELL_TO_OPEN
        else:
            intent = PositionIntent.BUY_TO_CLOSE if buy else PositionIntent.SELL_TO_CLOSE
        requests.append(
            OptionLegRequest(
                symbol=str(leg["symbol"]),
                ratio_qty=abs(ratio),
                side=OrderSide.BUY if buy else OrderSide.SELL,
                position_intent=intent,
            )
        )
    return LimitOrderRequest(
        qty=int(contracts),
        order_class=OrderClass.MLEG,
        time_in_force=time_in_force,
        limit_price=round(float(limit_price), 2),
        legs=requests,
        client_order_id=client_order_id,
    )
