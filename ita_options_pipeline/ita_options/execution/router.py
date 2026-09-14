"""Ejecución de estructuras sólo de opciones como una orden ``mleg``.

También contiene las utilidades que comparte la paridad: esperar un fill,
cancelar y confirmar el estado final, y cerrar la parte de opciones de una
estrategia abierta.

Política de una ``mleg``:

1. Límite al precio neto de la señal, que ya cruza el spread (compra al ask y
   vende al bid).
2. Esperar el fill hasta ``option_fill_timeout_s``; si no llena, cancelar y
   leer el estado final (puede haber llenado parcialmente mientras se cancelaba).
3. Re-precio: hasta ``max_reprices`` veces, empeorando ``reprice_step`` por
   unidad escalada, **sólo si el edge al nuevo precio sigue siendo al menos
   ``min_edge_usd``**.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..arbitrage import ExecutionCosts
from ..config import ExecutionSettings
from .broker import Broker, BrokerRejection, OrderSnapshot
from .ledger import Ledger
from .orders import build_mleg_order, integer_ratios, mleg_limit_price, scaled_net_debit

__all__ = [
    "Timing",
    "ExecutionOutcome",
    "order_id",
    "wait_for_fill",
    "cancel_and_settle",
    "edge_at_scaled_debit",
    "option_net_cash",
    "execute_mleg",
    "close_options",
]


@dataclass
class Timing:
    """Reloj y espera inyectables: los tests avanzan el tiempo sin dormir."""

    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic


@dataclass
class ExecutionOutcome:
    """Resultado de intentar abrir una estructura."""

    state: str
    contracts_open: int = 0
    entry_cash: float = 0.0
    residual_edge: float = float("nan")
    orphan: bool = False
    messages: list[str] = field(default_factory=list)


def order_id(strategy_id: str, role: str, attempt: int = 0) -> str:
    """``client_order_id`` único por estrategia, rol e intento."""
    return f"ita-{strategy_id[:16]}-{role}-{attempt}"


def wait_for_fill(
    broker: Broker, ledger: Ledger, snapshot: OrderSnapshot, timeout: float,
    poll: float, timing: Timing,
) -> OrderSnapshot:
    """Consulta la orden hasta que llene, llegue a un estado final o venza el plazo."""
    deadline = timing.now() + timeout
    current = snapshot
    while not current.is_filled and not current.is_final and timing.now() < deadline:
        timing.sleep(poll)
        current = broker.get_order(current.id)
        ledger.update_order(current)
    return current


def cancel_and_settle(
    broker: Broker, ledger: Ledger, snapshot: OrderSnapshot, poll: float,
    timing: Timing, attempts: int = 5,
) -> OrderSnapshot:
    """Cancela una orden abierta y devuelve su estado final, con los fills que tuvo."""
    if snapshot.is_filled or snapshot.is_final:
        return snapshot
    broker.cancel(snapshot.id)
    current = broker.get_order(snapshot.id)
    for _ in range(attempts):
        if current.is_final:
            break
        timing.sleep(poll)
        current = broker.get_order(snapshot.id)
    ledger.update_order(current)
    return current


def _scale(legs: list[Mapping[str, Any]]) -> float:
    """Factor entre los ratios enteros y las cantidades de la señal."""
    ratios = integer_ratios([float(leg["qty"]) for leg in legs])
    return ratios[0] / float(legs[0]["qty"])


def edge_at_scaled_debit(
    signal: Mapping[str, Any], scaled_debit: float, costs: ExecutionCosts, contracts: float
) -> float:
    """Edge en USD si la ``mleg`` se llena a un precio neto escalado dado.

    El precio de la señal ya produce ``net_edge_usd``; cada USD de débito extra
    por unidad de la señal lo reduce en ``multiplicador`` USD.
    """
    legs = [leg for leg in signal["leg_spec"] if leg.get("kind") == "option"]
    signal_debit = sum(float(leg["qty"]) * float(leg["price"]) for leg in legs)
    debit = scaled_debit / _scale(legs)
    return (float(signal["net_edge_usd"]) + (signal_debit - debit) * costs.multiplier) * contracts


def option_net_cash(snapshot: OrderSnapshot, limit_price: float, units: int) -> float:
    """Caja de la parte de opciones: negativa si se pagó, positiva si se cobró.

    Usa el precio promedio del fill si Alpaca lo informa y, si no, el límite,
    que para un fill es el peor precio posible.
    """
    net = snapshot.filled_avg_price if math.isfinite(snapshot.filled_avg_price) else limit_price
    return -float(net) * 100.0 * units


def execute_mleg(
    broker: Broker,
    ledger: Ledger,
    strategy_id: str,
    signal: Mapping[str, Any],
    settings: ExecutionSettings,
    costs: ExecutionCosts,
    timing: Timing | None = None,
) -> ExecutionOutcome:
    """Abre una estructura sólo de opciones con una orden ``mleg``."""
    timing = timing or Timing()
    legs = list(signal["leg_spec"])
    prices = [float(leg["price"]) for leg in legs]
    base_debit = scaled_net_debit(legs, prices)
    contracts = settings.contracts
    messages: list[str] = []

    for attempt in range(settings.max_reprices + 1):
        debit = base_debit + attempt * settings.reprice_step
        edge = edge_at_scaled_debit(signal, debit, costs, contracts)
        if edge < settings.min_edge_usd * contracts:
            messages.append(f"re-precio {attempt} descartado: edge {edge:,.2f} USD bajo el mínimo")
            break
        limit = mleg_limit_price(debit)
        request = build_mleg_order(
            legs, contracts, limit, client_order_id=order_id(strategy_id, "mleg", attempt)
        )
        try:
            snapshot = broker.submit(request)
        except BrokerRejection as exc:
            ledger.record_order(strategy_id, "mleg_open", request, error=exc.message)
            return ExecutionOutcome("rejected", messages=[*messages, exc.message])
        ledger.record_order(strategy_id, "mleg_open", request, snapshot)
        snapshot = wait_for_fill(broker, ledger, snapshot, settings.option_fill_timeout_s,
                                 settings.order_poll_s, timing)
        if not snapshot.is_filled:
            snapshot = cancel_and_settle(broker, ledger, snapshot, settings.order_poll_s, timing)
        units = int(snapshot.filled_qty)
        if units > 0:
            if units < contracts:
                messages.append(f"mleg llenó {units} de {contracts} unidades")
            return ExecutionOutcome(
                "open", units, option_net_cash(snapshot, limit, units),
                edge_at_scaled_debit(signal, debit, costs, units), messages=messages,
            )
        messages.append(f"intento {attempt}: sin fill a {limit:+.2f}")
    return ExecutionOutcome("unfilled", messages=messages)


def close_options(
    broker: Broker,
    ledger: Ledger,
    strategy: Mapping[str, Any],
    quotes: pd.DataFrame,
    settings: ExecutionSettings,
    timing: Timing | None = None,
) -> tuple[int, float, list[str]]:
    """Cierra la parte de opciones de una estrategia con una ``mleg`` inversa.

    Cada pata larga se vende al bid y cada corta se recompra al ask; el límite
    cede ``close_cushion`` por unidad escalada para priorizar el cierre.

    Args:
        broker: Broker.
        ledger: Ledger.
        strategy: Estrategia del ledger, con ``legs`` y ``contracts_open``.
        quotes: Quotes indexados por símbolo con ``bid`` y ``ask``.
        settings: Parámetros de ejecución.
        timing: Reloj inyectable.

    Returns:
        ``(unidades cerradas, caja de salida, mensajes)``.
    """
    timing = timing or Timing()
    legs = [leg for leg in strategy["legs"] if leg["kind"] == "option"]
    contracts = int(strategy["contracts_open"] or 0)
    if contracts <= 0 or not legs:
        return 0, 0.0, ["sin opciones abiertas"]

    debit = 0.0
    for leg in legs:
        if leg["symbol"] not in quotes.index:
            return 0, 0.0, [f"sin quote para cerrar {leg['symbol']}"]
        row = quotes.loc[leg["symbol"]]
        price = float(row["bid"] if leg["units"] > 0 else row["ask"])
        if not math.isfinite(price) or price <= 0:
            return 0, 0.0, [f"quote inválido para cerrar {leg['symbol']}"]
        debit += -leg["units"] * price
    debit += settings.close_cushion
    limit = mleg_limit_price(debit)
    attempt = len([o for o in ledger.orders(strategy["id"]) if o["role"] == "mleg_close"])
    request = build_mleg_order(
        legs, contracts, limit, opening=False,
        client_order_id=order_id(strategy["id"], "close", attempt),
    )
    try:
        snapshot = broker.submit(request)
    except BrokerRejection as exc:
        ledger.record_order(strategy["id"], "mleg_close", request, error=exc.message)
        return 0, 0.0, [exc.message]
    ledger.record_order(strategy["id"], "mleg_close", request, snapshot)
    snapshot = wait_for_fill(broker, ledger, snapshot, settings.option_fill_timeout_s,
                             settings.order_poll_s, timing)
    if not snapshot.is_filled:
        snapshot = cancel_and_settle(broker, ledger, snapshot, settings.order_poll_s, timing)
    units = int(snapshot.filled_qty)
    return units, option_net_cash(snapshot, limit, units), []
