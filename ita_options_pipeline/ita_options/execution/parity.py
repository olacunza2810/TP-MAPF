"""Secuencia de la paridad put-call en patas separadas.

Alpaca no admite patas de acción dentro de una ``mleg``, y la prueba en paper
(``reportes/paper_probe_20260914T175808Z.json``) mostró el riesgo concreto: la
compra de 100 acciones quedó ``partially_filled`` y la ``mleg`` que vendía el
call, enviada inmediatamente, fue rechazada por descubierta
(``40310000``). La secuencia evita ese estado por construcción:

1. **Acciones primero**, con un límite marcable (precio de la señal más
   ``stock_limit_tolerance``).
2. **Fill completo o cancelación.** Si vence ``stock_fill_timeout_s``, se
   cancela y se lee cuánto llenó. Sólo cuentan lotes completos de 100; el
   remanente se deshace.
3. **Confirmación de la posición.** Se consulta la posición del broker hasta
   ver las acciones nuevas; si no aparecen, se deshace todo.
4. **Edge residual** con el precio real de las acciones
   (:func:`arbitrage.execution_edge`). Si cayó bajo el mínimo, se deshace.
5. **Recién entonces la ``mleg`` de opciones**, por los lotes cubiertos.
6. Si la ``mleg`` es rechazada o no llena, se deshacen las acciones de los
   lotes sin opciones: nunca queda acción sin su cobertura de opciones ni
   opción corta sin acción.

La conversión reversa sigue la misma secuencia con la venta en corto. En paper
Alpaca aceptó la ``mleg`` de la reversa aun con el short parcial, pero se exige
igual el fill completo: la integridad de la cobertura no debe depender de lo que
el broker tolere.

El cierre invierte el orden: primero se cierran las opciones y sólo después se
deshacen las acciones de las unidades cerradas.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from ..arbitrage import ExecutionCosts, execution_edge
from ..config import ExecutionSettings
from .broker import Broker, BrokerRejection
from .ledger import Ledger
from .orders import build_mleg_order, mleg_limit_price, scaled_net_debit
from .router import (
    ExecutionOutcome,
    Timing,
    cancel_and_settle,
    close_options,
    option_net_cash,
    order_id,
    wait_for_fill,
)

__all__ = ["ParitySequencer"]


class ParitySequencer:
    """Abre y cierra conversiones y conversiones reversas."""

    def __init__(
        self,
        broker: Broker,
        ledger: Ledger,
        settings: ExecutionSettings,
        costs: ExecutionCosts,
        timing: Timing | None = None,
    ) -> None:
        self._broker = broker
        self._ledger = ledger
        self._settings = settings
        self._costs = costs
        self._timing = timing or Timing()
        self._counters: dict[str, int] = {}

    def _next_id(self, strategy_id: str, role: str) -> str:
        key = f"{strategy_id}:{role}"
        attempt = self._counters.get(key, 0)
        self._counters[key] = attempt + 1
        return order_id(strategy_id, role, attempt)

    # -- acciones -----------------------------------------------------------

    def _trade_shares(self, strategy_id: str, symbol: str, shares: float, role: str) -> tuple[float, float]:
        """Orden a mercado por ``shares`` con signo. Devuelve (llenado con signo, precio)."""
        if shares == 0:
            return 0.0, float("nan")
        request = MarketOrderRequest(
            symbol=symbol, qty=abs(shares),
            side=OrderSide.BUY if shares > 0 else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=self._next_id(strategy_id, role),
        )
        try:
            snapshot = self._broker.submit(request)
        except BrokerRejection as exc:
            self._ledger.record_order(strategy_id, role, request, error=exc.message)
            return 0.0, float("nan")
        self._ledger.record_order(strategy_id, role, request, snapshot)
        snapshot = wait_for_fill(self._broker, self._ledger, snapshot,
                                 self._settings.unwind_timeout_s,
                                 self._settings.order_poll_s, self._timing)
        if not snapshot.is_filled:
            snapshot = cancel_and_settle(self._broker, self._ledger, snapshot,
                                         self._settings.order_poll_s, self._timing)
        return math.copysign(snapshot.filled_qty, shares), snapshot.filled_avg_price

    def _unwind(self, strategy_id: str, symbol: str, held: float) -> bool:
        """Deshace ``held`` acciones con signo. Devuelve ``True`` si quedó algo huérfano."""
        if held == 0:
            return False
        filled, _ = self._trade_shares(strategy_id, symbol, -held, "unwind")
        orphan = abs(filled) < abs(held)
        if orphan:
            self._ledger.event(
                "orphan_shares",
                {"symbol": symbol, "to_unwind": held, "unwound": -filled},
                strategy_id,
            )
        return orphan

    def _confirm_position(self, symbol: str, baseline: float, held: float) -> bool:
        """Espera a que la posición del broker refleje las acciones nuevas."""
        for check in range(self._settings.coverage_checks):
            delta = self._broker.positions().get(symbol, 0.0) - baseline
            if (held > 0 and delta >= held) or (held < 0 and delta <= held):
                return True
            if check < self._settings.coverage_checks - 1:
                self._timing.sleep(self._settings.order_poll_s)
        return False

    # -- apertura -----------------------------------------------------------

    def open(self, strategy_id: str, signal: Mapping[str, Any]) -> ExecutionOutcome:
        """Ejecuta la secuencia completa de apertura."""
        settings = self._settings
        legs = list(signal["leg_spec"])
        stock_legs = [leg for leg in legs if leg.get("kind") == "stock"]
        option_legs = [leg for leg in legs if leg.get("kind") == "option"]
        if len(stock_legs) != 1 or not option_legs:
            raise ValueError("Una paridad necesita exactamente una pata de acción y opciones.")

        stock_leg = stock_legs[0]
        symbol = str(stock_leg["symbol"])
        direction = 1 if float(stock_leg["qty"]) > 0 else -1
        target = 100 * settings.contracts
        baseline = self._broker.positions().get(symbol, 0.0)
        limit = round(float(stock_leg["price"]) + direction * settings.stock_limit_tolerance, 2)

        # 1. Acciones con límite marcable.
        request = LimitOrderRequest(
            symbol=symbol, qty=target,
            side=OrderSide.BUY if direction > 0 else OrderSide.SELL,
            time_in_force=TimeInForce.DAY, limit_price=limit,
            client_order_id=self._next_id(strategy_id, "stock"),
        )
        try:
            snapshot = self._broker.submit(request)
        except BrokerRejection as exc:
            self._ledger.record_order(strategy_id, "stock_open", request, error=exc.message)
            return ExecutionOutcome("rejected", messages=[f"acciones rechazadas: {exc.message}"])
        self._ledger.record_order(strategy_id, "stock_open", request, snapshot)

        # 2. Fill completo o cancelación.
        snapshot = wait_for_fill(self._broker, self._ledger, snapshot,
                                 settings.stock_fill_timeout_s, settings.order_poll_s,
                                 self._timing)
        if not snapshot.is_filled:
            snapshot = cancel_and_settle(self._broker, self._ledger, snapshot,
                                         settings.order_poll_s, self._timing)
        filled = float(snapshot.filled_qty)
        average = snapshot.filled_avg_price if math.isfinite(snapshot.filled_avg_price) else limit
        lots = int(filled // 100)
        messages: list[str] = []
        orphan = False

        if lots == 0:
            orphan = self._unwind(strategy_id, symbol, direction * filled)
            return ExecutionOutcome(
                "aborted", orphan=orphan,
                messages=[f"acciones llenaron {filled:g} de {target}: ningún lote completo"],
            )
        leftover = filled - lots * 100
        if leftover > 0:
            messages.append(f"acciones llenaron {filled:g}: se deshacen {leftover:g}")
            orphan |= self._unwind(strategy_id, symbol, direction * leftover)
        held = direction * lots * 100

        # 3. Posición confirmada en el broker.
        if not self._confirm_position(symbol, baseline, held):
            orphan |= self._unwind(strategy_id, symbol, held)
            return ExecutionOutcome(
                "aborted", orphan=orphan,
                messages=[*messages, "la posición en acciones no se confirmó en el broker"],
            )

        # 4. Edge residual con el precio real de las acciones.
        fills = [average if leg.get("kind") == "stock" else float(leg["price"]) for leg in legs]
        edge = execution_edge(signal, fills, self._costs, lots)
        if edge < settings.min_edge_usd * lots:
            orphan |= self._unwind(strategy_id, symbol, held)
            return ExecutionOutcome(
                "edge_gone", residual_edge=edge, orphan=orphan,
                messages=[*messages, f"edge con acciones a {average:.2f}: {edge:,.2f} USD"],
            )

        # 5. Opciones por los lotes cubiertos.
        option_prices = [float(leg["price"]) for leg in option_legs]
        option_limit = mleg_limit_price(scaled_net_debit(option_legs, option_prices))
        mleg = build_mleg_order(
            option_legs, lots, option_limit,
            client_order_id=self._next_id(strategy_id, "mleg"),
        )
        try:
            option_snapshot = self._broker.submit(mleg)
        except BrokerRejection as exc:
            self._ledger.record_order(strategy_id, "mleg_open", mleg, error=exc.message)
            orphan |= self._unwind(strategy_id, symbol, held)
            return ExecutionOutcome(
                "hedge_failed", orphan=orphan,
                messages=[*messages, f"mleg rechazada ({exc.code}): {exc.message}"],
            )
        self._ledger.record_order(strategy_id, "mleg_open", mleg, option_snapshot)
        option_snapshot = wait_for_fill(self._broker, self._ledger, option_snapshot,
                                        settings.option_fill_timeout_s,
                                        settings.order_poll_s, self._timing)
        if not option_snapshot.is_filled:
            option_snapshot = cancel_and_settle(self._broker, self._ledger, option_snapshot,
                                                settings.order_poll_s, self._timing)
        units = int(option_snapshot.filled_qty)

        # 6. Deshacer las acciones de los lotes sin opciones.
        if units < lots:
            orphan |= self._unwind(strategy_id, symbol, direction * (lots - units) * 100)
        if units == 0:
            return ExecutionOutcome(
                "hedge_failed", orphan=orphan,
                messages=[*messages, "la mleg no llenó: acciones deshechas"],
            )
        if units < lots:
            messages.append(f"mleg llenó {units} de {lots} lotes")

        entry_cash = (
            -direction * average * units * 100
            + option_net_cash(option_snapshot, option_limit, units)
        )
        return ExecutionOutcome(
            "open", units, entry_cash,
            execution_edge(signal, fills, self._costs, units), orphan, messages,
        )

    # -- cierre -------------------------------------------------------------

    def close(
        self, strategy: Mapping[str, Any], quotes: pd.DataFrame
    ) -> tuple[int, float, list[str], bool]:
        """Cierra opciones y después las acciones de las unidades cerradas.

        Returns:
            ``(unidades cerradas, caja de salida, mensajes, hubo huérfanos)``.
        """
        units, option_cash, messages = close_options(
            self._broker, self._ledger, strategy, quotes, self._settings, self._timing
        )
        if units == 0:
            return 0, 0.0, [*messages, "opciones sin cerrar: las acciones se mantienen"], False
        stock_leg = next(leg for leg in strategy["legs"] if leg["kind"] == "stock")
        held = float(stock_leg["units"]) * units
        filled, average = self._trade_shares(strategy["id"], stock_leg["symbol"], -held, "stock_close")
        orphan = abs(filled) < abs(held)
        if orphan:
            self._ledger.event("orphan_shares", {"symbol": stock_leg["symbol"], "to_close": held,
                                                 "closed": -filled}, strategy["id"])
        stock_cash = -filled * (average if math.isfinite(average) else 0.0)
        return units, option_cash + stock_cash, messages, orphan
