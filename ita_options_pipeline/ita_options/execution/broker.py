"""Acceso al broker con una interfaz mínima: órdenes, posiciones, cuenta y reloj.

El resto de la capa de ejecución no toca ``alpaca-py`` directamente: trabaja con
:class:`OrderSnapshot` y con el protocolo :class:`Broker`. Eso permite probar la
secuencia de la paridad, la reconciliación y el loop intradía con un broker
falso que reproduce los comportamientos observados en paper, como los fills
parciales de acciones y el rechazo de opciones descubiertas.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "FINAL_STATUSES",
    "ALREADY_FINAL_CODE",
    "UNCOVERED_OPTIONS_CODE",
    "OrderSnapshot",
    "BrokerRejection",
    "Broker",
    "AlpacaBroker",
    "snapshot_from_order",
]

#: Estados de Alpaca después de los cuales la orden ya no cambia.
FINAL_STATUSES = frozenset(
    {"filled", "canceled", "expired", "rejected", "done_for_day", "replaced",
     "stopped", "suspended"}
)

#: Código de Alpaca al cancelar una orden que ya está en estado final.
ALREADY_FINAL_CODE = 42210000

#: Código observado en paper al enviar una ``mleg`` con un call corto descubierto.
UNCOVERED_OPTIONS_CODE = 40310000


def _number(value: Any) -> float:
    try:
        return float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


@dataclass(frozen=True)
class OrderSnapshot:
    """Estado de una orden en un instante.

    Attributes:
        id: Identificador del broker.
        status: Estado de Alpaca (``new``, ``partially_filled``, ``filled``...).
        qty: Cantidad pedida: acciones, o unidades de la estructura en una ``mleg``.
        filled_qty: Cantidad llenada.
        filled_avg_price: Precio promedio del fill; en una ``mleg``, precio neto
            con la convención de signo de Alpaca.
        client_order_id: Identificador propio, para idempotencia.
        symbol: Símbolo, vacío en una ``mleg``.
        side: ``buy`` o ``sell``.
        limit_price: Precio límite enviado.
        legs: Patas de una ``mleg``.
    """

    id: str
    status: str
    qty: float
    filled_qty: float = 0.0
    filled_avg_price: float = float("nan")
    client_order_id: str | None = None
    symbol: str | None = None
    side: str | None = None
    limit_price: float = float("nan")
    legs: tuple["OrderSnapshot", ...] = ()

    @property
    def is_filled(self) -> bool:
        """Llenada por completo."""
        return self.status == "filled" or (self.qty > 0 and self.filled_qty >= self.qty)

    @property
    def is_final(self) -> bool:
        """No va a cambiar más."""
        return self.status in FINAL_STATUSES


def snapshot_from_order(order: Any) -> OrderSnapshot:
    """Convierte un ``Order`` de ``alpaca-py`` en :class:`OrderSnapshot`."""
    status = getattr(order.status, "value", order.status)
    side = getattr(order, "side", None)
    filled = _number(getattr(order, "filled_qty", None))
    return OrderSnapshot(
        id=str(order.id),
        status=str(status),
        qty=_number(getattr(order, "qty", None)),
        filled_qty=0.0 if math.isnan(filled) else filled,
        filled_avg_price=_number(getattr(order, "filled_avg_price", None)),
        client_order_id=getattr(order, "client_order_id", None),
        symbol=getattr(order, "symbol", None),
        side=str(getattr(side, "value", side)) if side is not None else None,
        limit_price=_number(getattr(order, "limit_price", None)),
        legs=tuple(snapshot_from_order(leg) for leg in (getattr(order, "legs", None) or [])),
    )


class BrokerRejection(RuntimeError):
    """El broker rechazó la operación. Conserva el código y el mensaje de Alpaca."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class Broker(Protocol):
    """Operaciones que la capa de ejecución necesita del broker."""

    def submit(self, request: Any) -> OrderSnapshot: ...

    def get_order(self, order_id: str) -> OrderSnapshot: ...

    def cancel(self, order_id: str) -> None: ...

    def positions(self) -> dict[str, float]: ...

    def account(self) -> Any: ...

    def clock(self) -> Any: ...


def _error_code(exc: Exception) -> int | None:
    try:
        return int(getattr(exc, "code"))
    except Exception:  # noqa: BLE001 - el código puede no venir en formato JSON
        try:
            return int(json.loads(str(exc)).get("code"))
        except Exception:  # noqa: BLE001
            return None


def _error_message(exc: Exception) -> str:
    try:
        message = getattr(exc, "message")
        if message:
            return str(message)
    except Exception:  # noqa: BLE001
        pass
    return str(exc)


class AlpacaBroker:
    """Implementación de :class:`Broker` sobre ``TradingClient``."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def submit(self, request: Any) -> OrderSnapshot:
        """Envía una orden; los rechazos de la API salen como :class:`BrokerRejection`."""
        from alpaca.common.exceptions import APIError

        try:
            order = self._client.submit_order(request)
        except APIError as exc:
            raise BrokerRejection(_error_message(exc), _error_code(exc)) from exc
        return snapshot_from_order(order)

    def get_order(self, order_id: str) -> OrderSnapshot:
        return snapshot_from_order(self._client.get_order_by_id(order_id))

    def cancel(self, order_id: str) -> None:
        """Cancela; si la orden ya estaba en estado final no hace nada."""
        from alpaca.common.exceptions import APIError

        try:
            self._client.cancel_order_by_id(order_id)
        except APIError as exc:
            if _error_code(exc) == ALREADY_FINAL_CODE:
                return
            raise BrokerRejection(_error_message(exc), _error_code(exc)) from exc

    def positions(self) -> dict[str, float]:
        """Posición con signo por símbolo: acciones, o contratos de opciones."""
        return {p.symbol: float(p.qty) for p in self._client.get_all_positions()}

    def account(self) -> Any:
        return self._client.get_account()

    def clock(self) -> Any:
        return self._client.get_clock()
