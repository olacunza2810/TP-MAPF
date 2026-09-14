"""Broker falso que reproduce lo observado en la cuenta paper de Alpaca.

- Las órdenes limit de acciones pueden llenar parcialmente (``stock_fill_qty``).
- Una ``mleg`` que vende calls sin acciones o calls largos que los cubran se
  rechaza con ``40310000``, como en ``paper_probe_20260914T175808Z.json``.
- La posición en acciones puede tardar en verse (``position_lag_calls``).
- Las órdenes a mercado llenan completas salvo ``market_fill=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from alpaca.trading.enums import OrderClass, OrderSide, OrderType

from ita_options.execution.broker import BrokerRejection, OrderSnapshot, UNCOVERED_OPTIONS_CODE


@dataclass
class SimTime:
    """Reloj simulado: ``sleep`` avanza el tiempo sin esperar."""

    t: float = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


@dataclass
class FakeBroker:
    stock_fill_qty: float | None = None
    option_fill_units: int | None = None
    reject_mleg: str | None = None
    enforce_coverage: bool = True
    position_lag_calls: int = 0
    market_fill: bool = True
    market_price: float = 100.0
    is_open: bool = True
    next_close: Any = None
    account_info: Any = field(default_factory=lambda: SimpleNamespace(
        options_buying_power="100000", shorting_enabled=True))
    submitted: list[Any] = field(default_factory=list)
    _orders: dict[str, OrderSnapshot] = field(default_factory=dict)
    _positions: dict[str, float] = field(default_factory=dict)
    _lagged: list[list[Any]] = field(default_factory=list)

    # -- utilidades ----------------------------------------------------------

    def _apply(self, symbol: str, delta: float, lag: bool = False) -> None:
        if lag and self.position_lag_calls > 0:
            self._lagged.append([self.position_lag_calls, symbol, delta])
            self._positions.setdefault(f"__real__{symbol}", 0.0)
            self._positions[f"__real__{symbol}"] += delta
            return
        self._positions[symbol] = self._positions.get(symbol, 0.0) + delta
        if self._positions[symbol] == 0:
            del self._positions[symbol]

    def _real_shares(self, symbol: str) -> float:
        return self._positions.get(symbol, 0.0) + self._positions.get(f"__real__{symbol}", 0.0)

    def kinds(self) -> list[str]:
        out = []
        for request in self.submitted:
            if request.order_class == OrderClass.MLEG:
                out.append("mleg")
            elif request.type == OrderType.MARKET:
                out.append(f"market_{request.side.value}")
            else:
                out.append(f"limit_{request.side.value}")
        return out

    # -- protocolo Broker ----------------------------------------------------

    def submit(self, request: Any) -> OrderSnapshot:
        order_id = f"o{len(self._orders) + 1}"
        qty = float(request.qty)
        if request.order_class == OrderClass.MLEG:
            if self.reject_mleg:
                raise BrokerRejection(self.reject_mleg, UNCOVERED_OPTIONS_CODE)
            if self.enforce_coverage:
                short_calls = sum(l.ratio_qty for l in request.legs
                                  if l.side == OrderSide.SELL and "C" in l.symbol[-9:-8])
                long_calls = sum(l.ratio_qty for l in request.legs
                                 if l.side == OrderSide.BUY and "C" in l.symbol[-9:-8])
                underlying = request.legs[0].symbol[:-15]
                covered_by_shares = max(self._real_shares(underlying), 0.0) // 100
                if (short_calls - long_calls) * qty > covered_by_shares:
                    raise BrokerRejection(
                        "account not eligible to trade uncovered option contracts",
                        UNCOVERED_OPTIONS_CODE,
                    )
            self.submitted.append(request)
            filled = qty if self.option_fill_units is None else float(min(self.option_fill_units, qty))
            for leg in request.legs:
                sign = 1.0 if leg.side == OrderSide.BUY else -1.0
                if filled:
                    self._apply(leg.symbol, sign * leg.ratio_qty * filled)
            status = "filled" if filled == qty else ("partially_filled" if filled else "new")
            snapshot = OrderSnapshot(order_id, status, qty, filled,
                                     float(request.limit_price) if filled else float("nan"),
                                     request.client_order_id, None, None,
                                     float(request.limit_price))
        else:
            self.submitted.append(request)
            sign = 1.0 if request.side == OrderSide.BUY else -1.0
            if request.type == OrderType.MARKET:
                filled = qty if self.market_fill else 0.0
                price = self.market_price
            else:
                filled = qty if self.stock_fill_qty is None else float(min(self.stock_fill_qty, qty))
                price = float(request.limit_price)
            if filled:
                self._apply(request.symbol, sign * filled, lag=True)
            status = "filled" if filled == qty else ("partially_filled" if filled else "new")
            snapshot = OrderSnapshot(order_id, status, qty, filled,
                                     price if filled else float("nan"),
                                     request.client_order_id, request.symbol, request.side.value,
                                     float(getattr(request, "limit_price", None) or float("nan")))
        self._orders[order_id] = snapshot
        return snapshot

    def get_order(self, order_id: str) -> OrderSnapshot:
        return self._orders[order_id]

    def cancel(self, order_id: str) -> None:
        snapshot = self._orders[order_id]
        if not snapshot.is_filled and not snapshot.is_final:
            self._orders[order_id] = OrderSnapshot(**{**snapshot.__dict__, "status": "canceled"})

    def positions(self) -> dict[str, float]:
        for item in self._lagged:
            item[0] -= 1
        ready = [item for item in self._lagged if item[0] <= 0]
        self._lagged = [item for item in self._lagged if item[0] > 0]
        for _, symbol, delta in ready:
            self._positions[f"__real__{symbol}"] -= delta
            if self._positions[f"__real__{symbol}"] == 0:
                del self._positions[f"__real__{symbol}"]
            self._apply(symbol, delta)
        return {s: q for s, q in self._positions.items() if not s.startswith("__real__")}

    def account(self) -> Any:
        return self.account_info

    def clock(self) -> Any:
        return SimpleNamespace(is_open=self.is_open, next_close=self.next_close)
