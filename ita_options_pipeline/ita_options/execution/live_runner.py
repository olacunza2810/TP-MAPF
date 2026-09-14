"""Loop intradía del piloto de paper trading.

Cada ciclo, con el mercado abierto:

1. **Datos:** spot, universo (una vez por rueda) y snapshots; arma la cadena.
2. **Corte manual:** si existe ``kill_file``, detiene entradas y cierra todo.
3. **Gestión de posiciones:** marca cada estructura con el lado de cierre del
   NBBO y la cierra por kill-switch, por vencer en la rueda o por un cierre
   pendiente. Si la pérdida diaria supera el límite, detiene entradas y cierra
   todo.
4. **Reconciliación:** compara el ledger con las posiciones del broker; si hay
   diferencias varios ciclos seguidos, detiene entradas.
5. **Entradas:** dentro de la ventana de la rueda, detecta, deduplica por
   estructura, aplica los controles previos y enruta. En ``dry_run`` sólo
   registra las órdenes que habría enviado.

Con el feed ``indicative`` el piloto valida la cañería (órdenes, estado,
reconciliación, kill-switch), no el edge: los quotes llegan demorados y el
paper llena contra el NBBO en tiempo real. El paper tampoco cobra dividendos
ni costo de préstamo. El reporte de sesión lo advierte.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest

from ..arbitrage import ExecutionCosts
from ..config import ExecutionSettings, LiquidityThresholds, PricingAssumptions
from .broker import Broker
from .ledger import Ledger, reconcile, structure_key
from .live_data import MarketData, build_live_chain, detect_live
from .orders import OrderConstructionError, build_mleg_order, mleg_limit_price, scaled_net_debit
from .parity import ParitySequencer
from .risk import pre_trade_check
from .router import Timing, close_options, execute_mleg

__all__ = ["CycleReport", "LiveRunner", "mark_value", "write_session_report"]

NY = ZoneInfo("America/New_York")
_LOG = logging.getLogger(__name__)

SESSION_WARNINGS = [
    "Feed indicative: los quotes llegan demorados y el paper llena contra NBBO en "
    "tiempo real. El P&L valida la cañería de órdenes, no mide edge.",
    "El paper de Alpaca no simula slippage, impacto, dividendos ni costo de préstamo.",
]


@dataclass
class CycleReport:
    """Resumen de un ciclo."""

    at: str
    phase: str
    chain_rows: int = 0
    tradable: int = 0
    signals: int = 0
    pretrade_rejections: int = 0
    entries_attempted: int = 0
    entries_opened: int = 0
    closes: int = 0
    daily_pnl: float = 0.0
    reconciliation_diffs: int = 0
    halted: str | None = None
    exclusions: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def mark_value(
    legs: list[Mapping[str, Any]], contracts: int, quotes: pd.DataFrame,
    spots: Mapping[str, float],
) -> float | None:
    """Valor de liquidar la estructura ahora, cruzando el spread.

    Las patas largas se valúan al bid y las cortas al ask; la acción, al spot.
    Devuelve ``None`` si falta algún precio.
    """
    total = 0.0
    for leg in legs:
        units = float(leg["units"]) * contracts
        if leg["kind"] == "stock":
            price = spots.get(leg["symbol"])
            if price is None or not math.isfinite(float(price)):
                return None
            total += units * float(price)
            continue
        if leg["symbol"] not in quotes.index:
            return None
        row = quotes.loc[leg["symbol"]]
        price = float(row["bid"] if units > 0 else row["ask"])
        if not math.isfinite(price) or price <= 0:
            return None
        total += units * price * 100.0
    return total


class LiveRunner:
    """Orquesta los ciclos del piloto."""

    def __init__(
        self,
        broker: Broker,
        market_data: MarketData,
        ledger: Ledger,
        settings: ExecutionSettings,
        costs: ExecutionCosts | None = None,
        liquidity: LiquidityThresholds | None = None,
        pricing: PricingAssumptions | None = None,
        dividends: pd.DataFrame | None = None,
        dry_run: bool = True,
        report_dir: Path = Path("reportes/paper"),
        timing: Timing | None = None,
        now: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        self.broker = broker
        self.market = market_data
        self.ledger = ledger
        self.settings = settings
        self.costs = costs or ExecutionCosts(min_net_edge=settings.min_edge_usd)
        self.liquidity = liquidity or LiquidityThresholds()
        self.pricing = pricing or PricingAssumptions()
        self.dividends = dividends
        self.dry_run = dry_run
        self.report_dir = Path(report_dir)
        self.timing = timing or Timing()
        self._now = now or (lambda: pd.Timestamp.now(tz="UTC"))
        self.halted_reason: str | None = None
        self.cycles: list[CycleReport] = []
        self._master: pd.DataFrame | None = None
        self._master_date: date | None = None
        self._errors = 0
        self._reconcile_strikes = 0
        self._parity = ParitySequencer(broker, ledger, settings, self.costs, self.timing)

    # -- control ------------------------------------------------------------

    def _halt(self, reason: str) -> None:
        if self.halted_reason is None:
            self.halted_reason = reason
            self.ledger.event("halt", {"reason": reason})

    def _entry_window(self, now: pd.Timestamp, clock: Any) -> bool:
        local = now.tz_convert(NY)
        session_open = local.normalize() + pd.Timedelta(hours=9, minutes=30)
        next_close = pd.Timestamp(clock.next_close)
        next_close = next_close.tz_localize("UTC") if next_close.tz is None else next_close
        after_open = (local - session_open) >= pd.Timedelta(minutes=self.settings.entry_start_minutes)
        before_close = (next_close.tz_convert(NY) - local) > pd.Timedelta(
            minutes=self.settings.entry_stop_minutes)
        return bool(after_open and before_close)

    def _minutes_to_close(self, now: pd.Timestamp, clock: Any) -> float:
        next_close = pd.Timestamp(clock.next_close)
        next_close = next_close.tz_localize("UTC") if next_close.tz is None else next_close
        return (next_close - now).total_seconds() / 60.0

    def _open_symbols(self) -> set[str]:
        return {
            leg["symbol"] for strategy in self.ledger.open_strategies()
            for leg in strategy["legs"] if leg["kind"] == "option"
        }

    # -- ciclo --------------------------------------------------------------

    def run_cycle(self) -> CycleReport:
        """Ejecuta un ciclo completo."""
        now = self._now()
        today = now.tz_convert(NY).date()
        report = CycleReport(at=now.isoformat(), phase="open")
        try:
            clock = self.broker.clock()
            if not clock.is_open:
                report.phase = "closed"
                self.cycles.append(report)
                return report
            spots = self.market.spots()
            if self._master is None or self._master_date != today:
                self._master = self.market.universe(spots)
                self._master_date = today
                self.ledger.event("universe", {"contracts": len(self._master)})
            symbols = sorted(set(self._master["symbol"]) | self._open_symbols())
            quotes = self.market.quotes(symbols)
            bars = self.market.bars(now)
            chain, filters = build_live_chain(quotes, self._master, spots, self.pricing,
                                              self.liquidity, now, self.settings.max_quote_age_s,
                                              bars=bars)
            if filters is not None:
                report.exclusions = dict(filters.exclusions)
                report.exclusions["quote_viejo"] = int(chain["excl_quote_viejo"].sum())
            self._errors = 0
        except Exception as exc:  # noqa: BLE001 - un ciclo fallido no detiene el loop
            self._errors += 1
            self.ledger.event("cycle_error", {"error": repr(exc), "consecutive": self._errors})
            # Un error de ciclo nunca debe pasar en silencio: el loop sigue, pero
            # quien lo mira tiene que verlo en la consola.
            _LOG.warning("Ciclo fallido (%d seguidos de %d permitidos): %r",
                         self._errors, self.settings.max_consecutive_errors, exc)
            if self._errors >= self.settings.max_consecutive_errors:
                self._halt("errores_consecutivos")
            report.phase = "error"
            report.halted = self.halted_reason
            self.cycles.append(report)
            return report

        report.chain_rows = len(chain)
        report.tradable = int(chain["is_tradable"].sum()) if not chain.empty else 0
        quote_index = (
            quotes.drop_duplicates("symbol", keep="last").set_index("symbol")
            if not quotes.empty else pd.DataFrame(columns=["bid", "ask"])
        )

        try:
            if Path(self.settings.kill_file).exists():
                self._halt("kill_file")
                self._close_all("kill_file", quote_index, spots, report)

            self._manage_positions(quote_index, spots, clock, now, today, report)
            if not self.dry_run:
                self._reconcile(report)
            if self.halted_reason is None and self._entry_window(now, clock):
                self._enter(chain, today, report)
        except Exception as exc:  # noqa: BLE001 - nunca abandonar el loop con órdenes vivas
            # Una excepción acá puede ocurrir en medio de una secuencia (por
            # ejemplo, con las acciones ya llenadas y antes de enviar las
            # opciones). Morir dejaría patas sin registrar ni gestionar: se
            # registra, se detienen las entradas y el loop sigue marcando,
            # reconciliando y aplicando el kill-switch en los ciclos siguientes.
            import traceback

            self.ledger.event("execution_error", {"error": repr(exc),
                                                  "traceback": traceback.format_exc()})
            _LOG.error("Excepción durante la ejecución; se detienen las entradas: %r", exc)
            self._halt("excepcion_en_ejecucion")
            report.notes.append(repr(exc))
        report.halted = self.halted_reason
        self.cycles.append(report)
        return report

    def _manage_positions(
        self, quotes: pd.DataFrame, spots: Mapping[str, float], clock: Any,
        now: pd.Timestamp, today: date, report: CycleReport,
    ) -> None:
        unrealized_total = 0.0
        for strategy in self.ledger.open_strategies():
            value = mark_value(strategy["legs"], int(strategy["contracts_open"] or 0), quotes, spots)
            if value is None:
                self.ledger.event("sin_marca", None, strategy["id"])
                continue
            unrealized = float(strategy["entry_cash"] or 0) + float(strategy["exit_cash"] or 0) + value
            unrealized_total += unrealized
            expirations = [leg["expiration"] for leg in strategy["legs"] if leg.get("expiration")]
            expires_today = bool(expirations) and min(expirations) <= today.isoformat()
            reason = None
            if unrealized < -abs(self.settings.kill_switch_usd):
                reason = "kill_switch"
            elif expires_today and self._minutes_to_close(now, clock) <= self.settings.entry_stop_minutes:
                reason = "pre_expiry"
            elif strategy["state"] == "closing":
                reason = strategy["exit_reason"] or "cierre_pendiente"
            if reason:
                self._close(strategy, reason, quotes, report)

        report.daily_pnl = self.ledger.realized_pnl(today) + unrealized_total
        if report.daily_pnl < -abs(self.settings.daily_loss_limit_usd):
            self._halt("perdida_diaria")
            self._close_all("perdida_diaria", quotes, spots, report)

    def _close_all(
        self, reason: str, quotes: pd.DataFrame, spots: Mapping[str, float], report: CycleReport
    ) -> None:
        for strategy in self.ledger.open_strategies():
            self._close(strategy, reason, quotes, report)

    def _close(
        self, strategy: Mapping[str, Any], reason: str, quotes: pd.DataFrame, report: CycleReport
    ) -> None:
        if self.dry_run:
            return
        self.ledger.update_strategy(strategy["id"], state="closing", exit_reason=reason)
        orphan = False
        if strategy["route"] == "parity":
            units, cash, messages, orphan = self._parity.close(strategy, quotes)
        else:
            units, cash, messages = close_options(self.broker, self.ledger, strategy, quotes,
                                                  self.settings, self.timing)
        if orphan:
            self._halt("pata_huerfana")
        remaining = int(strategy["contracts_open"] or 0) - units
        exit_cash = float(strategy["exit_cash"] or 0) + cash
        if remaining <= 0:
            self.ledger.update_strategy(
                strategy["id"], state="closed", contracts_open=0, exit_cash=exit_cash,
                realized_pnl=float(strategy["entry_cash"] or 0) + exit_cash,
                closed_at=pd.Timestamp.now(tz="UTC").isoformat(),
                message="; ".join(messages) or None,
            )
            report.closes += 1
        else:
            self.ledger.update_strategy(
                strategy["id"], contracts_open=remaining, exit_cash=exit_cash,
                message="; ".join(messages) or None,
            )

    def _reconcile(self, report: CycleReport) -> None:
        roots = sorted(set(self._master["underlying"])) if self._master is not None else None
        result = reconcile(self.ledger, self.broker, roots)
        if result.ok:
            self._reconcile_strikes = 0
            return
        self._reconcile_strikes += 1
        report.reconciliation_diffs = len(result.diffs)
        self.ledger.event("reconciliation_diff", {s: list(v) for s, v in result.diffs.items()})
        if self._reconcile_strikes >= self.settings.reconcile_tolerance_cycles:
            self._halt("reconciliacion")

    def _planned_requests(self, signal: Mapping[str, Any], route: str) -> list[dict[str, Any]]:
        """Órdenes que se enviarían, para el registro del dry-run."""
        legs = list(signal["leg_spec"])
        option_legs = [leg for leg in legs if leg.get("kind") == "option"]
        contracts = self.settings.contracts
        planned = []
        try:
            if route == "parity":
                stock = next(leg for leg in legs if leg.get("kind") == "stock")
                direction = 1 if float(stock["qty"]) > 0 else -1
                planned.append(LimitOrderRequest(
                    symbol=str(stock["symbol"]), qty=100 * contracts,
                    side=OrderSide.BUY if direction > 0 else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(float(stock["price"]) + direction * self.settings.stock_limit_tolerance, 2),
                ).model_dump(mode="json", exclude_none=True))
            debit = scaled_net_debit(option_legs, [float(leg["price"]) for leg in option_legs])
            planned.append(build_mleg_order(option_legs, contracts, mleg_limit_price(debit))
                           .model_dump(mode="json", exclude_none=True))
        except (OrderConstructionError, StopIteration, ValueError) as exc:
            planned.append({"error": str(exc)})
        return planned

    def _enter(self, chain: pd.DataFrame, today: date, report: CycleReport) -> None:
        open_count = len(self.ledger.open_strategies())
        if open_count >= self.settings.max_open_strategies:
            return
        signals = detect_live(chain, self.costs, self.settings.detectors)
        report.signals = len(signals)
        if signals.empty:
            return
        account = self.broker.account()
        buying_power = float(getattr(account, "options_buying_power", 0) or 0)
        shorting = bool(getattr(account, "shorting_enabled", False))
        sizes = {
            str(row.symbol): (float(row.bid_size), float(row.ask_size))
            for row in chain[["symbol", "bid_size", "ask_size"]].itertuples(index=False)
        }

        for _, signal in signals.iterrows():
            if open_count >= self.settings.max_open_strategies:
                break
            key = structure_key(signal)
            if self.ledger.seen_structure(key, today):
                continue
            route = "parity" if signal["detector"] == "put_call_parity" else "mleg"
            strategy_id = self.ledger.create_strategy(signal, route, self.settings.contracts, today)
            decision = pre_trade_check(
                signal, self.settings.contracts, buying_power, self.settings.max_loss_per_trade,
                sizes, today, self.dividends, shorting,
            )
            if not decision.approved:
                self.ledger.update_strategy(strategy_id, state="rejected_pretrade",
                                            message="; ".join(decision.reasons))
                report.pretrade_rejections += 1
                continue
            report.entries_attempted += 1
            if self.dry_run:
                self.ledger.update_strategy(strategy_id, state="dry_run")
                self.ledger.event("dry_run_orders", self._planned_requests(signal, route), strategy_id)
                continue

            if route == "parity":
                outcome = self._parity.open(strategy_id, signal)
            else:
                outcome = execute_mleg(self.broker, self.ledger, strategy_id, signal,
                                       self.settings, self.costs, self.timing)
            self.ledger.update_strategy(
                strategy_id, state=outcome.state, contracts_open=outcome.contracts_open,
                entry_cash=outcome.entry_cash, residual_edge=outcome.residual_edge,
                message="; ".join(outcome.messages) or None,
            )
            if outcome.orphan:
                self._halt("pata_huerfana")
            if outcome.state == "open":
                open_count += 1
                report.entries_opened += 1

    # -- sesión -------------------------------------------------------------

    def run(self, max_cycles: int | None = None) -> Path:
        """Corre ciclos hasta que termine la rueda o se alcance ``max_cycles``."""
        cycles = 0
        seen_open = False
        try:
            while max_cycles is None or cycles < max_cycles:
                report = self.run_cycle()
                cycles += 1
                if report.phase in ("open", "error"):
                    seen_open = True
                elif report.phase == "closed" and seen_open:
                    break
                if max_cycles is not None and cycles >= max_cycles:
                    break
                self.timing.sleep(self.settings.cycle_seconds if report.phase != "closed" else 300.0)
        finally:
            session = self._now().tz_convert(NY).date()
            path = write_session_report(self.ledger, session, self.report_dir, self.cycles,
                                        self.dry_run, self.halted_reason)
        return path


def write_session_report(
    ledger: Ledger,
    session_date: date,
    out_dir: Path,
    cycles: list[CycleReport] | None = None,
    dry_run: bool = True,
    halted_reason: str | None = None,
) -> Path:
    """Escribe estrategias, órdenes, eventos y un resumen JSON de la rueda."""
    target = Path(out_dir) / f"session_{session_date.isoformat()}"
    target.mkdir(parents=True, exist_ok=True)
    strategies = pd.DataFrame(ledger.strategies(session_date=session_date))
    orders = pd.DataFrame(ledger.orders())
    events = pd.DataFrame(ledger.events())
    strategies.drop(columns=["legs"], errors="ignore").to_csv(target / "strategies.csv", index=False)
    orders.to_csv(target / "orders.csv", index=False)
    events.to_csv(target / "events.csv", index=False)
    if cycles:
        pd.DataFrame([asdict(c) for c in cycles]).to_csv(target / "cycles.csv", index=False)

    slippage = []
    for order in ledger.orders():
        request = json.loads(order["request_json"] or "{}")
        limit = request.get("limit_price")
        price = order.get("filled_avg_price")
        if order["role"] == "stock_open" and limit is not None and price is not None:
            sign = 1.0 if request.get("side") == "buy" else -1.0
            slippage.append(sign * (float(price) - float(limit)))

    summary = {
        "session_date": session_date.isoformat(),
        "dry_run": dry_run,
        "halted_reason": halted_reason,
        "cycles": len(cycles or []),
        "strategies_by_state": strategies["state"].value_counts().to_dict() if not strategies.empty else {},
        "strategies_by_detector": strategies["detector"].value_counts().to_dict() if not strategies.empty else {},
        "orders": len(orders),
        "rejected_orders": int(orders["error"].notna().sum()) if not orders.empty else 0,
        "hedge_failures": int((strategies["state"] == "hedge_failed").sum()) if not strategies.empty else 0,
        "orphan_events": len(ledger.events("orphan_shares")),
        "realized_pnl": ledger.realized_pnl(session_date),
        "open_strategies": len(ledger.open_strategies()),
        "stock_slippage_vs_limit_mean": (sum(slippage) / len(slippage)) if slippage else None,
        "warnings": SESSION_WARNINGS,
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2, default=str), "utf-8")
    return target
