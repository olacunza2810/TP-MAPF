r"""Cobertura delta en vivo con acciones, un libro por modelo (Parte 2, paso 4).

Cada estrategia abierta pertenece a un modelo (``rv_crr_*`` al binomial,
``rv_bsm_*`` a Black-Scholes). En cada ciclo:

1. Se ajusta la sonrisa de cada modelo sobre la cadena del ciclo y se calculan
   las griegas de cada pata con **su** modelo (dividendo discreto escrowed).
2. Se agregan por modelo y subyacente: :math:`\Delta_B = \sum 100\, u\, c\, \Delta_i`
   (``u`` unidades con signo de la pata, ``c`` contratos abiertos), y lo mismo
   para :math:`\Gamma`, :math:`\mathcal V` y :math:`\Theta`.
3. El objetivo del libro es :math:`h^* = -\mathrm{round}(\Delta_B)` acciones. Se
   rebalancea sólo si :math:`|h^* - h| \ge` ``band_shares``: rebalancear por cada
   acción paga costos sin reducir riesgo (la banda es la versión práctica de
   Leland, 1985). Con :math:`h^* = 0` se cierra el libro entero.

Cada modelo cubre sus propias posiciones con sus propias deltas: así la
competencia también compara la calidad de las griegas, no sólo la del precio.
Los dos libros comparten la cuenta; el broker ve la suma. Una orden que haría
cruzar por cero la posición neta del broker se parte en dos (primero a cero).

La gamma no se cubre en vivo: se informa cuántos contratos de la call ATM la
anularían. El backtest del paso 4 mostró que, con los spreads de estas opciones,
cubrir gamma con opciones cuesta más de lo que reduce el riesgo, y además las
estructuras son de 1 contrato (la cobertura redondeada da casi siempre 0) y una
venta de opción descubierta no se puede hacer con nivel 3 de opciones.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

import pandas as pd
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from ..config import ExecutionSettings
from ..escrowed import row_dividends
from ..greeks import model_greeks
from .broker import Broker, BrokerRejection
from .ledger import Ledger
from .model_rv import RVSettings, fit_smile
from .router import Timing, cancel_and_settle, wait_for_fill

__all__ = ["HedgeSettings", "model_of_detector", "portfolio_greeks", "DeltaHedger"]

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HedgeSettings:
    """Parámetros de la cobertura en vivo.

    Attributes:
        mode: ``none`` sólo informa las griegas; ``delta`` cubre con acciones.
        band_shares: Desvío mínimo (acciones) entre objetivo y posición para rebalancear.
        max_shares: Tope de acciones por libro (protección ante una griega rota).
        steps: Pasos del árbol para las griegas del binomial.
    """

    mode: Literal["none", "delta"] = "none"
    band_shares: float = 10.0
    max_shares: float = 1000.0
    steps: int = 100


def model_of_detector(detector: str | None) -> str | None:
    """``rv_crr_vertical`` -> ``crr``; ``rv_bsm_butterfly`` -> ``bsm``; otro -> ``None``."""
    parts = str(detector or "").split("_")
    if len(parts) >= 2 and parts[0] == "rv" and parts[1] in ("crr", "bsm"):
        return parts[1]
    return None


def portfolio_greeks(
    strategies: list[Mapping[str, Any]], chain: pd.DataFrame, steps: int = 100,
) -> tuple[pd.DataFrame, list[str]]:
    """Griegas agregadas por modelo y subyacente de las estrategias abiertas.

    Returns:
        ``(griegas, faltantes)``: una fila por ``(modelo, underlying)`` con
        ``delta`` (acciones equivalentes), ``gamma`` (acciones por USD),
        ``vega`` (USD por 1,00 de vol), ``theta`` (USD por año) y ``patas``; y
        la lista de símbolos que no se pudieron valuar (sin cotización o sin
        sonrisa). Un libro con faltantes no se rebalancea en ese ciclo.
    """
    if chain.empty:
        return pd.DataFrame(), []
    by_symbol = chain.drop_duplicates("symbol", keep="last").set_index("symbol")
    fit_base = chain.loc[chain["is_tradable"]] if "is_tradable" in chain else chain
    smiles: dict[tuple, Any] = {}
    agg: dict[tuple[str, str], dict[str, float]] = {}
    missing: list[str] = []
    for strategy in strategies:
        model = model_of_detector(strategy.get("detector"))
        if model is None:
            continue
        contracts = int(strategy.get("contracts_open") or 0)
        key = (model, str(strategy.get("underlying")))
        book = agg.setdefault(key, {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "patas": 0,
                                    "faltantes": 0})
        for leg in strategy["legs"]:
            units = float(leg["units"]) * contracts
            if leg["kind"] == "stock":
                book["delta"] += units
                continue
            if leg["symbol"] not in by_symbol.index:
                missing.append(leg["symbol"])
                book["faltantes"] += 1
                continue
            row = by_symbol.loc[leg["symbol"]]
            skey = (model, row["underlying"], row["expiration"])
            if skey not in smiles:
                part = fit_base.loc[(fit_base["underlying"] == row["underlying"])
                                    & (fit_base["expiration"] == row["expiration"])]
                smiles[skey] = fit_smile(part, model, RVSettings(model=model, steps=steps)) if not part.empty else None
            smile = smiles[skey]
            if smile is None:
                missing.append(leg["symbol"])
                book["faltantes"] += 1
                continue
            divs = row_dividends(row) or ()
            try:
                g = model_greeks(model, float(row["underlying_price"]), float(row["strike"]), float(row["tau"]),
                                 float(row["risk_free_rate"]), smile.sigma(float(row["strike"])),
                                 str(row["option_type"]), divs, steps=steps)
            except ValueError:
                missing.append(leg["symbol"])
                book["faltantes"] += 1
                continue
            scale = 100.0 * units
            book["delta"] += scale * g.delta
            book["gamma"] += scale * g.gamma
            book["vega"] += scale * g.vega
            book["theta"] += scale * g.theta
            book["patas"] += 1
    rows = [{"modelo": m, "underlying": u, **v} for (m, u), v in agg.items()]
    return pd.DataFrame(rows), missing


class DeltaHedger:
    """Rebalancea los libros de cobertura con acciones."""

    def __init__(
        self, broker: Broker, ledger: Ledger, settings: HedgeSettings, execution: ExecutionSettings,
        timing: Timing | None = None, dry_run: bool = True,
    ) -> None:
        self.broker = broker
        self.ledger = ledger
        self.settings = settings
        self.execution = execution
        self.timing = timing or Timing()
        self.dry_run = dry_run
        self._counter = 0

    # -- órdenes ------------------------------------------------------------

    def _order_id(self, model: str, underlying: str) -> str:
        self._counter += 1
        return f"hedge-{model}-{underlying}-{pd.Timestamp.now(tz='UTC'):%Y%m%d%H%M%S}-{self._counter}"

    def _send(self, model: str, underlying: str, shares: float, spot: float) -> float:
        """Orden a mercado por ``shares`` con signo; registra el fill en el libro. Devuelve lo llenado."""
        request = MarketOrderRequest(
            symbol=underlying, qty=abs(shares),
            side=OrderSide.BUY if shares > 0 else OrderSide.SELL,
            time_in_force=TimeInForce.DAY, client_order_id=self._order_id(model, underlying),
        )
        book_id = f"hedge:{model}:{underlying}"
        try:
            snapshot = self.broker.submit(request)
        except BrokerRejection as exc:
            self.ledger.record_order(book_id, "hedge_delta", request, error=exc.message)
            self.ledger.event("cobertura_rechazada", {"libro": book_id, "acciones": shares, "error": exc.message})
            return 0.0
        self.ledger.record_order(book_id, "hedge_delta", request, snapshot)
        snapshot = wait_for_fill(self.broker, self.ledger, snapshot, self.execution.unwind_timeout_s,
                                 self.execution.order_poll_s, self.timing)
        if not snapshot.is_filled:
            snapshot = cancel_and_settle(self.broker, self.ledger, snapshot, self.execution.order_poll_s, self.timing)
        filled = math.copysign(snapshot.filled_qty, shares)
        if filled:
            price = snapshot.filled_avg_price
            if price is None or not math.isfinite(float(price)):
                price = spot
                self.ledger.event("cobertura_sin_precio", {"libro": book_id, "acciones": filled, "spot": spot})
            self.ledger.record_hedge_fill(model, underlying, filled, float(price))
        return filled

    def _trade(self, model: str, underlying: str, shares: float, spot: float, shorting: bool) -> float:
        """Envía ``shares``; parte la orden si cruza por cero la posición neta del broker."""
        net = float(self.broker.positions().get(underlying, 0.0))
        if not shorting and net + shares < 0:
            # Sin venta en corto habilitada, la cobertura sólo puede bajar el largo hasta cero.
            allowed = -net if net > 0 else 0.0
            self.ledger.event("cobertura_limitada", {"libro": f"{model}:{underlying}", "pedido": shares,
                                                     "posible": allowed, "motivo": "sin venta en corto"})
            if allowed == 0:
                return 0.0
            shares = allowed
        if net != 0 and (net > 0) != (net + shares > 0) and net + shares != 0:
            first = self._send(model, underlying, -net, spot)
            if abs(first) < abs(net):
                return first
            return first + self._send(model, underlying, shares + net, spot)
        return self._send(model, underlying, shares, spot)

    # -- ciclo ----------------------------------------------------------------

    def hedge_value(self, spots: Mapping[str, float]) -> float:
        """Valor de mercado de los libros: acciones al spot más el efectivo acumulado."""
        total = 0.0
        for book in self.ledger.hedge_books():
            spot = spots.get(book["underlying"])
            if spot is None or not math.isfinite(float(spot)):
                continue
            total += float(book["shares"]) * float(spot) + float(book["cash"])
        return total

    def run(self, strategies: list[Mapping[str, Any]], chain: pd.DataFrame,
            spots: Mapping[str, float], shorting: bool = True, flatten: bool = False) -> pd.DataFrame:
        """Calcula las griegas, las registra y, en modo ``delta``, rebalancea.

        Args:
            flatten: Cierra todos los libros (kill-switch o pérdida diaria).

        Returns:
            Tabla por libro con griegas, objetivo, posición y acciones operadas.
        """
        greeks, missing = (pd.DataFrame(), []) if flatten else portfolio_greeks(strategies, chain, self.settings.steps)
        books = {(b["model"], b["underlying"]): b for b in self.ledger.hedge_books()}
        keys = set(books)
        if not greeks.empty:
            keys |= set(zip(greeks["modelo"], greeks["underlying"]))
        rows = []
        for model, und in sorted(keys):
            g = (greeks.loc[(greeks["modelo"] == model) & (greeks["underlying"] == und)].iloc[0].to_dict()
                 if not greeks.empty and ((greeks["modelo"] == model) & (greeks["underlying"] == und)).any()
                 else {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "patas": 0, "faltantes": 0})
            current = float(books.get((model, und), {}).get("shares", 0.0))
            target = 0.0 if flatten else float(-round(g["delta"]))
            target = max(-self.settings.max_shares, min(self.settings.max_shares, target))
            trade = target - current
            action = "sin_cambio"
            if g.get("faltantes", 0) and not flatten:
                action = "faltan_patas"
            elif abs(trade) > 0 and (abs(trade) >= self.settings.band_shares or target == 0):
                action = "rebalanceo"
            done = 0.0
            spot = float(spots.get(und, float("nan")))
            if action == "rebalanceo" and self.settings.mode == "delta" and not self.dry_run:
                done = self._trade(model, und, trade, spot, shorting)
            rows.append({"modelo": model, "underlying": und, **{k: g[k] for k in ("delta", "gamma", "vega", "theta")},
                         "patas": g.get("patas", 0), "faltantes": g.get("faltantes", 0),
                         "objetivo": target, "posicion": current, "accion": action, "operado": done})
        table = pd.DataFrame(rows)
        if not table.empty:
            self.ledger.event("griegas_portafolio", {"libros": table.to_dict(orient="records"),
                                                     "faltantes": sorted(set(missing)), "modo": self.settings.mode})
        return table
