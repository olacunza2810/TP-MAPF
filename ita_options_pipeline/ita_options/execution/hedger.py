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

Modo ``delta_gamma``
--------------------
Antes de la delta se cubre la gamma del libro con la call ATM del vencimiento
más corto (al menos 14 días, distinto de los de las patas del libro):

.. math::

    n^* = -\frac{\Gamma_B}{100\,\Gamma_H}, \qquad
    n = \max\big(\mathrm{round}(n^*),\, 0\big),

y la delta que queda se cubre con acciones,
:math:`h^* = -\mathrm{round}(\Delta_B + 100\, n\, \Delta_H)`. Dos límites
reales de la cuenta: los contratos son enteros, y con nivel 3 de opciones no se
puede **vender** una opción descubierta, así que la cobertura sólo compra
opciones (o vende las que ya compró). Un libro con gamma larga (``n* < 0``)
queda cubierto sólo en delta. La opción se mantiene mientras siga a menos de 5%
del dinero y le queden 7 días; si no, se rota. El backtest del paso 4 mide lo
que cuesta: con los spreads de estas opciones, armar y desarmar la cobertura de
gamma cuesta más que la reducción de riesgo que compra.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

import pandas as pd
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

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
        mode: ``none`` sólo informa las griegas; ``delta`` cubre con acciones;
            ``delta_gamma`` compra la call ATM corta para la gamma y cubre la
            delta restante con acciones.
        band_shares: Desvío mínimo (acciones) entre objetivo y posición para rebalancear.
        max_shares: Tope de acciones por libro (protección ante una griega rota).
        max_option_contracts: Tope de contratos de la opción de cobertura por libro.
        min_hedge_dte: Plazo mínimo de la opción de cobertura al elegirla.
        steps: Pasos del árbol para las griegas del binomial.
    """

    mode: Literal["none", "delta", "delta_gamma"] = "none"
    band_shares: float = 10.0
    max_shares: float = 1000.0
    max_option_contracts: int = 5
    min_hedge_dte: int = 14
    steps: int = 100


def model_of_detector(detector: str | None) -> str | None:
    """``rv_crr_vertical`` -> ``crr``; ``rv_bsm_butterfly`` -> ``bsm``; otro -> ``None``."""
    parts = str(detector or "").split("_")
    if len(parts) >= 2 and parts[0] == "rv" and parts[1] in ("crr", "bsm"):
        return parts[1]
    return None


class _SmileCache:
    """Sonrisas por (modelo, subyacente, vencimiento) sobre la cadena del ciclo."""

    def __init__(self, chain: pd.DataFrame, steps: int) -> None:
        self.fit_base = chain.loc[chain["is_tradable"]] if "is_tradable" in chain else chain
        self.steps = steps
        self._cache: dict[tuple, Any] = {}

    def greeks(self, model: str, row: Mapping[str, Any]) -> Any | None:
        key = (model, row["underlying"], row["expiration"])
        if key not in self._cache:
            part = self.fit_base.loc[(self.fit_base["underlying"] == row["underlying"])
                                     & (self.fit_base["expiration"] == row["expiration"])]
            self._cache[key] = fit_smile(part, model, RVSettings(model=model, steps=self.steps)) if not part.empty else None
        smile = self._cache[key]
        if smile is None:
            return None
        divs = row_dividends(pd.Series(row)) or ()
        try:
            return model_greeks(model, float(row["underlying_price"]), float(row["strike"]), float(row["tau"]),
                                float(row["risk_free_rate"]), smile.sigma(float(row["strike"])),
                                str(row["option_type"]), divs, steps=self.steps)
        except ValueError:
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
    cache = _SmileCache(chain, steps)
    agg: dict[tuple[str, str], dict[str, float]] = {}
    missing: list[str] = []
    for strategy in strategies:
        model = model_of_detector(strategy.get("detector"))
        if model is None:
            continue
        contracts = int(strategy.get("contracts_open") or 0)
        key = (model, str(strategy.get("underlying")))
        book = agg.setdefault(key, {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "patas": 0,
                                    "faltantes": 0, "vencimientos": set()})
        for leg in strategy["legs"]:
            units = float(leg["units"]) * contracts
            if leg["kind"] == "stock":
                book["delta"] += units
                continue
            if leg["symbol"] not in by_symbol.index:
                missing.append(leg["symbol"])
                book["faltantes"] += 1
                continue
            row = by_symbol.loc[leg["symbol"]].to_dict()
            row["symbol"] = leg["symbol"]
            g = cache.greeks(model, row)
            if g is None:
                missing.append(leg["symbol"])
                book["faltantes"] += 1
                continue
            book["vencimientos"].add(row["expiration"])
            scale = 100.0 * units
            book["delta"] += scale * g.delta
            book["gamma"] += scale * g.gamma
            book["vega"] += scale * g.vega
            book["theta"] += scale * g.theta
            book["patas"] += 1
    rows = [{"modelo": m, "underlying": u, **v} for (m, u), v in agg.items()]
    return pd.DataFrame(rows), missing


def pick_gamma_option(chain: pd.DataFrame, underlying: str, exclude: set, keep: str | None = None,
                      min_dte: int = 14) -> pd.Series | None:
    """Call ATM del vencimiento más corto con al menos ``min_dte`` días, fuera de ``exclude``.

    Si ``keep`` (la opción que ya se tiene) sigue cotizando, le quedan 7 días y
    está a menos de 5% del dinero, se la mantiene: rotar a la ATM exacta en cada
    ciclo pagaría el spread entero.
    """
    base = chain.loc[(chain["underlying"] == underlying) & (chain["option_type"] == "c")
                     & (chain["bid"] > 0) & (chain["ask"] > chain["bid"])]
    if keep is not None:
        held = base.loc[base["symbol"] == keep]
        if not held.empty:
            row = held.iloc[0]
            if int(row["dte"]) >= 7 and abs(math.log(float(row["strike"]) / float(row["underlying_price"]))) <= 0.05:
                return row
    pool = base.loc[(base["dte"] >= min_dte) & ~base["expiration"].isin(exclude)]
    if "is_tradable" in pool:
        pool = pool.loc[pool["is_tradable"]]
    if pool.empty:
        return None
    first = pool.loc[pool["dte"] == pool["dte"].min()]
    k = (first["strike"] / first["underlying_price"]).map(math.log).abs()
    return first.loc[k.idxmin()]


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

    def _send_option(self, model: str, underlying: str, row: Mapping[str, Any], contracts: int) -> int:
        """Orden límite marcable por ``contracts`` (con signo) de la opción de cobertura.

        Compra al ask y vende al bid (cruza el spread, como en el backtest); si no
        llena en ``option_fill_timeout_s`` se cancela. Devuelve los contratos llenados.
        """
        if contracts == 0:
            return 0
        buy = contracts > 0
        price = float(row["ask"] if buy else row["bid"])
        request = LimitOrderRequest(
            symbol=str(row["symbol"]), qty=abs(int(contracts)),
            side=OrderSide.BUY if buy else OrderSide.SELL, time_in_force=TimeInForce.DAY,
            limit_price=round(price, 2), client_order_id=self._order_id(model, underlying),
        )
        book_id = f"hedge:{model}:{underlying}"
        try:
            snapshot = self.broker.submit(request)
        except BrokerRejection as exc:
            self.ledger.record_order(book_id, "hedge_gamma", request, error=exc.message)
            self.ledger.event("cobertura_rechazada", {"libro": book_id, "opcion": row["symbol"],
                                                     "contratos": contracts, "error": exc.message})
            return 0
        self.ledger.record_order(book_id, "hedge_gamma", request, snapshot)
        snapshot = wait_for_fill(self.broker, self.ledger, snapshot, self.execution.option_fill_timeout_s,
                                 self.execution.order_poll_s, self.timing)
        if not snapshot.is_filled:
            snapshot = cancel_and_settle(self.broker, self.ledger, snapshot, self.execution.order_poll_s, self.timing)
        filled = int(math.copysign(snapshot.filled_qty, contracts))
        if filled:
            fill = snapshot.filled_avg_price
            fill = price if fill is None or not math.isfinite(float(fill)) else float(fill)
            expiration = row.get("expiration")
            self.ledger.record_hedge_option_fill(model, underlying, str(row["symbol"]),
                                                 None if expiration is None else str(pd.Timestamp(expiration).date()),
                                                 filled, fill)
        return filled

    def _gamma_leg(self, model: str, und: str, g: Mapping[str, Any], chain: pd.DataFrame,
                   cache: "_SmileCache", flatten: bool) -> tuple[float, dict]:
        """Ajusta la opción de cobertura del libro. Devuelve la delta que aporta y el detalle."""
        held = self.ledger.hedge_options(model, und)
        by_symbol = chain.drop_duplicates("symbol", keep="last").set_index("symbol") if not chain.empty else None
        detail: dict[str, Any] = {"opcion": None, "n_objetivo_continuo": 0.0, "n_objetivo": 0, "n_operado": 0}
        keep = held[0]["symbol"] if held else None
        choice = None if flatten else pick_gamma_option(chain, und, set(g.get("vencimientos", set())), keep,
                                                        self.settings.min_hedge_dte)
        g_h = cache.greeks(model, choice.to_dict()) if choice is not None else None
        target = 0
        if choice is not None and g_h is not None and g_h.gamma > 1e-12:
            n_star = -float(g["gamma"]) / (100.0 * g_h.gamma)
            target = int(min(max(round(n_star), 0), self.settings.max_option_contracts))
            detail.update(opcion=str(choice["symbol"]), n_objetivo_continuo=n_star, n_objetivo=target)
            if n_star < -0.5:
                detail["limitacion"] = "gamma larga: cubrirla exige vender opciones descubiertas"
        trade_ok = self.settings.mode == "delta_gamma" and not self.dry_run
        # Rotación o cierre: se venden las opciones que ya no son la elegida.
        for h in held:
            if choice is not None and h["symbol"] == choice["symbol"]:
                continue
            if by_symbol is None or h["symbol"] not in by_symbol.index:
                self.ledger.event("cobertura_sin_cotizacion", {"libro": f"{model}:{und}", "opcion": h["symbol"]})
                continue
            if trade_ok:
                row = by_symbol.loc[h["symbol"]].to_dict()
                row["symbol"] = h["symbol"]
                detail["n_operado"] += self._send_option(model, und, row, -int(h["contracts"]))
        current = sum(float(h["contracts"]) for h in self.ledger.hedge_options(model, und)
                      if choice is not None and h["symbol"] == choice["symbol"])
        if choice is not None and trade_ok and target != current:
            detail["n_operado"] += self._send_option(model, und, choice.to_dict(), int(target - current))
        # Delta de las opciones de cobertura que efectivamente quedaron.
        delta = 0.0
        for h in self.ledger.hedge_options(model, und):
            if by_symbol is None or h["symbol"] not in by_symbol.index:
                continue
            row = by_symbol.loc[h["symbol"]].to_dict()
            row["symbol"] = h["symbol"]
            gh = cache.greeks(model, row)
            if gh is not None:
                delta += 100.0 * float(h["contracts"]) * gh.delta
        return delta, detail

    def hedge_value(self, spots: Mapping[str, float], quotes: pd.DataFrame | None = None) -> float:
        """Valor de mercado de los libros: acciones al spot, opciones al mid, más el efectivo acumulado."""
        total = 0.0
        for book in self.ledger.hedge_books():
            spot = spots.get(book["underlying"])
            if spot is None or not math.isfinite(float(spot)):
                continue
            total += float(book["shares"]) * float(spot) + float(book["cash"])
        for option in self.ledger.hedge_options():
            total += float(option["cash"])
            if quotes is not None and option["symbol"] in quotes.index:
                q = quotes.loc[option["symbol"]]
                total += float(option["contracts"]) * 100.0 * 0.5 * (float(q["bid"]) + float(q["ask"]))
        return total

    def run(self, strategies: list[Mapping[str, Any]], chain: pd.DataFrame,
            spots: Mapping[str, float], shorting: bool = True, flatten: bool = False) -> pd.DataFrame:
        """Calcula las griegas, las registra y, según el modo, rebalancea.

        Args:
            flatten: Cierra todos los libros (kill-switch o pérdida diaria).
                Las opciones de cobertura se venden con la cadena que se pase.

        Returns:
            Tabla por libro con griegas, objetivo, posición y lo operado.
        """
        greeks, missing = (pd.DataFrame(), []) if flatten else portfolio_greeks(strategies, chain, self.settings.steps)
        cache = _SmileCache(chain, self.settings.steps) if not chain.empty else None
        books = {(b["model"], b["underlying"]): b for b in self.ledger.hedge_books()}
        keys = set(books) | {(o["model"], o["underlying"]) for o in self.ledger.hedge_options()}
        if not greeks.empty:
            keys |= set(zip(greeks["modelo"], greeks["underlying"]))
        rows = []
        for model, und in sorted(keys):
            mask = ((greeks["modelo"] == model) & (greeks["underlying"] == und)) if not greeks.empty else None
            g = (greeks.loc[mask].iloc[0].to_dict() if mask is not None and mask.any()
                 else {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "patas": 0, "faltantes": 0,
                       "vencimientos": set()})
            incomplete = bool(g.get("faltantes", 0)) and not flatten
            gamma_delta, detail = 0.0, {}
            if self.settings.mode == "delta_gamma" and cache is not None and not incomplete:
                gamma_delta, detail = self._gamma_leg(model, und, g, chain, cache, flatten)
            current = float(books.get((model, und), {}).get("shares", 0.0))
            target = 0.0 if flatten else float(-round(g["delta"] + gamma_delta))
            target = max(-self.settings.max_shares, min(self.settings.max_shares, target))
            trade = target - current
            action = "sin_cambio"
            if incomplete:
                action = "faltan_patas"
            elif abs(trade) > 0 and (abs(trade) >= self.settings.band_shares or target == 0):
                action = "rebalanceo"
            done = 0.0
            spot = float(spots.get(und, float("nan")))
            if action == "rebalanceo" and self.settings.mode in ("delta", "delta_gamma") and not self.dry_run:
                done = self._trade(model, und, trade, spot, shorting)
            rows.append({"modelo": model, "underlying": und,
                         **{k: g[k] for k in ("delta", "gamma", "vega", "theta")},
                         "patas": g.get("patas", 0), "faltantes": g.get("faltantes", 0),
                         "delta_opcion_cobertura": gamma_delta, "objetivo": target, "posicion": current,
                         "accion": action, "operado": done, **{f"gamma_{k}": v for k, v in detail.items()}})
        table = pd.DataFrame(rows)
        if not table.empty:
            self.ledger.event("griegas_portafolio", {"libros": table.to_dict(orient="records"),
                                                     "faltantes": sorted(set(missing)), "modo": self.settings.mode})
        return table
