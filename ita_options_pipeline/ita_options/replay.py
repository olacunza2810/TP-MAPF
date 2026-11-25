r"""Simulación histórica del sistema de producción (competencia + coberturas + riesgo).

El backtest de la Parte 1 (:mod:`ita_options.rv_backtest`) simula cada modelo
por separado, sin coberturas ni las reglas de riesgo del loop en vivo. Este
módulo reproduce el sistema que corre en paper, foto por foto:

- **Señales.** Las mismas funciones que en vivo (:func:`execution.model_rv.detect_model_rv`
  con dividendo discreto escrowed), una vez por foto y por modelo.
- **Competencia.** Hasta ``cap_per_model`` estructuras abiertas por modelo y
  ``max_open`` en total; la prioridad entre modelos se alterna; las mismas patas
  no se abren dos veces.
- **Llenado.** El de la Parte 1: la orden límite publicada en una foto se llena
  en la **siguiente** sólo si esa foto la cruza (el ask neto de la estructura,
  con su signo, queda en o por debajo del límite), al precio límite. Si no,
  se cancela.
- **Riesgo**, como en producción después del paso 5: kill-switch por estructura
  al mid, take-profit a liquidación, cierre antes del vencimiento y pérdida
  máxima entre fotos de la cuenta al mid (con la cobertura), que cierra todo.
- **Coberturas.** Las del loop en vivo (:mod:`execution.hedger`): griegas de
  cada libro con su modelo; delta con acciones (banda de 10) y, en
  ``delta_gamma``, la call ATM corta en contratos enteros y sólo compras.

Contabilidad: todo en USD por cuenta, empezando en 0. Las estructuras se
valúan al mid para el P&L diario y a liquidación (cruzando el spread) al
cerrarse; las acciones al spot implícito de la foto, las opciones de cobertura
al mid. El efectivo rinde la tasa del día y la acción cobra sus dividendos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .execution.hedger import _SmileCache, pick_gamma_option, portfolio_greeks

__all__ = ["ReplayRules", "Structure", "replay", "trade_stats"]

MULT = 100.0


@dataclass(frozen=True)
class ReplayRules:
    """Reglas de producción (``paper_daily.ps1``)."""

    cap_per_model: int = 3
    max_open: int = 6
    kill_switch_usd: float = 250.0
    take_profit_fraction: float = 0.6
    daily_loss_usd: float = 1000.0
    pre_expiry_days: int = 2
    hedge: Literal["sin_cobertura", "delta", "delta_gamma"] = "delta"
    band_shares: float = 10.0
    max_option_contracts: int = 5
    stock_cost_bps: float = 1.0
    option_commission: float = 0.65


@dataclass
class Structure:
    """Una orden de la simulación y, si se llenó, la posición resultante."""
    sid: int
    model: str
    detector: str
    underlying: str
    expiration: date
    legs: list[dict[str, Any]]          # symbol, qty (con signo), strike, option_type
    limit: float                        # precio neto por acción, con el signo de la estructura
    edge: float                         # USD anunciados contra el modelo
    commission: float
    opened: date | None = None
    entry_cash: float = 0.0
    closed: date | None = None
    exit_cash: float = 0.0
    reason: str | None = None
    last_mid: float = 0.0
    last_liq: float = 0.0

    @property
    def key(self) -> frozenset:
        """Patas con su cantidad: identifica estructuras iguales para no abrirlas dos veces."""
        return frozenset((leg["symbol"], leg["qty"]) for leg in self.legs)


def _quotes(day_chain: pd.DataFrame) -> dict[str, tuple[float, float]]:
    ok = day_chain.loc[(day_chain["bid"] >= 0) & (day_chain["ask"] > 0) & (day_chain["ask"] >= day_chain["bid"])]
    return dict(zip(ok["symbol"], zip(ok["bid"].astype(float), ok["ask"].astype(float))))


def _value(legs: Sequence[Mapping[str, Any]], quotes: Mapping[str, tuple[float, float]]) -> tuple[float, float] | None:
    """``(mid, liquidación)`` por acción de la estructura; ``None`` si falta una pata."""
    mid = liq = 0.0
    for leg in legs:
        q = quotes.get(leg["symbol"])
        if q is None:
            return None
        bid, ask = q
        mid += leg["qty"] * 0.5 * (bid + ask)
        liq += leg["qty"] * (bid if leg["qty"] > 0 else ask)
    return mid, liq


def _ask_net(legs: Sequence[Mapping[str, Any]], quotes: Mapping[str, tuple[float, float]]) -> float | None:
    """Lo que cuesta comprar la estructura con su signo, cruzando el spread."""
    total = 0.0
    for leg in legs:
        q = quotes.get(leg["symbol"])
        if q is None:
            return None
        total += leg["qty"] * (q[1] if leg["qty"] > 0 else q[0])
    return total


def _intrinsic(legs: Sequence[Mapping[str, Any]], spot: float) -> float:
    return sum(leg["qty"] * (max(spot - leg["strike"], 0.0) if leg["option_type"] == "c"
                             else max(leg["strike"] - spot, 0.0)) for leg in legs)


@dataclass
class _Book:
    shares: float = 0.0
    opt_symbol: str | None = None
    opt_contracts: float = 0.0
    opt_strike: float = 0.0
    opt_expiration: date | None = None
    opt_last_mid: float = 0.0


def replay(
    chains: Mapping[date, pd.DataFrame], signals: Mapping[date, Mapping[str, pd.DataFrame]],
    spots: Mapping[tuple[str, date], float], dividends: pd.DataFrame, rules: ReplayRules = ReplayRules(),
) -> dict[str, pd.DataFrame]:
    """Corre la cuenta foto por foto.

    Args:
        chains: Cadena curada de cada foto (con ``dividend_schedule``).
        signals: Señales de cada foto por modelo (salida de ``detect_model_rv``).
        spots: Spot (implícito) por ``(subyacente, fecha)``; también se usa
            para liquidar al vencimiento y para la cobertura.
        dividends: Dividendos realizados (``ticker``, ``ex_dividend_date``, ``cash_amount``).
        rules: Reglas de producción.

    Returns:
        ``diario`` (valor de la cuenta por modelo al mid), ``estructuras``
        (cada orden y su resultado) y ``eventos`` (cortes y coberturas).
    """
    days = sorted(chains)
    models = ("crr", "bsm")
    ex = pd.to_datetime(dividends["ex_dividend_date"]).dt.date
    structures: list[Structure] = []
    pending: list[Structure] = []
    books: dict[tuple[str, str], _Book] = {}
    cash = {m: 0.0 for m in models}
    hedge_cost = {m: 0.0 for m in models}
    events: list[dict] = []
    daily: list[dict] = []
    prev_value: float | None = None
    priority = 0
    sid = 0
    prev_day: date | None = None
    for day in days:
        chain = chains[day]
        quotes = _quotes(chain)
        # 1. Carry del efectivo y dividendos de las acciones de cobertura desde la foto anterior.
        if prev_day is not None:
            gap = (day - prev_day).days / 365.0
            rate = float(chain["risk_free_rate"].iloc[0]) if "risk_free_rate" in chain else 0.0
            for m in models:
                cash[m] *= math.exp(rate * gap)
            for (m, und), book in books.items():
                paid = dividends.loc[(dividends["ticker"] == und) & (ex > prev_day) & (ex <= day), "cash_amount"].sum()
                cash[m] += book.shares * float(paid)
        # 2. Órdenes de la foto anterior: se llenan si esta foto cruza el límite.
        open_now = [s for s in structures if s.opened is not None and s.closed is None]
        for s in pending:
            per_model = sum(1 for o in open_now if o.model == s.model)
            if per_model >= rules.cap_per_model or len(open_now) >= rules.max_open:
                s.reason = "sin_cupo"
                continue
            cost = _ask_net(s.legs, quotes)
            if cost is None or cost > s.limit + 1e-9:
                s.reason = "no_llenada"
                continue
            s.opened = day
            s.entry_cash = -MULT * s.limit - s.commission
            cash[s.model] += s.entry_cash
            open_now.append(s)
        pending = []
        # 3. Marcas y cierres por estructura.
        for s in open_now:
            marks = _value(s.legs, quotes)
            if marks is not None:
                s.last_mid, s.last_liq = marks
            reason = None
            settle = None
            if day >= s.expiration:
                spot = spots.get((s.underlying, s.expiration), spots.get((s.underlying, day)))
                settle = _intrinsic(s.legs, spot)
                reason = "vencimiento"
            elif (s.expiration - day).days <= rules.pre_expiry_days:
                reason = "pre_vencimiento"
            elif s.entry_cash + MULT * s.last_mid < -rules.kill_switch_usd:
                reason = "kill_switch"
            elif s.entry_cash + MULT * s.last_liq >= rules.take_profit_fraction * s.edge:
                reason = "convergencia"
            if reason:
                value = settle if settle is not None else s.last_liq
                commission = 0.0 if settle is not None else rules.option_commission * sum(abs(l["qty"]) for l in s.legs)
                s.exit_cash = MULT * value - commission
                cash[s.model] += s.exit_cash
                s.closed, s.reason = day, reason
        open_now = [s for s in open_now if s.closed is None]
        # 4. Valor de la cuenta al mid y pérdida máxima entre fotos.
        value_by_model = _account_value(models, cash, open_now, books, spots, quotes, day)
        total = sum(value_by_model.values())
        halted = prev_value is not None and total - prev_value < -rules.daily_loss_usd
        if halted:
            events.append({"fecha": day, "evento": "perdida_diaria", "valor": total - prev_value})
            for s in open_now:
                s.exit_cash = MULT * s.last_liq - rules.option_commission * sum(abs(l["qty"]) for l in s.legs)
                cash[s.model] += s.exit_cash
                s.closed, s.reason = day, "perdida_diaria"
            open_now = []
        # 5. Cobertura de cada libro con su modelo.
        if rules.hedge != "sin_cobertura":
            _hedge(day, chain, quotes, open_now, books, cash, hedge_cost, spots, rules, flatten=halted, events=events)
        value_by_model = _account_value(models, cash, open_now, books, spots, quotes, day)
        prev_value = sum(value_by_model.values())
        daily.append({"fecha": day, **{f"valor_{m}": v for m, v in value_by_model.items()},
                      "valor": prev_value, "abiertas": len(open_now),
                      **{f"costo_cobertura_{m}": hedge_cost[m] for m in models}})
        # 6. Señales nuevas: se publican y se llenan (o no) en la foto siguiente.
        if not halted:
            taken = {s.key for s in open_now}
            order = list(models[priority:] + models[:priority])
            priority = 1 - priority
            counts = {m: sum(1 for s in open_now if s.model == m) for m in models}
            for m in order:
                frame = signals.get(day, {}).get(m)
                if frame is None or frame.empty:
                    continue
                for row in frame.itertuples(index=False):
                    if counts[m] >= rules.cap_per_model or len(open_now) + len(pending) >= rules.max_open:
                        break
                    legs = [{"symbol": l["symbol"], "qty": float(l["qty"]), "strike": float(l["strike"]),
                             "option_type": l["option_type"]} for l in row.leg_spec]
                    s = Structure(sid, m, row.detector, row.underlying, pd.Timestamp(row.expiration).date(), legs,
                                  float(row.limit_net), float(row.net_edge_usd), float(row.commission))
                    if s.key in taken:
                        continue
                    sid += 1
                    taken.add(s.key)
                    pending.append(s)
                    structures.append(s)
                    counts[m] += 1
        prev_day = day
    return {
        "diario": pd.DataFrame(daily),
        "estructuras": pd.DataFrame([{
            "id": s.sid, "modelo": s.model, "detector": s.detector, "subyacente": s.underlying,
            "vencimiento": s.expiration, "limite": s.limit, "edge": s.edge, "abierta": s.opened,
            "cerrada": s.closed, "motivo": s.reason, "efectivo_entrada": s.entry_cash, "efectivo_salida": s.exit_cash,
            "pnl": (s.entry_cash + s.exit_cash) if s.closed else np.nan,
            # P&L si se hubiera salido al mid (sin cruzar el spread de salida ni su comisión).
            "pnl_al_mid": (s.entry_cash + MULT * s.last_mid) if s.closed and s.reason != "vencimiento" else
                          ((s.entry_cash + s.exit_cash) if s.closed else np.nan),
        } for s in structures]),
        "eventos": pd.DataFrame(events),
    }


def _account_value(models, cash, open_now, books, spots, quotes, day) -> dict[str, float]:
    value = dict(cash)
    for s in open_now:
        value[s.model] += MULT * s.last_mid
    for (m, und), book in books.items():
        spot = spots.get((und, day))
        if spot is not None:
            value[m] += book.shares * spot
        if book.opt_contracts:
            q = quotes.get(book.opt_symbol)
            if q is not None:
                book.opt_last_mid = 0.5 * (q[0] + q[1])
            elif book.opt_expiration is not None and day >= book.opt_expiration and spot is not None:
                book.opt_last_mid = max(spot - book.opt_strike, 0.0)
            value[m] += MULT * book.opt_contracts * book.opt_last_mid
    return value


def _hedge(day, chain, quotes, open_now, books, cash, hedge_cost, spots, rules, flatten, events) -> None:
    strategies = [{"detector": s.detector, "underlying": s.underlying, "contracts_open": 1,
                   "legs": [{"kind": "option", "symbol": l["symbol"], "units": l["qty"]} for l in s.legs]}
                  for s in open_now]
    greeks, _ = portfolio_greeks(strategies, chain) if strategies and not flatten else (pd.DataFrame(), [])
    cache = _SmileCache(chain, 100)
    keys = set(books) | (set(zip(greeks["modelo"], greeks["underlying"])) if not greeks.empty else set())
    for m, und in sorted(keys):
        book = books.setdefault((m, und), _Book())
        row = (greeks.loc[(greeks["modelo"] == m) & (greeks["underlying"] == und)].iloc[0]
               if not greeks.empty and ((greeks["modelo"] == m) & (greeks["underlying"] == und)).any() else None)
        if row is not None and row["faltantes"]:
            continue  # como en vivo: con patas sin precio, el libro no se toca
        spot = spots.get((und, day))
        if spot is None:
            continue
        delta = float(row["delta"]) if row is not None else 0.0
        gamma = float(row["gamma"]) if row is not None else 0.0
        hedge_delta = 0.0
        if rules.hedge == "delta_gamma":
            exclude = set(row["vencimientos"]) if row is not None else set()
            # La opción vencida o lejos del dinero se vende (o se liquida al intrínseco).
            if book.opt_contracts and book.opt_expiration is not None and day >= book.opt_expiration:
                cash[m] += MULT * book.opt_contracts * max(spot - book.opt_strike, 0.0)
                book.opt_contracts, book.opt_symbol = 0.0, None
            pick = None if flatten or row is None else pick_gamma_option(chain, und, exclude, book.opt_symbol)
            if book.opt_contracts and (pick is None or pick["symbol"] != book.opt_symbol):
                q = quotes.get(book.opt_symbol)
                price = q[0] if q is not None else book.opt_last_mid
                cash[m] += MULT * book.opt_contracts * price - rules.option_commission * book.opt_contracts
                hedge_cost[m] += book.opt_contracts * (MULT * ((q[1] - q[0]) / 2 if q else 0.0) + rules.option_commission)
                book.opt_contracts, book.opt_symbol = 0.0, None
            if pick is not None:
                g_h = cache.greeks(m, pick.to_dict())
                if g_h is not None and g_h.gamma > 1e-12:
                    target = int(min(max(round(-gamma / (MULT * g_h.gamma)), 0), rules.max_option_contracts))
                    trade = target - book.opt_contracts
                    if trade:
                        bid, ask = float(pick["bid"]), float(pick["ask"])
                        price = ask if trade > 0 else bid
                        cash[m] -= MULT * trade * price + rules.option_commission * abs(trade)
                        hedge_cost[m] += abs(trade) * (MULT * (ask - bid) / 2 + rules.option_commission)
                        book.opt_contracts = float(target)
                        book.opt_symbol, book.opt_strike = str(pick["symbol"]), float(pick["strike"])
                        book.opt_expiration = pd.Timestamp(pick["expiration"]).date()
                        book.opt_last_mid = 0.5 * (bid + ask)
                    if book.opt_contracts:
                        hedge_delta = MULT * book.opt_contracts * g_h.delta
        target = 0.0 if flatten else float(-round(delta + hedge_delta))
        trade = target - book.shares
        if trade and (abs(trade) >= rules.band_shares or target == 0):
            cost = abs(trade) * spot * rules.stock_cost_bps * 1e-4
            cash[m] -= trade * spot + cost
            hedge_cost[m] += cost
            book.shares = target


def trade_stats(trades: pd.DataFrame) -> pd.DataFrame:
    """Resumen por modelo de las estructuras cerradas."""
    closed = trades.loc[trades["cerrada"].notna()]
    rows = []
    for m, g in closed.groupby("modelo"):
        orders = trades.loc[trades["modelo"] == m]
        rows.append({
            "modelo": m, "ordenes": len(orders), "llenadas": int(orders["abierta"].notna().sum()),
            "cerradas": len(g), "pnl_total": g["pnl"].sum(), "pnl_medio": g["pnl"].mean(),
            "error_estandar": g["pnl"].std(ddof=1) / math.sqrt(len(g)) if len(g) > 1 else np.nan,
            "acierto": (g["pnl"] > 0).mean(),
            **{f"cierres_{k}": int(v) for k, v in g["motivo"].value_counts().items()},
        })
    return pd.DataFrame(rows)
