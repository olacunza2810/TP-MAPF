r"""Backtest de la estrategia de valor relativo con fotos diarias de bid/ask.

Es el backtest de la estrategia que corre en paper (:mod:`execution.model_rv`),
con las mismas fórmulas y sobre la foto diaria de DoltHub.

Línea de tiempo (sin lookahead)
-------------------------------
- **Rueda t.** Con la foto de :math:`t` se ajusta la sonrisa del modelo por
  vencimiento, se valúa cada estructura candidata y se publican órdenes límite
  :math:`L = V - m` (compra) o :math:`L = V + m` (venta) dentro del spread.
- **Rueda t+1.** La orden se llena **sólo si** la foto de :math:`t+1` cruza el
  límite: para comprar, el ask neto de la estructura tiene que ser
  :math:`\le L`; para vender, el bid neto :math:`\ge L`. Se llena a
  :math:`L`. Las órdenes no llenadas se cancelan: duran una rueda.
- **Ruedas siguientes.** La posición se marca al lado de cierre (vender al
  bid lo comprado, recomprar al ask lo vendido). Se cierra por **convergencia**
  cuando la ganancia *después de cruzar el spread de salida* alcanza la
  fracción pedida del edge, por **kill-switch** cuando la pérdida medida
  **contra el mid** supera la fracción pedida de la pérdida máxima, o por
  **vencimiento**, liquidando por valor intrínseco contra el cierre de la
  acción.

El modelo de llenado es conservador: con una sola foto por rueda sólo se ve un
precio por día, y una orden real podría llenarse con cualquier cruce intradía.

Precomputación
--------------
Las sonrisas y los valores de modelo de cada estructura no dependen de los
hiperparámetros, así que se calculan una vez por modelo
(:func:`precompute_valuations`) y la grilla de hiperparámetros sólo simula la
cartera (:func:`simulate`).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from .escrowed import row_dividends
from .execution.model_rv import RVSettings, Smile, _price as _rv_price, fit_smile

_LOG = logging.getLogger(__name__)

Model = Literal["crr", "bsm"]
COMMISSION = 0.65
MULTIPLIER = 100.0


def _effective_yield(chain: pd.DataFrame) -> pd.Series:
    r"""Yield continuo equivalente a los dividendos discretos proyectados.

    El árbol CRR y BSM del pipeline usan yield continuo. Se elige :math:`q` tal
    que :math:`S e^{-q\tau} = S - D`, con :math:`D` el valor presente de los
    dividendos proyectados point-in-time: el forward del modelo coincide con el
    forward con dividendos discretos.
    """
    spot = chain["underlying_price"]
    pv = chain.get("pv_dividends", pd.Series(0.0, index=chain.index)).fillna(0.0)
    ratio = (1.0 - pv / spot).clip(lower=0.5)
    return (-np.log(ratio) / chain["tau"].clip(lower=1e-6)).fillna(0.0)


def _price(model: Model, s: float, k: float, t: float, r: float, q: float, sig: float, f: str, steps: int,
           dividends: tuple[tuple[float, float], ...] | None = None) -> float:
    """El mismo pricer que la estrategia en vivo (:func:`execution.model_rv._price`)."""
    return _rv_price(model, s, k, t, r, q, sig, f, steps, dividends)


def candidate_structures(slice_: pd.DataFrame, forward: float, band: float) -> list[tuple[str, tuple[tuple[str, float], ...]]]:
    """Verticales contiguos y butterflies simétricas cerca del dinero.

    Returns:
        Lista de ``(tipo, ((símbolo, cantidad), ...))`` en dirección de compra.
    """
    out: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    near = slice_.loc[np.abs(np.log(slice_["strike"] / forward)) <= band]
    for flag in ("c", "p"):
        side = near.loc[near["option_type"] == flag].sort_values("strike")
        syms, ks = side["symbol"].tolist(), side["strike"].tolist()
        for i in range(len(syms) - 1):
            out.append(("vertical", ((syms[i], 1.0), (syms[i + 1], -1.0))))
        for i in range(1, len(syms) - 1):
            if np.isclose(ks[i] - ks[i - 1], ks[i + 1] - ks[i]):
                out.append(("butterfly", ((syms[i - 1], 1.0), (syms[i], -2.0), (syms[i + 1], 1.0))))
    return out


def precompute_valuations(
    chain: pd.DataFrame, model: Model, settings: RVSettings | None = None,
    min_dte: int = 14, max_dte: int = 90, band: float = 0.12,
    dividends: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sonrisas y valores de modelo por rueda, independientes de los hiperparámetros.

    Args:
        chain: Cadena curada de DoltHub (todas las ruedas).
        model: ``crr`` o ``bsm``.
        settings: Parámetros de ajuste de la sonrisa (pasos del árbol, puntos mínimos).
        min_dte, max_dte: Ventana amplia de vencimientos; la grilla la recorta.
        band: :math:`|\\ln(K/F)|` máximo de las patas.
        dividends: Tabla de dividendos (``merge_dividends``). Si se pasa, cada
            contrato se valúa con sus dividendos discretos proyectados
            point-in-time (spot escrowed), igual que la estrategia en vivo; si
            no, con el yield continuo equivalente.

    Returns:
        ``(estructuras, sonrisas)``. Cada fila de ``estructuras`` es una
        estructura en una rueda, en dirección de compra, con ``model_value``,
        ``bid_net``, ``ask_net``, ``dte``, ``legs`` y ``max_loss``.
    """
    settings = settings or RVSettings(model=model, steps=100)
    base = chain.loc[(chain["bid"] > 0) & (chain["ask"] > chain["bid"])].copy()
    base["q_eff"] = _effective_yield(base)
    base["dividend_yield"] = base["q_eff"]
    if dividends is not None:
        from .dolthub import attach_dividend_schedule

        base = attach_dividend_schedule(base, dividends)
    structures: list[dict[str, object]] = []
    smiles: list[dict[str, object]] = []
    for (day, und, exp), slice_ in base.groupby(["trade_date", "underlying", "expiration"], sort=True):
        dte = int(slice_["dte"].iloc[0])
        if not min_dte <= dte <= max_dte:
            continue
        smile: Smile | None = fit_smile(slice_, model, settings)
        if smile is None:
            continue
        smiles.append({"trade_date": day, "underlying": und, "expiration": exp, "dte": dte, **asdict(smile)})
        row0 = slice_.iloc[0]
        s, t, r, q = float(row0["underlying_price"]), float(row0["tau"]), float(row0["risk_free_rate"]), float(row0["q_eff"])
        divs = row_dividends(row0)
        quotes = slice_.set_index("symbol")
        values: dict[str, float] = {}
        for kind, legs in candidate_structures(slice_, smile.forward, band):
            v = 0.0
            for sym, qty in legs:
                if sym not in values:
                    rq = quotes.loc[sym]
                    values[sym] = _price(model, s, float(rq["strike"]), t, r, q, smile.sigma(float(rq["strike"])),
                                         str(rq["option_type"]), settings.steps, divs)
                v += qty * values[sym]
            bid_net = sum(qty * float(quotes.loc[sym, "bid" if qty > 0 else "ask"]) for sym, qty in legs)
            ask_net = sum(qty * float(quotes.loc[sym, "ask" if qty > 0 else "bid"]) for sym, qty in legs)
            strikes = [float(quotes.loc[sym, "strike"]) for sym, _ in legs]
            flag = str(quotes.loc[legs[0][0], "option_type"])
            if kind == "vertical":
                width = strikes[1] - strikes[0]
                payoff_bounds = (0.0, width) if flag == "c" else (-width, 0.0)
            else:
                payoff_bounds = (0.0, strikes[1] - strikes[0])
            structures.append({
                "trade_date": day, "underlying": und, "expiration": pd.Timestamp(exp).date(),
                "dte": dte, "kind": kind,
                "option_type": flag, "strikes": "/".join(f"{k:g}" for k in strikes),
                "legs": legs, "model_value": v, "bid_net": bid_net, "ask_net": ask_net,
                "payoff_min": payoff_bounds[0], "payoff_max": payoff_bounds[1],
                "moneyness": float(np.log(np.mean(strikes) / smile.forward)),
            })
    _LOG.info("Valuaciones %s: %d estructuras-rueda, %d sonrisas", model, len(structures), len(smiles))
    return pd.DataFrame(structures), pd.DataFrame(smiles)


@dataclass(frozen=True, slots=True)
class RVParams:
    """Hiperparámetros de la cartera, calibrables in-sample.

    Attributes:
        margin_usd: Edge contra el modelo exigido por estructura, neto de comisión.
        take_profit: Fracción del edge anunciado que dispara el cierre; ``None``
            mantiene hasta el vencimiento.
        stop_fraction: Pérdida no realizada, como fracción de la pérdida máxima
            de la estructura, que dispara el kill-switch; ``None`` lo desactiva.
        min_dte, max_dte: Ventana de vencimientos al publicar.
        max_new_per_day: Órdenes nuevas por rueda.
        max_open: Posiciones abiertas simultáneas.
    """

    margin_usd: float = 10.0
    take_profit: float | None = 0.6
    stop_fraction: float | None = 0.5
    min_dte: int = 21
    max_dte: int = 75
    max_new_per_day: int = 2
    max_open: int = 10


def _structure_quotes(quotes: dict[tuple[object, str], tuple[float, float]], day: object, legs) -> tuple[float, float] | None:
    """(bid_net, ask_net) de una estructura en una rueda, o None si falta una pata."""
    bid_net = ask_net = 0.0
    for sym, qty in legs:
        quote = quotes.get((day, sym))
        if quote is None:
            return None
        bid, ask = quote
        bid_net += qty * (bid if qty > 0 else ask)
        ask_net += qty * (ask if qty > 0 else bid)
    return bid_net, ask_net


def simulate(
    valuations: pd.DataFrame, chain: pd.DataFrame, spots: pd.DataFrame,
    params: RVParams, start: object | None = None, end: object | None = None,
    data_end: object | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Simula la cartera de órdenes límite sobre las valuaciones precomputadas.

    Args:
        valuations: Salida de :func:`precompute_valuations`.
        chain: Cadena curada (para bid/ask de llenado y de marcado).
        spots: Cierres de la acción con ``symbol``, ``date``, ``close``.
        params: Hiperparámetros.
        start, end: Ventana en la que se **publican** órdenes.
        data_end: Última rueda que se usa para marcar y cerrar. Con
            ``data_end = end`` las posiciones abiertas se cierran a mercado al
            final de la ventana (corte limpio para la muestra in-sample).

    Returns:
        ``(operaciones, pnl_diario, embudo)``.
    """
    days = sorted(chain["trade_date"].unique())
    start = start or days[0]
    end = end or days[-1]
    data_end = data_end or days[-1]
    days = [d for d in days if start <= d <= data_end]
    quotes = {
        (d, s): (b, a) for d, s, b, a in chain[["trade_date", "symbol", "bid", "ask"]].itertuples(index=False)
    }
    spot = {(r.symbol, r.date): r.close for r in spots.itertuples(index=False)}
    vals = valuations.loc[valuations["dte"].between(params.min_dte, params.max_dte)].copy()
    vals["expiration"] = pd.to_datetime(vals["expiration"]).dt.date
    by_day = {d: g for d, g in vals.groupby("trade_date")}
    margin = params.margin_usd / MULTIPLIER

    funnel = {"candidatas": 0, "ordenes": 0, "llenadas": 0, "sin_cotizacion_t1": 0}
    pending: list[dict[str, object]] = []
    open_pos: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    daily: list[dict[str, object]] = []

    def close(pos: dict[str, object], day: object, value: float, commission: float, reason: str) -> None:
        pos.update(closed=day, exit_value=value, exit_commission=commission, reason=reason,
                   pnl=(value - pos["fill"]) * MULTIPLIER * pos["sign"] - pos["entry_commission"] - commission)
        trades.append(pos)

    for idx, day in enumerate(days):
        realized_before = sum(t["pnl"] for t in trades)
        # 1. Llenados de las órdenes publicadas ayer contra la foto de hoy.
        for order in pending:
            sq = _structure_quotes(quotes, day, order["legs"])
            if sq is None:
                funnel["sin_cotizacion_t1"] += 1
                continue
            bid_net, ask_net = sq
            buy = order["sign"] > 0
            crossed = ask_net <= order["limit"] if buy else bid_net >= order["limit"]
            if crossed and len(open_pos) < params.max_open:
                funnel["llenadas"] += 1
                n_contracts = sum(abs(q) for _, q in order["legs"])
                open_pos.append({**order, "opened": day, "fill": order["limit"],
                                 "entry_commission": COMMISSION * n_contracts})
        pending = []

        # 2. Gestión de posiciones: vencimiento, kill-switch, convergencia.
        still: list[dict[str, object]] = []
        for pos in open_pos:
            n_contracts = sum(abs(q) for _, q in pos["legs"])
            if day >= pos["expiration"]:
                s = spot.get((pos["underlying"], day))
                if s is not None and np.isfinite(s):
                    value = sum(qty * max((s - k) if pos["option_type"] == "c" else (k - s), 0.0)
                                for (_, qty), k in zip(pos["legs"], pos["strike_list"]))
                    close(pos, day, value, 0.0, "vencimiento")
                    continue
            sq = _structure_quotes(quotes, day, pos["legs"])
            if sq is not None:
                exit_value = sq[0] if pos["sign"] > 0 else sq[1]  # vender lo comprado al bid neto
                unreal = (exit_value - pos["fill"]) * MULTIPLIER * pos["sign"] - pos["entry_commission"] - COMMISSION * n_contracts
                pos["last_mark"] = unreal
                # El kill-switch mide contra el mid: medirlo contra el valor de
                # liquidación lo dispararía el mismo día del fill por el solo costo
                # de cruzar un spread del 20%, cristalizando ese costo como pérdida.
                mid_mark = ((sq[0] + sq[1]) / 2 - pos["fill"]) * MULTIPLIER * pos["sign"] - pos["entry_commission"]
                if params.stop_fraction is not None and mid_mark < -params.stop_fraction * pos["max_loss_usd"]:
                    close(pos, day, exit_value, COMMISSION * n_contracts, "kill_switch")
                    continue
                if params.take_profit is not None and unreal >= params.take_profit * pos["edge_usd"]:
                    close(pos, day, exit_value, COMMISSION * n_contracts, "convergencia")
                    continue
                if day >= data_end:
                    close(pos, day, exit_value, COMMISSION * n_contracts, "fin_de_datos")
                    continue
            elif day >= data_end:
                close(pos, day, pos["fill"], 0.0, "fin_de_datos_sin_cotizacion")
                continue
            still.append(pos)
        open_pos = still

        # 3. Órdenes nuevas con la foto de hoy (sólo dentro de la ventana de publicación).
        if start <= day <= end and day < data_end and day in by_day:
            cands = []
            for row in by_day[day].itertuples(index=False):
                funnel["candidatas"] += 1
                for sign in (+1.0, -1.0):
                    value = sign * row.model_value
                    bid_net = row.bid_net if sign > 0 else -row.ask_net
                    ask_net = row.ask_net if sign > 0 else -row.bid_net
                    n_contracts = sum(abs(q) for _, q in row.legs)
                    limit = min(value - margin - COMMISSION * n_contracts / MULTIPLIER, ask_net)
                    if limit <= bid_net + 0.01:
                        continue
                    aggressiveness = (limit - bid_net) / max(ask_net - bid_net, 1e-9)
                    cands.append((aggressiveness, sign, limit, value, row))
            cands.sort(key=lambda c: c[0], reverse=True)
            held = {p["key"] for p in open_pos}
            for aggressiveness, sign, limit, value, row in cands:
                key = (row.underlying, row.expiration, row.kind, row.option_type, row.strikes)
                if key in held or len(pending) >= params.max_new_per_day:
                    continue
                held.add(key)
                n_contracts = sum(abs(q) for _, q in row.legs)
                lo, hi = row.payoff_min, row.payoff_max
                # Pérdida máxima por acción: lo pagado menos el peor payoff. Vender
                # la estructura es comprar su opuesta, cuyo payoff va de -hi a -lo.
                worst_payoff = lo if sign > 0 else -hi
                max_loss = limit - worst_payoff
                pending.append({
                    "key": key, "underlying": row.underlying, "expiration": row.expiration,
                    "kind": row.kind, "option_type": row.option_type, "strikes": row.strikes,
                    "strike_list": [float(k) for k in row.strikes.split("/")],
                    "legs": tuple((s, sign * q) for s, q in row.legs), "sign": 1.0,
                    "limit": limit, "signal_day": day,
                    "model_value": value, "edge_usd": (value - limit) * MULTIPLIER - COMMISSION * n_contracts,
                    "max_loss_usd": max(max_loss, 0.05) * MULTIPLIER, "aggressiveness": aggressiveness,
                    "direction": "compra" if sign > 0 else "venta",
                })
            funnel["ordenes"] += len(pending)

        unreal_total = sum(p.get("last_mark", 0.0) for p in open_pos)
        realized = sum(t["pnl"] for t in trades)
        daily.append({"trade_date": day, "realized_pnl": realized, "unrealized_pnl": unreal_total,
                      "equity": realized + unreal_total, "open_positions": len(open_pos),
                      "pnl_dia_realizado": realized - realized_before})

    frame = pd.DataFrame(trades)
    if not frame.empty:
        frame = frame.drop(columns=["legs", "strike_list"], errors="ignore")
    return frame, pd.DataFrame(daily), funnel


def grid(params_list: Iterable[RVParams]) -> list[RVParams]:
    """Materializa la grilla para poder contarla (N del Sharpe deflactado)."""
    return list(params_list)


def default_grid() -> list[RVParams]:
    """Grilla declarada antes de mirar resultados: 3 × 3 × 2 = 18 configuraciones."""
    out = []
    for margin in (5.0, 10.0, 20.0):
        for tp in (0.5, 1.0, None):
            for stop in (0.5, None):
                out.append(RVParams(margin_usd=margin, take_profit=tp, stop_fraction=stop))
    return out


def summarize(trades: pd.DataFrame, daily: pd.DataFrame) -> dict[str, float]:
    """Métricas de una corrida: operaciones, P&L, acierto, Sharpe y drawdown."""
    if daily.empty:
        return {"trades": 0, "pnl": 0.0, "hit_rate": np.nan, "sharpe": np.nan, "max_dd": 0.0}
    equity = daily["equity"].to_numpy()
    rets = np.diff(np.concatenate([[0.0], equity]))
    sd = rets.std(ddof=1) if len(rets) > 1 else 0.0
    sharpe = float(rets.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan
    dd = float(np.max(np.maximum.accumulate(equity) - equity)) if len(equity) else 0.0
    n = int(len(trades))
    return {
        "trades": n,
        "pnl": float(trades["pnl"].sum()) if n else 0.0,
        "hit_rate": float((trades["pnl"] > 0).mean()) if n else np.nan,
        "sharpe": sharpe,
        "max_dd": dd,
        "sharpe_daily": float(rets.mean() / sd) if sd > 0 else np.nan,
        "skew": float(pd.Series(rets).skew()) if len(rets) > 2 else np.nan,
        "kurt": float(pd.Series(rets).kurt() + 3) if len(rets) > 3 else np.nan,
        "n_days": int(len(rets)),
    }
