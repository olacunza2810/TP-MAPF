r"""Coberturas con griegas, con el binomial y con Black-Scholes (Parte 2, paso 4).

La consigna pide estrategias basadas en griegas, aplicables a los dos modelos,
para cubrir posiciones y mejorar la exposición del portafolio. Este módulo
define tres coberturas y la contabilidad para medirlas sobre datos reales.

Coberturas
----------
Para una posición de :math:`q` contratos de una opción :math:`O` (multiplicador
100), con griegas del modelo :math:`\Delta_O, \Gamma_O, \mathcal V_O`:

- **Delta** (acción): :math:`h = -100\,q\,\Delta_O` acciones.
- **Delta-gamma** (opción ATM corta + acción): primero se anula la gamma con
  :math:`n = -q\,\Gamma_O/\Gamma_H` contratos de una opción ATM :math:`H`, y
  después la delta que queda con acciones:
  :math:`h = -100\,(q\,\Delta_O + n\,\Delta_H)`.
- **Delta-vega** (opción ATM larga + acción): igual, con
  :math:`n = -q\,\mathcal V_O/\mathcal V_H`.
- **Delta-gamma entera**: la versión que se puede ejecutar en la cuenta,
  :math:`n = \max(\mathrm{round}(-q\,\Gamma_O/\Gamma_H), 0)`: contratos
  enteros y sólo compras (con nivel 3 de opciones no se puede vender una opción
  descubierta). Es la que corre en paper con ``--hedge delta_gamma``.

La opción para la gamma es la ATM del vencimiento más corto (más gamma por
dólar); la de la vega, la ATM del más largo (más vega por dólar de gamma). En
los dos casos se excluye el vencimiento de la propia posición: cubrirse con la
misma opción es trivial.

P&L de un día
-------------
Entre las fotos :math:`t` y :math:`t+1`, con mids observados y el spot :math:`S`:

.. math::

    \Pi_{t+1} - \Pi_t = 100\,q\,(m^O_{t+1} - m^O_t) + 100\,n\,(m^H_{t+1} - m^H_t)
    + h\,(S_{t+1} - S_t + d_{t+1}) + C_t\, r_t\, \delta t ,

con :math:`d_{t+1}` el dividendo si la fecha ex cae en :math:`(t, t+1]` (el que
tiene la acción lo cobra) y :math:`C_t` el efectivo de la cartera (el premium
cobrado menos lo invertido en la cobertura), que rinde la tasa del día.

Costos de rebalanceo: medio spread observado y USD 0,65 por contrato en las
opciones; en la acción, un costo proporcional (``stock_cost_bps``) por la
cantidad negociada. Cuando la opción de cobertura cambia de símbolo se paga la
salida de la vieja y la entrada de la nueva.

Efectividad
-----------
.. math::

    E = 1 - \frac{\operatorname{Var}(\Delta\Pi^{\text{cubierta}})}
                 {\operatorname{Var}(\Delta\Pi^{\text{sin cobertura}})}

sobre los mismos días-posición. :math:`E=1` es una cobertura perfecta.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .greeks import Greeks, model_greeks

__all__ = [
    "STRATEGIES",
    "HedgeCosts",
    "hedge_quantities",
    "pick_instrument",
    "PositionPath",
    "simulate_position",
    "effectiveness",
    "effectiveness_table",
]

Strategy = Literal["sin_cobertura", "delta", "delta_gamma", "delta_gamma_entera", "delta_vega"]
STRATEGIES: tuple[Strategy, ...] = ("sin_cobertura", "delta", "delta_gamma", "delta_gamma_entera", "delta_vega")
#: Opción de cobertura que usa cada estrategia (las dos de gamma comparten la ATM corta).
INSTRUMENT = {"delta_gamma": "delta_gamma", "delta_gamma_entera": "delta_gamma", "delta_vega": "delta_vega"}
MULTIPLIER = 100.0


@dataclass(frozen=True, slots=True)
class HedgeCosts:
    """Costos de rebalanceo.

    Attributes:
        option_commission: USD por contrato de opción.
        stock_cost_bps: Costo por acción negociada, en puntos básicos del precio
            (medio spread más impacto). Las acciones de RTX, BA y LMT cotizan
            con spread de 1 centavo, menos de 0,1 pb; 1 pb es conservador.
    """

    option_commission: float = 0.65
    stock_cost_bps: float = 1.0


def hedge_quantities(strategy: Strategy, q: float, g_opt: Greeks, g_hedge: Greeks | None) -> tuple[float, float]:
    r"""Contratos de la opción de cobertura y acciones para neutralizar la(s) griega(s).

    Args:
        strategy: Cobertura.
        q: Contratos de la posición (con signo; -1 es vendido).
        g_opt: Griegas por acción de la opción de la posición.
        g_hedge: Griegas por acción de la opción de cobertura (``None`` en delta).

    Returns:
        ``(n, h)``: contratos de la opción de cobertura y acciones.
    """
    if strategy == "sin_cobertura":
        return 0.0, 0.0
    n = 0.0
    if strategy in INSTRUMENT:
        if g_hedge is None:
            raise ValueError(f"{strategy} necesita una opción de cobertura.")
        target, hedge = (g_opt.vega, g_hedge.vega) if strategy == "delta_vega" else (g_opt.gamma, g_hedge.gamma)
        if abs(hedge) < 1e-12:
            raise ValueError("La opción de cobertura no tiene exposición a la griega a cubrir.")
        n = -q * target / hedge
        if strategy == "delta_gamma_entera":
            n = float(max(round(n), 0))
    h = -MULTIPLIER * (q * g_opt.delta + (n * g_hedge.delta if g_hedge is not None else 0.0))
    return n, h


def pick_instrument(candidates: pd.DataFrame, exclude_expiration: object, kind: Literal["gamma", "vega"],
                    min_dte: int = 14) -> pd.Series | None:
    """Elige la opción de cobertura entre las ATM de cada vencimiento de un día y subyacente.

    Args:
        candidates: Una fila por vencimiento (la call ATM), con ``expiration`` y ``dte``.
        exclude_expiration: Vencimiento de la posición (no se usa como cobertura).
        kind: ``gamma`` toma el vencimiento más corto; ``vega``, el más largo.
        min_dte: Plazo mínimo de la opción de cobertura.
    """
    pool = candidates.loc[(candidates["expiration"] != exclude_expiration) & (candidates["dte"] >= min_dte)]
    if pool.empty:
        return None
    pool = pool.sort_values("dte")
    return pool.iloc[0] if kind == "gamma" else pool.iloc[-1]


@dataclass
class PositionPath:
    """Una posición seguida día a día.

    Attributes:
        days: Fechas de las fotos donde la posición está marcada (consecutivas).
        option_mid, option_half: Mid y medio spread de la opción en cada foto.
        spot: Spot en cada foto.
        rate: Tasa de cada foto.
        dividend: Dividendo cobrado entre la foto anterior y esta (0 en la primera).
        greeks: Griegas de la opción por modelo y foto: ``greeks[modelo][i]``.
        hedges: Por cobertura con opción (``delta_gamma``, ``delta_vega``),
            una entrada por foto salvo la última: un dict con ``symbol``,
            ``mid``, ``half`` y ``greeks`` (por modelo) de la opción elegida
            en esa foto, y ``mid_next``, ``half_next`` de esa misma opción en
            la foto siguiente, para marcarla y para desarmarla.
    """

    days: list
    option_mid: list[float]
    option_half: list[float]
    spot: list[float]
    rate: list[float]
    dividend: list[float]
    greeks: dict[str, list[Greeks]]
    hedges: dict[str, list[dict]]


def simulate_position(path: PositionPath, model: str, strategy: Strategy, q: float = -1.0,
                      costs: HedgeCosts = HedgeCosts()) -> pd.DataFrame:
    """P&L diario de una posición cubierta con las griegas de ``model``.

    Returns:
        Una fila por día de tenencia con ``pnl`` (bruto, a mids), ``costo``
        (rebalanceo de la cobertura, incluido el desarme en el último día),
        ``h`` (acciones) y ``n`` (contratos de la opción de cobertura).
    """
    rows = []
    h_prev, n_prev, sym_prev, half_prev_today = 0.0, 0.0, None, 0.0
    steps = len(path.days) - 1
    for i in range(steps):
        g_opt = path.greeks[model][i]
        hedge = path.hedges[INSTRUMENT[strategy]][i] if strategy in INSTRUMENT else None
        g_h = hedge["greeks"][model] if hedge is not None else None
        n, h = hedge_quantities(strategy, q, g_opt, g_h)
        S0, S1 = path.spot[i], path.spot[i + 1]
        # Costo de llevar la cobertura de (h_prev, n_prev) a (h, n) en la foto i.
        cost = abs(h - h_prev) * S0 * costs.stock_cost_bps * 1e-4
        if hedge is not None:
            per_contract = MULTIPLIER * hedge["half"] + costs.option_commission
            if sym_prev == hedge["symbol"] or (n == 0 and n_prev == 0):
                cost += abs(n - n_prev) * per_contract
            else:
                cost += abs(n) * per_contract
                if sym_prev is not None and abs(n_prev) > 0:
                    # La opción vieja se vende en esta foto, a su medio spread de hoy.
                    cost += abs(n_prev) * (MULTIPLIER * half_prev_today + costs.option_commission)
        pnl_opt = MULTIPLIER * q * (path.option_mid[i + 1] - path.option_mid[i])
        pnl_hedge_opt = MULTIPLIER * n * (hedge["mid_next"] - hedge["mid"]) if hedge is not None else 0.0
        pnl_stock = h * (S1 - S0 + path.dividend[i + 1])
        cash = -(MULTIPLIER * q * path.option_mid[i] + (MULTIPLIER * n * hedge["mid"] if hedge is not None else 0.0)
                 + h * S0)
        dt = (pd.Timestamp(path.days[i + 1]) - pd.Timestamp(path.days[i])).days / 365.0
        carry = cash * path.rate[i] * dt
        if i == steps - 1:
            # Desarme de la cobertura al final de la tenencia.
            cost += abs(h) * S1 * costs.stock_cost_bps * 1e-4
            if hedge is not None:
                cost += abs(n) * (MULTIPLIER * hedge["half_next"] + costs.option_commission)
        rows.append({"dia": path.days[i + 1], "paso": i, "pnl": pnl_opt + pnl_hedge_opt + pnl_stock + carry,
                     "costo": cost, "h": h, "n": n})
        h_prev, n_prev = h, n
        sym_prev = hedge["symbol"] if hedge is not None else None
        half_prev_today = hedge["half_next"] if hedge is not None else 0.0
    return pd.DataFrame(rows)


def effectiveness(hedged: Iterable[float], unhedged: Iterable[float]) -> float:
    """:math:`1 - \\operatorname{Var}(cubierta)/\\operatorname{Var}(sin cobertura)`."""
    a, b = np.asarray(list(hedged), float), np.asarray(list(unhedged), float)
    var_b = np.var(b, ddof=1)
    return float(1.0 - np.var(a, ddof=1) / var_b) if var_b > 0 else float("nan")


def effectiveness_table(days: pd.DataFrame, by: Sequence[str] = ()) -> pd.DataFrame:
    """Efectividad bruta y neta, costo y P&L por modelo y cobertura.

    Args:
        days: Salida concatenada de :func:`simulate_position`, con columnas
            ``modelo``, ``cobertura``, ``posicion``, ``paso`` y las de ``by``.
    """
    keys = ["posicion", "paso"]
    base = days.loc[days["cobertura"] == "sin_cobertura"].set_index(["modelo", *keys])["pnl"]
    rows = []
    group_cols = ["modelo", "cobertura", *by]
    for key, part in days.loc[days["cobertura"] != "sin_cobertura"].groupby(group_cols, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        ref = base.reindex(pd.MultiIndex.from_arrays([part["modelo"], part["posicion"], part["paso"]])).to_numpy()
        net = part["pnl"] - part["costo"]
        rows.append({
            **dict(zip(group_cols, key)),
            "dias_posicion": len(part),
            "efectividad": effectiveness(part["pnl"], ref),
            "efectividad_neta": effectiveness(net, ref),
            "desvio_sin_cobertura": float(np.std(ref, ddof=1)),
            "desvio_cubierta": float(part["pnl"].std(ddof=1)),
            "pnl_medio": float(part["pnl"].mean()),
            "costo_medio": float(part["costo"].mean()),
            "pnl_neto_medio": float(net.mean()),
            "acciones_medias": float(part["h"].abs().mean()),
            "contratos_cobertura_medios": float(part["n"].abs().mean()),
        })
    return pd.DataFrame(rows)
