r"""Precio e IV con dividendos discretos (spot escrowed), para toda la cadena.

Es la convención de dividendos que respaldó el paso 3 de la Parte 2: con el
mismo ajuste, el árbol que sabe en qué fecha cae cada dividendo erra menos que
el árbol con yield continuo equivalente (Diebold-Mariano 6,1 sobre 181 ruedas),
y la diferencia se concentra en los calls dentro del dinero antes de una fecha
ex. Ver :mod:`ita_options.greeks` para el modelo.

Una cadena lleva sus dividendos en la columna :data:`SCHEDULE_COLUMN`: por fila,
una tupla de ``(años a la fecha ex, monto)`` con los dividendos proyectados
point-in-time dentro de la vida del contrato (tupla vacía si no hay). Si la
columna no está, los pricers caen al yield continuo ``dividend_yield``: es lo
que usan las cadenas sintéticas de los tests, generadas con yield continuo.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .greeks import bs_greeks, crr_price, pv_dividends
from .volatility import crr_american_price

__all__ = [
    "SCHEDULE_COLUMN",
    "Dividends",
    "row_dividends",
    "equivalent_yield",
    "model_price",
    "implied_vol",
]

SCHEDULE_COLUMN = "dividend_schedule"
Dividends = Sequence[tuple[float, float]]
Model = Literal["crr", "bsm", "crr_q"]


def row_dividends(row: pd.Series | object) -> tuple[tuple[float, float], ...] | None:
    """Dividendos de una fila de la cadena, o ``None`` si la cadena no los trae."""
    value = row.get(SCHEDULE_COLUMN) if isinstance(row, pd.Series) else getattr(row, SCHEDULE_COLUMN, None)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return tuple((float(t), float(d)) for t, d in value)


def equivalent_yield(spot: float, rate: float, tau: float, dividends: Dividends) -> float:
    r""":math:`q` continuo con :math:`S e^{-q\tau} = S - D`."""
    D = pv_dividends(dividends, rate, tau)
    return -math.log(max(1.0 - D / spot, 0.5)) / max(tau, 1e-6)


def model_price(model: Model, spot: float, strike: float, tau: float, rate: float, sigma: float,
                flag: str, dividends: Dividends, steps: int = 100) -> float:
    """Precio de ``crr`` (americano escrowed), ``bsm`` (europeo escrowed) o ``crr_q`` (americano, yield continuo)."""
    if model == "crr":
        return crr_price(spot, strike, tau, rate, sigma, flag, dividends, steps=steps)
    if model == "crr_q":
        q = equivalent_yield(spot, rate, tau, dividends)
        return crr_american_price(spot, strike, tau, rate, q, sigma, flag, steps=steps)
    return bs_greeks(spot, strike, tau, rate, sigma, flag, dividends).price


def implied_vol(model: Model, price: float, spot: float, strike: float, tau: float, rate: float,
                flag: str, dividends: Dividends, steps: int = 100, lo: float = 0.03, hi: float = 3.0) -> float:
    r"""IV con el precio del modelo; ``nan`` si el precio está fuera de lo alcanzable.

    Para el árbol arranca de la IV de Black-Scholes y hace Newton con la vega de
    BS (la vega del árbol y la de BS son casi iguales): converge en dos o tres
    árboles. Si Newton se escapa, cae a Brent.
    """
    def f(sig: float) -> float:
        return model_price(model, spot, strike, tau, rate, sig, flag, dividends, steps) - price

    try:
        f_lo, f_hi = f(lo), f(hi)
    except ValueError:
        return float("nan")
    if not (f_lo < 0 < f_hi):
        return float("nan")
    if model == "bsm":
        return float(brentq(f, lo, hi, xtol=1e-6))
    start = implied_vol("bsm", price, spot, strike, tau, rate, flag, dividends)
    sig = start if np.isfinite(start) else 0.3
    if model == "crr_q":
        # sigma de S*  ->  sigma de S (Hull): punto de partida para el árbol con yield continuo.
        sig *= 1.0 - pv_dividends(dividends, rate, tau) / spot
    for _ in range(6):
        err = f(sig)
        if abs(err) < 1e-4:
            return float(sig)
        vega = bs_greeks(spot, strike, tau, rate, sig, flag, dividends).vega
        if vega < 1e-6:
            break
        nxt = sig - err / vega
        if not lo < nxt < hi:
            break
        sig = nxt
    return float(brentq(f, lo, hi, xtol=1e-6))
