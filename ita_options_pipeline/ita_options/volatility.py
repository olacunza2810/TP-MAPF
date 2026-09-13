"""Valuación y extracción de volatilidad implícita.

Contiene dos motores:

1. **Black-Scholes-Merton con dividend yield continuo**, para la IV europea.
   Es rápido y vectorizable, pero está mal especificado para opciones sobre
   acciones individuales estadounidenses, que son de ejercicio americano.

2. **Árbol binomial Cox-Ross-Rubinstein con ejercicio anticipado**, que es el
   motor correcto para RTX y BA. Es el que hay que usar para la Parte 1 del
   trabajo práctico, y el que define la superficie contra la cual se detectan
   inconsistencias.

La diferencia entre ambas IVs no es ruido: es la prima de ejercicio anticipado.
Reportar la IV europea sobre una put americana in-the-money y llamar
"arbitraje" a la brecha resultante es el error clásico en este ejercicio.
"""

from __future__ import annotations

import math
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from scipy.stats import norm

__all__ = [
    "bsm_price",
    "bsm_vega",
    "crr_american_price",
    "no_arbitrage_bounds",
    "implied_volatility",
    "implied_volatility_vectorized",
]

OptionFlag = Literal["c", "p"]

_MIN_SIGMA: Final[float] = 1e-3
_MAX_SIGMA: Final[float] = 5.0
_PRICE_TOL: Final[float] = 1e-8


def _crr_sigma_floor(tau: float, rate: float, dividend_yield: float, steps: int) -> float:
    r"""Volatilidad mínima admisible por el árbol CRR.

    La probabilidad riesgo-neutral :math:`p = (e^{(r-q)\Delta t} - d)/(u - d)`
    permanece en :math:`(0,1)` sólo si el movimiento del árbol domina al drift:

    .. math::

        \sigma\sqrt{\Delta t} > |r - q|\,\Delta t
        \;\Longleftrightarrow\;
        \sigma > |r - q|\sqrt{\Delta t}

    Si se abre la búsqueda de raíz por debajo de esta cota, el árbol admite
    arbitraje interno y el solver falla devolviendo ``NaN`` sobre contratos
    perfectamente válidos. Se aplica un margen de seguridad de 2x.
    """
    if tau <= 0.0 or steps <= 0:
        return _MIN_SIGMA
    dt = tau / steps
    return max(_MIN_SIGMA, 2.0 * abs(rate - dividend_yield) * math.sqrt(dt))


def bsm_price(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    dividend_yield: float,
    sigma: float,
    flag: OptionFlag,
) -> float:
    r"""Precio Black-Scholes-Merton con dividend yield continuo.

    Para un call:

    .. math::

        C = S e^{-q\tau} \Phi(d_1) - K e^{-r\tau} \Phi(d_2)

    .. math::

        d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^2)\tau}
                   {\sigma\sqrt{\tau}},
        \qquad d_2 = d_1 - \sigma\sqrt{\tau}

    y por paridad put-call, :math:`P = C - S e^{-q\tau} + K e^{-r\tau}`.

    Args:
        spot: Precio del subyacente :math:`S`.
        strike: Precio de ejercicio :math:`K`.
        tau: Tiempo al vencimiento en años :math:`\tau`.
        rate: Tasa libre de riesgo continua :math:`r`.
        dividend_yield: Dividend yield continuo :math:`q`.
        sigma: Volatilidad anualizada :math:`\sigma`.
        flag: ``'c'`` para call, ``'p'`` para put.

    Returns:
        Prima teórica del contrato.
    """
    if tau <= 0.0 or sigma <= 0.0:
        intrinsic = spot - strike if flag == "c" else strike - spot
        return max(intrinsic, 0.0)

    vol_sqrt_t = sigma * math.sqrt(tau)
    d1 = (
        math.log(spot / strike) + (rate - dividend_yield + 0.5 * sigma**2) * tau
    ) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    disc_s = spot * math.exp(-dividend_yield * tau)
    disc_k = strike * math.exp(-rate * tau)

    if flag == "c":
        return disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2)
    return disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1)


def bsm_vega(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    dividend_yield: float,
    sigma: float,
) -> float:
    r"""Vega BSM, :math:`\partial V/\partial\sigma`.

    .. math::

        \mathcal{V} = S e^{-q\tau}\phi(d_1)\sqrt{\tau}

    Se usa como diagnóstico de condicionamiento: cuando :math:`\mathcal{V}` es
    muy chica (deep ITM/OTM o vencimiento inmediato), la inversión de la IV está
    mal condicionada y su valor no debe alimentar la calibración.
    """
    if tau <= 0.0 or sigma <= 0.0:
        return 0.0
    vol_sqrt_t = sigma * math.sqrt(tau)
    d1 = (
        math.log(spot / strike) + (rate - dividend_yield + 0.5 * sigma**2) * tau
    ) / vol_sqrt_t
    return spot * math.exp(-dividend_yield * tau) * norm.pdf(d1) * math.sqrt(tau)


def crr_american_price(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    dividend_yield: float,
    sigma: float,
    flag: OptionFlag,
    steps: int = 256,
) -> float:
    r"""Precio de una opción americana por árbol binomial Cox-Ross-Rubinstein.

    Con :math:`\Delta t = \tau/N`:

    .. math::

        u = e^{\sigma\sqrt{\Delta t}},\quad d = 1/u,\quad
        p = \frac{e^{(r-q)\Delta t} - d}{u - d}

    y la recursión con ejercicio anticipado, para :math:`n = N-1,\dots,0`:

    .. math::

        V_i^{n} = \max\Big(
            \Psi(S_i^{n}),\;
            e^{-r\Delta t}\big[p\,V_{i+1}^{n+1} + (1-p)\,V_i^{n+1}\big]
        \Big)

    donde :math:`\Psi` es el payoff de ejercicio.

    Args:
        spot: Precio del subyacente.
        strike: Strike.
        tau: Años al vencimiento.
        rate: Tasa libre de riesgo continua.
        dividend_yield: Dividend yield continuo.
        sigma: Volatilidad anualizada.
        flag: ``'c'`` o ``'p'``.
        steps: Número de pasos :math:`N` del árbol.

    Returns:
        Prima del contrato americano.

    Raises:
        ValueError: si la parametrización viola :math:`0 < p < 1`, lo que indica
            que ``steps`` es insuficiente para la combinación de ``sigma`` y
            ``tau`` y el árbol admitiría arbitraje interno.
    """
    if tau <= 0.0:
        intrinsic = spot - strike if flag == "c" else strike - spot
        return max(intrinsic, 0.0)

    dt = tau / steps
    up = math.exp(sigma * math.sqrt(dt))
    down = 1.0 / up
    disc = math.exp(-rate * dt)
    prob = (math.exp((rate - dividend_yield) * dt) - down) / (up - down)
    if not 0.0 < prob < 1.0:
        raise ValueError(
            f"Probabilidad riesgo-neutral fuera de (0,1): p={prob:.4f}. "
            "Aumentar 'steps' o revisar sigma/tau."
        )

    indices = np.arange(steps + 1, dtype=np.float64)
    prices = spot * up ** (2.0 * indices - steps)
    sign = 1.0 if flag == "c" else -1.0
    values = np.maximum(sign * (prices - strike), 0.0)

    for step in range(steps - 1, -1, -1):
        values = disc * (prob * values[1:] + (1.0 - prob) * values[:-1])
        prices = prices[1:] * down
        values = np.maximum(values, sign * (prices - strike))

    return float(values[0])


def no_arbitrage_bounds(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    dividend_yield: float,
    flag: OptionFlag,
    american: bool = True,
) -> tuple[float, float]:
    r"""Cotas de no arbitraje del precio de la opción.

    Para un call americano:
    :math:`\max(Se^{-q\tau} - Ke^{-r\tau},\, S - K,\, 0) \le C \le S`.

    Para una put americana:
    :math:`\max(Ke^{-r\tau} - Se^{-q\tau},\, K - S,\, 0) \le P \le K`.

    Un mid observado fuera de estas cotas es una de dos cosas: un error de datos
    (quote stale, cruzado o de una sesión sin actividad) o un arbitraje estático
    genuino. Nunca es una IV. Por eso el solver devuelve ``NaN`` y el registro
    queda marcado, en lugar de forzar una raíz inexistente.

    Returns:
        Tupla ``(cota_inferior, cota_superior)``.
    """
    disc_s = spot * math.exp(-dividend_yield * tau)
    disc_k = strike * math.exp(-rate * tau)
    if flag == "c":
        lower = max(disc_s - disc_k, 0.0)
        if american:
            lower = max(lower, spot - strike, 0.0)
        return lower, spot
    lower = max(disc_k - disc_s, 0.0)
    if american:
        lower = max(lower, strike - spot, 0.0)
    return lower, strike


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    dividend_yield: float,
    flag: OptionFlag,
    american: bool = True,
    steps: int = 256,
) -> float:
    r"""Invierte la volatilidad implícita por Brent sobre :math:`f(\sigma)=0`.

    Se resuelve

    .. math::

        f(\sigma) = V_{\text{modelo}}(\sigma) - V_{\text{mercado}} = 0

    sobre :math:`\sigma \in [10^{-6},\, 5]`. Brent es preferible a
    Newton-Raphson acá porque :math:`f` es monótona creciente en
    :math:`\sigma` pero su derivada (vega) se anula en las colas, lo que hace
    que Newton diverja justamente en los contratos deep OTM que más interesan
    para detectar dislocaciones.

    Args:
        price: Prima observada de mercado, típicamente el mid.
        spot: Precio del subyacente en el mismo timestamp.
        strike: Strike.
        tau: Años al vencimiento.
        rate: Tasa libre de riesgo continua.
        dividend_yield: Dividend yield continuo.
        flag: ``'c'`` o ``'p'``.
        american: Si ``True`` usa el árbol CRR; si no, BSM cerrado.
        steps: Pasos del árbol cuando ``american`` es ``True``.

    Returns:
        La volatilidad implícita, o ``float('nan')`` si el precio viola las
        cotas de no arbitraje o la raíz no está acotada en el intervalo.
    """
    if not math.isfinite(price) or price <= 0.0 or tau <= 0.0 or spot <= 0.0:
        return float("nan")

    lower, upper = no_arbitrage_bounds(
        spot, strike, tau, rate, dividend_yield, flag, american
    )
    if price < lower - _PRICE_TOL or price > upper + _PRICE_TOL:
        return float("nan")

    def objective(sigma: float) -> float:
        theoretical = (
            crr_american_price(
                spot, strike, tau, rate, dividend_yield, sigma, flag, steps
            )
            if american
            else bsm_price(spot, strike, tau, rate, dividend_yield, sigma, flag)
        )
        return theoretical - price

    sigma_lo = (
        _crr_sigma_floor(tau, rate, dividend_yield, steps) if american else _MIN_SIGMA
    )
    try:
        if objective(sigma_lo) > 0.0 or objective(_MAX_SIGMA) < 0.0:
            return float("nan")
        return float(brentq(objective, sigma_lo, _MAX_SIGMA, xtol=1e-8, maxiter=100))
    except (ValueError, RuntimeError):
        return float("nan")


def implied_volatility_vectorized(
    prices: npt.NDArray[np.float64],
    spots: npt.NDArray[np.float64],
    strikes: npt.NDArray[np.float64],
    taus: npt.NDArray[np.float64],
    rates: npt.NDArray[np.float64],
    dividend_yields: npt.NDArray[np.float64],
    flags: npt.NDArray[np.str_],
    american: bool = True,
    steps: int = 256,
) -> npt.NDArray[np.float64]:
    """Aplica :func:`implied_volatility` fila por fila sobre arrays alineados.

    El árbol binomial no admite vectorización trivial en ``sigma`` porque cada
    contrato requiere su propia búsqueda de raíz. Para datasets grandes conviene
    paralelizar por proceso; acá se prioriza legibilidad y correctitud, que es
    lo que se evalúa.

    Args:
        prices: Primas de mercado.
        spots: Spots alineados al mismo timestamp que cada prima.
        strikes: Strikes.
        taus: Años al vencimiento.
        rates: Tasas libres de riesgo por fila.
        dividend_yields: Dividend yields por fila.
        flags: Array de ``'c'``/``'p'``.
        american: Motor de valuación.
        steps: Pasos del árbol.

    Returns:
        Array de IVs, con ``NaN`` donde la inversión falla.

    Raises:
        ValueError: si los arrays no tienen la misma longitud.
    """
    arrays = (prices, spots, strikes, taus, rates, dividend_yields, flags)
    if len({len(a) for a in arrays}) != 1:
        raise ValueError("Todos los arrays de entrada deben tener igual longitud.")

    out = np.empty(len(prices), dtype=np.float64)
    for i in range(len(prices)):
        out[i] = implied_volatility(
            price=float(prices[i]),
            spot=float(spots[i]),
            strike=float(strikes[i]),
            tau=float(taus[i]),
            rate=float(rates[i]),
            dividend_yield=float(dividend_yields[i]),
            flag="c" if str(flags[i]).lower().startswith("c") else "p",
            american=american,
            steps=steps,
        )
    return out
