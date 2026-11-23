r"""Black-Scholes y binomial con dividendos discretos, y sus griegas.

Parte 2 del trabajo: Black-Scholes compite contra el binomial y ambos alimentan
las estrategias de cobertura. Este módulo pone a los dos modelos en igualdad de
condiciones: misma convención de dividendos, mismas unidades de las griegas.

Dividendos discretos: modelo de spot escrowed
---------------------------------------------
Con dividendos conocidos :math:`d_i` en fechas ex :math:`t_i`, el precio se
separa en el valor presente de los dividendos y una parte riesgosa:

.. math::

    S_t = S^*_t + D(t), \qquad
    D(t) = \sum_{t < t_i \le T} d_i\, e^{-r (t_i - t)},

y la volatilidad :math:`\sigma` se aplica a :math:`S^*` (Hull, cap. 21).

- **Black-Scholes:** :math:`V = BS(S - D(0), K, r, \sigma, \tau)`. Es idéntico a
  usar un yield continuo :math:`q` tal que :math:`S e^{-q\tau} = S - D`, que es
  la convención del resto del pipeline: los dos dan el mismo precio.
- **Binomial:** el árbol se arma sobre :math:`S^*`, pero en cada nodo el valor
  de ejercicio usa el precio real :math:`S^*_{n,j} + D(t_n)`. Así el árbol sabe
  cuándo cae cada dividendo, y el ejercicio anticipado de un call en la víspera
  de la fecha ex queda bien valuado. Con yield continuo el dividendo se reparte
  en todo el plazo y esa decisión se pierde.

Griegas
-------
Unidades (las mismas que en el material de clase): :math:`\Delta` por USD de
spot, :math:`\Gamma` por USD², :math:`\mathcal{V}` por 1,00 de volatilidad
(dividir por 100 para un punto), :math:`\Theta` por año (dividir por 365 para un
día) y :math:`\rho` por 1,00 de tasa.

- **Black-Scholes:** fórmulas cerradas sobre :math:`S^*`. Como
  :math:`\partial S^*/\partial S = 1`, :math:`\Delta` y :math:`\Gamma` respecto de
  :math:`S` coinciden con las de :math:`S^*`. La :math:`\rho` y la :math:`\Theta`
  llevan un término más, porque :math:`D` depende de :math:`r` y de :math:`t`:
  :math:`\rho = \rho^* + \Delta \sum_i d_i t_i e^{-r t_i}` y
  :math:`\Theta = \Theta^* - \Delta\, r D`.
- **Binomial:** :math:`\Delta`, :math:`\Gamma` y :math:`\Theta` salen de los
  nodos del propio árbol (pasos 1 y 2), sin revaluar;
  :math:`\mathcal{V}` y :math:`\rho` por diferencias centrales.

Nota sobre el material de clase: en la slide 9 del tutorial 7 y en la tabla de
griegas del notebook la put tiene un error de signo. La correcta es
:math:`P = K e^{-r\tau} N(-d_2) - S\,N(-d_1)`, que es la que usa el código.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.stats import norm

__all__ = [
    "Greeks",
    "pv_dividends",
    "bs_greeks",
    "crr_greeks",
    "model_greeks",
]

OptionFlag = Literal["c", "p"]
Dividends = Sequence[tuple[float, float]]  # (años hasta la fecha ex, monto por acción)


@dataclass(frozen=True, slots=True)
class Greeks:
    """Precio y griegas de una opción (por acción)."""

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def scaled(self, quantity: float) -> "Greeks":
        """Griegas de ``quantity`` unidades (con signo); útil para agregar carteras."""
        return Greeks(*(quantity * v for v in (self.price, self.delta, self.gamma,
                                                self.vega, self.theta, self.rho)))


def pv_dividends(dividends: Dividends, rate: float, tau: float, t: float = 0.0) -> float:
    r"""Valor presente en :math:`t` de los dividendos con fecha ex en :math:`(t, \tau]`."""
    return float(sum(d * math.exp(-rate * (ti - t)) for ti, d in dividends if t < ti <= tau))


# --------------------------------------------------------------------------- #
# Black-Scholes con spot escrowed
# --------------------------------------------------------------------------- #


def bs_greeks(
    spot: float, strike: float, tau: float, rate: float, sigma: float, flag: OptionFlag,
    dividends: Dividends = (), pv_div: float | None = None,
) -> Greeks:
    r"""Precio y griegas europeas de Black-Scholes con dividendos discretos.

    Con :math:`S^* = S - D`:

    .. math::

        d_1 = \frac{\ln(S^*/K) + (r + \tfrac12\sigma^2)\tau}{\sigma\sqrt\tau},
        \qquad d_2 = d_1 - \sigma\sqrt\tau

    .. math::

        C = S^* N(d_1) - K e^{-r\tau} N(d_2), \qquad
        P = K e^{-r\tau} N(-d_2) - S^* N(-d_1)

    Args:
        spot: Precio de la acción.
        strike: Strike.
        tau: Años al vencimiento.
        rate: Tasa libre de riesgo continua.
        sigma: Volatilidad de :math:`S^*`.
        flag: ``"c"`` o ``"p"``.
        dividends: Dividendos ``(años a la fecha ex, monto)``.
        pv_div: Valor presente ya calculado (si se pasa, ignora ``dividends``).
    """
    D = pv_div if pv_div is not None else pv_dividends(dividends, rate, tau)
    s = spot - D
    if tau <= 0 or sigma <= 0 or s <= 0:
        intrinsic = max(spot - strike, 0.0) if flag == "c" else max(strike - spot, 0.0)
        delta = (1.0 if spot > strike else 0.0) if flag == "c" else (-1.0 if spot < strike else 0.0)
        return Greeks(intrinsic, delta, 0.0, 0.0, 0.0, 0.0)
    sqrt_t = math.sqrt(tau)
    d1 = (math.log(s / strike) + (rate + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc = math.exp(-rate * tau)
    pdf = norm.pdf(d1)
    gamma = pdf / (s * sigma * sqrt_t)
    vega = s * pdf * sqrt_t
    common_theta = -s * pdf * sigma / (2 * sqrt_t)
    if flag == "c":
        price = s * norm.cdf(d1) - strike * disc * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = common_theta - rate * strike * disc * norm.cdf(d2)
        rho = strike * tau * disc * norm.cdf(d2)
    else:
        price = strike * disc * norm.cdf(-d2) - s * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
        theta = common_theta + rate * strike * disc * norm.cdf(-d2)
        rho = -strike * tau * disc * norm.cdf(-d2)
    # Con dividendos discretos D depende de r y de t; a S fijo, S* = S - D también:
    #   dV/dr = rho* - Delta dD/dr,  con dD/dr = -sum d_i t_i e^{-r t_i}
    #   dV/dt = theta* - Delta dD/dt, con dD/dt = r D (D crece hacia la fecha ex)
    # Sin estos términos la rho y la theta analíticas no coinciden con el árbol.
    if pv_div is None and dividends:
        dD_dr = -sum(di * ti * math.exp(-rate * ti) for ti, di in dividends if 0 < ti <= tau)
        rho -= delta * dD_dr
        theta -= delta * rate * D
    return Greeks(price, delta, gamma, vega, theta, rho)


# --------------------------------------------------------------------------- #
# Binomial CRR con spot escrowed
# --------------------------------------------------------------------------- #


def _crr_tree(
    spot: float, strike: float, tau: float, rate: float, sigma: float, flag: OptionFlag,
    dividends: Dividends, steps: int, american: bool,
) -> tuple[float, tuple[float, float], tuple[float, float, float],
           tuple[float, float], tuple[float, float, float], float]:
    """Árbol CRR sobre :math:`S^*`; devuelve valores y spots de los pasos 0, 1 y 2."""
    if steps < 3:
        raise ValueError("El árbol necesita al menos 3 pasos para sacar las griegas de sus nodos.")
    if spot - pv_dividends(dividends, rate, tau) <= 0:
        raise ValueError("Los dividendos superan al spot: S* = S - D no es positivo.")
    dt = tau / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp(rate * dt)
    p = (growth - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("Parametrización CRR inválida: subir steps o revisar sigma.")
    disc = 1.0 / growth
    s_star0 = spot - pv_dividends(dividends, rate, tau)

    def remaining_pv(n: int) -> float:
        return pv_dividends(dividends, rate, tau, t=n * dt)

    def payoff(s: np.ndarray) -> np.ndarray:
        return np.maximum(s - strike, 0.0) if flag == "c" else np.maximum(strike - s, 0.0)

    j = np.arange(steps + 1)
    s_star_T = s_star0 * u ** (steps - j) * d ** j   # índice 0 = todo arriba
    values = payoff(s_star_T)
    snapshots: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for n in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        jj = np.arange(n + 1)
        spots = s_star0 * u ** (n - jj) * d ** jj + remaining_pv(n)
        if american:
            values = np.maximum(values, payoff(spots))
        if n <= 2:
            snapshots[n] = (values.copy(), spots)
    v1, s1 = snapshots[1]
    v2, s2 = snapshots[2]
    return (float(snapshots[0][0][0]), (float(v1[0]), float(v1[1])), (float(v2[0]), float(v2[1]), float(v2[2])),
            (float(s1[0]), float(s1[1])), (float(s2[0]), float(s2[1]), float(s2[2])), dt)


def crr_greeks(
    spot: float, strike: float, tau: float, rate: float, sigma: float, flag: OptionFlag,
    dividends: Dividends = (), steps: int = 200, american: bool = True,
    vol_bump: float = 0.01, rate_bump: float = 1e-4,
) -> Greeks:
    r"""Precio y griegas del binomial CRR con dividendos discretos.

    De los nodos del árbol (sin revaluar):

    .. math::

        \Delta = \frac{V_{1,u} - V_{1,d}}{S_{1,u} - S_{1,d}}, \qquad
        \Gamma = \frac{\dfrac{V_{2,uu}-V_{2,ud}}{S_{2,uu}-S_{2,ud}}
                     - \dfrac{V_{2,ud}-V_{2,dd}}{S_{2,ud}-S_{2,dd}}}
                    {\tfrac12 (S_{2,uu} - S_{2,dd})}, \qquad
        \Theta = \frac{V_{2,ud} - V_0 - \Delta\,(S_{2,ud} - S_0)}{2\Delta t}

    y :math:`\mathcal{V}`, :math:`\rho` por diferencias centrales revaluando el
    árbol con :math:`\sigma \pm 0{,}01` y :math:`r \pm 0{,}0001`, promediando
    los árboles de :math:`N` y :math:`N+1` pasos para cancelar la oscilación
    par-impar del precio.

    Args:
        american: ``False`` da la versión europea (para validar contra BS).
    """
    if tau <= 0:
        return bs_greeks(spot, strike, tau, rate, sigma, flag, dividends)
    v0, (vu, vd), (vuu, vud, vdd), (su, sd), (suu, sud, sdd), dt = _crr_tree(
        spot, strike, tau, rate, sigma, flag, dividends, steps, american)
    delta = (vu - vd) / (su - sd)
    gamma = ((vuu - vud) / (suu - sud) - (vud - vdd) / (sud - sdd)) / (0.5 * (suu - sdd))
    # El nodo central del paso 2 está en S*_0 + D(2dt), no en S: con dividendos D
    # crece hacia la fecha ex y el nodo queda corrido. Se descuenta ese
    # desplazamiento con Delta para medir dV/dt a S fijo.
    theta = (vud - v0 - delta * (sud - spot)) / (2 * dt)

    def price(sig: float, r: float) -> float:
        # Promedio de N y N+1 pasos: el precio del árbol oscila con la paridad de
        # N cuando el bump mueve los nodos respecto del strike, y esa oscilación,
        # dividida por un bump chico, contamina vega y rho. Promediar la cancela.
        return 0.5 * (_crr_tree(spot, strike, tau, r, sig, flag, dividends, steps, american)[0]
                      + _crr_tree(spot, strike, tau, r, sig, flag, dividends, steps + 1, american)[0])

    vega = (price(sigma + vol_bump, rate) - price(max(sigma - vol_bump, 1e-4), rate)) / (2 * vol_bump)
    rho = (price(sigma, rate + rate_bump) - price(sigma, rate - rate_bump)) / (2 * rate_bump)
    return Greeks(v0, delta, gamma, vega, theta, rho)


def model_greeks(
    model: Literal["crr", "bsm"], spot: float, strike: float, tau: float, rate: float,
    sigma: float, flag: OptionFlag, dividends: Dividends = (), steps: int = 200,
) -> Greeks:
    """Griegas con el modelo pedido: ``crr`` (americana) o ``bsm`` (europea)."""
    if model == "crr":
        return crr_greeks(spot, strike, tau, rate, sigma, flag, dividends, steps=steps)
    return bs_greeks(spot, strike, tau, rate, sigma, flag, dividends)
