r"""Stress-testing por Monte Carlo de la cartera en producción (Parte 2, paso 5).

La consigna pide someter las estrategias a escenarios artificiales y medir su
comportamiento en condiciones adversas. Se estresa la cartera que la estrategia
tendría abierta a capacidad máxima (tres estructuras por modelo), con sus
coberturas y con las reglas de riesgo de producción: kill-switch por estructura,
pérdida diaria máxima de la cuenta y take-profit.

Dinámicas del subyacente
------------------------
Para los tres subyacentes juntos, con shocks correlacionados por Cholesky
(:math:`Z = L\varepsilon`, :math:`LL^\top = \Sigma` la correlación de los
retornos diarios):

- **GBM:** :math:`d\ln S = (r - \tfrac12\sigma^2)dt + \sigma\, dW`.
- **Heston** (Heston, 1993), esquema de truncamiento completo (Lord, Koekkoek y
  van Dijk, 2010):

  .. math::

      v_{k+1} = v_k + \kappa(\theta - v_k^+)\Delta t + \xi\sqrt{v_k^+\Delta t}\, Z^v, \qquad
      \ln S_{k+1} = \ln S_k + (r - \tfrac12 v_k^+)\Delta t + \sqrt{v_k^+ \Delta t}\, Z^S,

  con :math:`\operatorname{corr}(Z^v, Z^S) = \rho`. El notebook de clase
  (``DeltaHedgeSimulatorWithStochasticVol.price_paths``) difunde el precio con
  ``np.sqrt(Var[-1])``, la **última** fila de la matriz, que en el paso ``i``
  todavía vale :math:`v_0`: el precio queda con volatilidad constante y la
  volatilidad estocástica no llega nunca al subyacente. Acá se usa
  :math:`v_k`, la varianza del paso anterior.
- **Merton** (1976): difusión más saltos,
  :math:`d\ln S = (r - \lambda\bar k - \tfrac12\sigma_d^2)dt + \sigma_d dW + J\,dN`,
  :math:`J\sim N(\mu_J, \sigma_J^2)`, :math:`\bar k = e^{\mu_J + \sigma_J^2/2} - 1`.
  En la versión sectorial los tres subyacentes saltan juntos (un mismo proceso
  de Poisson), que es el riesgo de evento propio de defensa (presupuesto,
  contratos, geopolítica).

Los dividendos siguen el mismo modelo escrowed que el pricer: la difusión se
aplica a :math:`S^* = S - D(t)` y :math:`S = S^* + D(t)`, así que en la fecha
ex el precio cae el dividendo; la acción de la cobertura lo cobra.

Revaluación
-----------
El "mercado" valúa cada opción con el árbol binomial americano con dividendo
discreto (escrowed), con la sonrisa ajustada en la foto inicial desplazada en
paralelo por el cambio de la volatilidad ATM del escenario. Al valor del árbol
se le suma el residuo inicial :math:`m_0 - V_0` (lo que la estrategia leyó como
mispricing), que se mantiene constante: el escenario estresa el riesgo de
mercado de la cartera y **no** supone que el mispricing converja, coherente con
el walk-forward de la Parte 1. Para liquidar, las patas compradas se venden al
bid y las vendidas se recompran al ask (medio spread inicial, multiplicado por
el factor de liquidez del escenario), igual que el mark del loop en vivo.

Cada libro se cubre con las griegas de **su** modelo (árbol o Black-Scholes) y
el rebalanceo es diario.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from scipy.stats import norm

__all__ = [
    "vec_crr",
    "vec_bs",
    "ScenarioPaths",
    "simulate_gbm",
    "simulate_heston",
    "simulate_merton",
    "Leg",
    "Structure",
    "HedgeOption",
    "Book",
    "RiskRules",
    "run_account",
    "risk_metrics",
]

Dividends = Sequence[tuple[float, float]]
TRADING_DAYS = 252.0


# --------------------------------------------------------------------------- #
# Pricers vectorizados sobre trayectorias
# --------------------------------------------------------------------------- #


def _pv_grid(dividends: Dividends, rate: float, tau: float, times: np.ndarray) -> np.ndarray:
    pv = np.zeros_like(times)
    for ti, di in dividends:
        if 0 < ti <= tau:
            pv += np.where(times < ti, di * np.exp(-rate * (ti - times)), 0.0)
    return pv


def vec_crr(spot: np.ndarray, strike: float | np.ndarray, tau: float, rate: float, sigma: np.ndarray, flag: str,
            dividends: Dividends = (), steps: int = 100, american: bool = True
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Árbol CRR escrowed evaluado a la vez para un vector de spots y volatilidades.

    Es :func:`ita_options.greeks.crr_greeks` (precio, :math:`\Delta` y
    :math:`\Gamma` de los nodos 1 y 2) con una dimensión más: cada trayectoria
    tiene su spot y su volatilidad.
    """
    spot = np.asarray(spot, float)
    sigma = np.broadcast_to(np.asarray(sigma, float), spot.shape)
    strike = np.broadcast_to(np.asarray(strike, float), spot.shape)
    if tau <= 0:
        sign = 1.0 if flag == "c" else -1.0
        intrinsic = np.maximum(sign * (spot - strike), 0.0)
        delta = np.where(sign * (spot - strike) > 0, sign, 0.0)
        return intrinsic, delta, np.zeros_like(spot)
    K = strike[:, None]
    dt = tau / steps
    pv = _pv_grid(dividends, rate, tau, np.arange(steps + 1) * dt)
    s0 = spot - pv[0]
    if np.any(s0 <= 0):
        raise ValueError("Los dividendos superan al spot: S* = S - D no es positivo.")
    lu = sigma * math.sqrt(dt)
    u = np.exp(lu)
    d = 1.0 / u
    growth = math.exp(rate * dt)
    p = (growth - d) / (u - d)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("Parametrización CRR inválida: subir steps o revisar sigma.")
    a, b = (p / growth)[:, None], ((1 - p) / growth)[:, None]
    sign = 1.0 if flag == "c" else -1.0
    j = np.arange(steps + 1)
    values = np.maximum(sign * (s0[:, None] * np.exp(lu[:, None] * (steps - 2 * j)[None, :]) - K), 0.0)
    snap: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for n in range(steps - 1, -1, -1):
        values = a * values[:, :-1] + b * values[:, 1:]
        if american or n <= 2:
            spots = s0[:, None] * np.exp(lu[:, None] * (n - 2 * j[: n + 1])[None, :]) + pv[n]
            if american:
                np.maximum(values, sign * (spots - K), out=values)
            if n <= 2:
                snap[n] = (values.copy(), spots)
    v1, s1 = snap[1]
    v2, s2 = snap[2]
    delta = (v1[:, 0] - v1[:, 1]) / (s1[:, 0] - s1[:, 1])
    gamma = ((v2[:, 0] - v2[:, 1]) / (s2[:, 0] - s2[:, 1]) - (v2[:, 1] - v2[:, 2]) / (s2[:, 1] - s2[:, 2])) \
        / (0.5 * (s2[:, 0] - s2[:, 2]))
    return snap[0][0][:, 0], delta, gamma


def vec_bs(spot: np.ndarray, strike: float, tau: float, rate: float, sigma: np.ndarray, flag: str,
           dividends: Dividends = ()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Black-Scholes europeo escrowed, vectorizado: precio, delta y gamma."""
    spot = np.asarray(spot, float)
    sigma = np.broadcast_to(np.asarray(sigma, float), spot.shape)
    sign = 1.0 if flag == "c" else -1.0
    if tau <= 0:
        return np.maximum(sign * (spot - strike), 0.0), np.where(sign * (spot - strike) > 0, sign, 0.0), \
            np.zeros_like(spot)
    D = sum(di * math.exp(-rate * ti) for ti, di in dividends if 0 < ti <= tau)
    s = spot - D
    sq = sigma * math.sqrt(tau)
    d1 = (np.log(s / strike) + (rate + 0.5 * sigma ** 2) * tau) / sq
    d2 = d1 - sq
    disc = math.exp(-rate * tau)
    if flag == "c":
        price = s * norm.cdf(d1) - strike * disc * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = strike * disc * norm.cdf(-d2) - s * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
    return price, delta, norm.pdf(d1) / (s * sq)


# --------------------------------------------------------------------------- #
# Escenarios
# --------------------------------------------------------------------------- #


@dataclass
class ScenarioPaths:
    """Trayectorias diarias de un escenario.

    Attributes:
        spot: ``(P, D+1, U)``: spot al cierre de cada día (ya descontado el
            dividendo en su fecha ex).
        atm_vol: ``(P, D+1, U)``: volatilidad ATM implícita de cada día.
        dividend_paid: ``(D+1, U)``: dividendo cobrado ese día por acción.
        spread_factor: Multiplicador de los spreads al liquidar.
    """

    spot: np.ndarray
    atm_vol: np.ndarray
    dividend_paid: np.ndarray
    spread_factor: float = 1.0


def _ex_drops(spot: np.ndarray, ex_days: Sequence[Sequence[tuple[int, float]]]) -> np.ndarray:
    """Matriz ``(D+1, U)`` de dividendos por día."""
    D1, U = spot.shape[1], spot.shape[2]
    paid = np.zeros((D1, U))
    for u, flows in enumerate(ex_days):
        for day, amount in flows:
            if 0 < day < D1:
                paid[day, u] += amount
    return paid


def _pv_remaining(s0: np.ndarray, ex_days: Sequence, rate: float, calendar: np.ndarray) -> np.ndarray:
    """``(D+1, U)``: valor presente en cada día de los dividendos que faltan cobrar en el horizonte."""
    D1, U = len(calendar), len(s0)
    pv = np.zeros((D1, U))
    for u, flows in enumerate(ex_days):
        for day, amount in flows:
            if 0 < day < D1:
                t_ex = calendar[day]
                for t in range(day):
                    pv[t, u] += amount * math.exp(-rate * (t_ex - calendar[t]))
    return pv


def _escrowed_spot(log_star: np.ndarray, pv: np.ndarray) -> np.ndarray:
    r"""Spot a partir de la parte riesgosa: :math:`S_t = S^*_t + D(t)`.

    La difusión se aplica a :math:`S^* = S - D` (el mismo modelo escrowed con
    el que se valúa); en la fecha ex :math:`D(t)` cae el dividendo y el precio
    cae con él.
    """
    return np.exp(log_star) + pv[None, :, :]


def _calendar(days: int, calendar: np.ndarray | None) -> np.ndarray:
    return np.arange(days + 1) / TRADING_DAYS if calendar is None else np.asarray(calendar, float)


def simulate_gbm(s0: np.ndarray, sigma: np.ndarray, corr: np.ndarray, rate: float, days: int, paths: int,
                 rng: np.random.Generator, ex_days: Sequence = (), substeps: int = 4,
                 calendar: np.ndarray | None = None) -> ScenarioPaths:
    """GBM correlado sobre :math:`S^*`. ``sigma`` es la volatilidad anual de cada subyacente."""
    U = len(s0)
    pv = _pv_remaining(s0, ex_days, rate, _calendar(days, calendar))
    s0 = s0 - pv[0]
    L = np.linalg.cholesky(corr)
    dt = 1.0 / (TRADING_DAYS * substeps)
    log_s = np.empty((paths, days + 1, U))
    log_s[:, 0] = np.log(s0)
    x = np.tile(np.log(s0), (paths, 1))
    for day in range(1, days + 1):
        for _ in range(substeps):
            z = rng.standard_normal((paths, U)) @ L.T
            x = x + (rate - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
        log_s[:, day] = x
    paid = _ex_drops(log_s, ex_days)
    vol = np.broadcast_to(sigma, log_s.shape).copy()
    return ScenarioPaths(_escrowed_spot(log_s, pv), vol, paid)


def simulate_heston(s0: np.ndarray, v0: np.ndarray, kappa: np.ndarray, theta: np.ndarray, xi: np.ndarray,
                    rho: np.ndarray, corr: np.ndarray, rate: float, days: int, paths: int,
                    rng: np.random.Generator, ex_days: Sequence = (), substeps: int = 4,
                    calendar: np.ndarray | None = None) -> ScenarioPaths:
    """Heston con truncamiento completo; correlación entre subyacentes por Cholesky.

    La volatilidad ATM implícita del escenario es :math:`\\sqrt{v_t^+}`.
    """
    U = len(s0)
    pv = _pv_remaining(s0, ex_days, rate, _calendar(days, calendar))
    s0 = s0 - pv[0]
    L = np.linalg.cholesky(corr)
    dt = 1.0 / (TRADING_DAYS * substeps)
    log_s = np.empty((paths, days + 1, U))
    vol = np.empty((paths, days + 1, U))
    x = np.tile(np.log(s0), (paths, 1))
    v = np.tile(v0, (paths, 1)).astype(float)
    log_s[:, 0], vol[:, 0] = x, np.sqrt(v)
    for day in range(1, days + 1):
        for _ in range(substeps):
            zs = rng.standard_normal((paths, U)) @ L.T
            zv = rho * zs + np.sqrt(1 - rho ** 2) * rng.standard_normal((paths, U))
            vp = np.maximum(v, 0.0)
            # Precio con la varianza del paso anterior (no la última fila, como el notebook).
            x = x + (rate - 0.5 * vp) * dt + np.sqrt(vp * dt) * zs
            v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * zv
        log_s[:, day] = x
        vol[:, day] = np.sqrt(np.maximum(v, 0.0))
    paid = _ex_drops(log_s, ex_days)
    return ScenarioPaths(_escrowed_spot(log_s, pv), vol, paid)


def simulate_merton(s0: np.ndarray, sigma_diff: np.ndarray, implied: np.ndarray, lam: np.ndarray,
                    mu_j: np.ndarray, sigma_j: np.ndarray, corr: np.ndarray, rate: float, days: int,
                    paths: int, rng: np.random.Generator, ex_days: Sequence = (), common: bool = False,
                    jump_corr: float = 0.7, substeps: int = 4, calendar: np.ndarray | None = None) -> ScenarioPaths:
    """Merton con saltos lognormales; ``common=True`` usa un solo proceso de Poisson sectorial.

    Args:
        sigma_diff: Volatilidad de la parte difusiva.
        implied: Volatilidad implícita con la que el mercado revalúa (incluye los saltos).
        lam: Intensidad de saltos por año (en el sectorial se usa ``lam[0]``).
        jump_corr: Correlación de los tamaños de salto entre subyacentes (sectorial).
    """
    U = len(s0)
    pv = _pv_remaining(s0, ex_days, rate, _calendar(days, calendar))
    s0 = s0 - pv[0]
    L = np.linalg.cholesky(corr)
    Lj = np.linalg.cholesky(np.full((U, U), jump_corr) + (1 - jump_corr) * np.eye(U))
    dt = 1.0 / (TRADING_DAYS * substeps)
    kbar = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1
    log_s = np.empty((paths, days + 1, U))
    x = np.tile(np.log(s0), (paths, 1))
    log_s[:, 0] = x
    for day in range(1, days + 1):
        for _ in range(substeps):
            z = rng.standard_normal((paths, U)) @ L.T
            x = x + (rate - lam * kbar - 0.5 * sigma_diff ** 2) * dt + sigma_diff * math.sqrt(dt) * z
            if common:
                n = rng.poisson(lam[0] * dt, size=(paths, 1))
                size = mu_j + sigma_j * (rng.standard_normal((paths, U)) @ Lj.T)
                x = x + n * size
            else:
                n = rng.poisson(lam * dt, size=(paths, U))
                x = x + n * (mu_j + sigma_j * rng.standard_normal((paths, U)))
        log_s[:, day] = x
    paid = _ex_drops(log_s, ex_days)
    vol = np.broadcast_to(implied, log_s.shape).copy()
    return ScenarioPaths(_escrowed_spot(log_s, pv), vol, paid)


# --------------------------------------------------------------------------- #
# Cartera
# --------------------------------------------------------------------------- #


@dataclass
class Leg:
    """Pata de una estructura (o la opción de cobertura).

    Attributes:
        u: Índice del subyacente.
        strike, flag: Strike y tipo.
        tau0: Años al vencimiento en la foto inicial.
        units: Contratos con signo por estructura.
        sigma: Volatilidad inicial: ``crr`` y ``bsm`` de la sonrisa de cada modelo
            en el strike (con la que cada libro calcula sus griegas) y ``mkt``,
            la IV del árbol que reproduce el mid observado (con la que el
            "mercado" revalúa: el mispricing persiste como diferencia de vol).
        resid: Mid observado menos valor del árbol con la sonrisa (informativo).
        half: Medio spread observado (USD por acción).
        dividends: Dividendos ``(años desde la foto inicial, monto)``.
    """

    u: int
    strike: float
    flag: str
    tau0: float
    units: float
    sigma: dict[str, float]
    resid: float
    half: float
    dividends: tuple[tuple[float, float], ...] = ()


@dataclass
class Structure:
    model: str
    name: str
    legs: list[Leg]
    entry_cash: float  # USD; negativo si se pagó débito
    predicted_edge: float


@dataclass
class HedgeOption:
    """Call ATM corta para la gamma de un subyacente, con la regla de rotación de producción.

    Como en vivo (:func:`execution.hedger.pick_gamma_option`), la opción se
    mantiene mientras siga a menos de ``max_moneyness`` del dinero; si no, se
    vende y se compra la call del mismo vencimiento con el strike de la grilla
    más cercano al spot. Cada trayectoria tiene así su propio strike.

    Attributes:
        leg: La opción inicial (``units`` no se usa).
        smile: Sonrisa de cada modelo del vencimiento (``crr``, ``bsm``): da la
            volatilidad de cualquier strike nuevo.
        strike_step: Paso de la grilla de strikes.
    """

    leg: Leg
    smile: dict[str, Any] = field(default_factory=dict)
    strike_step: float = 5.0
    max_moneyness: float = 0.05

    def sigma(self, model: str, strikes: np.ndarray) -> np.ndarray:
        sm = self.smile[model]
        k = np.log(strikes / sm.forward)
        return np.clip(sm.a + sm.b * k + sm.c * k * k, 0.03, 3.0)


@dataclass
class Book:
    model: str
    structures: list[Structure]
    gamma_options: dict[int, HedgeOption] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskRules:
    """Reglas de producción (las de ``paper_daily.ps1``)."""

    kill_switch_usd: float = 250.0
    daily_loss_usd: float = 1000.0
    take_profit_fraction: float = 0.6
    stock_cost_bps: float = 1.0
    option_commission: float = 0.65
    band_shares: float = 10.0
    max_option_contracts: int = 5
    #: Marca del kill-switch y de la pérdida diaria: ``mid`` (producción tras el
    #: paso 5) o ``liquidacion`` (cruzando el spread, la versión anterior).
    mark: Literal["mid", "liquidacion"] = "mid"


def _leg_paths(leg: Leg, paths: ScenarioPaths, rate: float, calendar: np.ndarray, steps: int) -> dict[str, np.ndarray]:
    """Valor de mercado y griegas de los dos modelos de una pata, para todos los días (``(P, D+1)``)."""
    P, D1 = paths.spot.shape[0], paths.spot.shape[1]
    out = {k: np.empty((P, D1)) for k in ("market", "delta_crr", "gamma_crr", "delta_bsm", "gamma_bsm")}
    sig_mkt0 = leg.sigma.get("mkt", leg.sigma["crr"])
    for day in range(D1):
        t = calendar[day]
        tau = max(leg.tau0 - t, 0.0)
        divs = tuple((ti - t, di) for ti, di in leg.dividends if ti > t)
        shift = paths.atm_vol[:, day, leg.u] - paths.atm_vol[:, 0, leg.u]
        s = paths.spot[:, day, leg.u]
        price, _, _ = vec_crr(s, leg.strike, tau, rate, np.maximum(sig_mkt0 + shift, 0.05), leg.flag, divs, steps)
        out["market"][:, day] = price
        _, out["delta_crr"][:, day], out["gamma_crr"][:, day] = vec_crr(
            s, leg.strike, tau, rate, np.maximum(leg.sigma["crr"] + shift, 0.05), leg.flag, divs, steps)
        _, out["delta_bsm"][:, day], out["gamma_bsm"][:, day] = vec_bs(
            s, leg.strike, tau, rate, np.maximum(leg.sigma["bsm"] + shift, 0.05), leg.flag, divs)
    return out


def run_account(books: list[Book], paths: ScenarioPaths, rate: float, calendar: np.ndarray,
                hedge: Literal["sin_cobertura", "delta", "delta_gamma"], rules: RiskRules = RiskRules(),
                steps: int = 100, cache: dict | None = None) -> dict[str, np.ndarray]:
    """Simula la cuenta entera (todos los libros) con las reglas de riesgo.

    Returns:
        Por trayectoria: ``pnl`` (cuenta, cerrando todo al final cruzando el
        spread), ``pnl_mid`` (marcando lo abierto al mid), ``pnl_<modelo>``, ``kill`` (alguna
        estructura cortada por kill-switch), ``perdida_diaria`` (se cortó la
        cuenta), ``take_profit``, ``costo_cobertura`` y ``peor_dia``.
    """
    cache = {} if cache is None else cache
    P, D1, U = paths.spot.shape
    f = paths.spread_factor

    def legdata(leg: Leg, model: str):
        if id(leg) not in cache:
            cache[id(leg)] = _leg_paths(leg, paths, rate, calendar, steps)
        d = cache[id(leg)]
        return d["market"], d[f"delta_{model}"], d[f"gamma_{model}"]

    out: dict[str, np.ndarray] = {"pnl": np.zeros(P), "kill": np.zeros(P, bool), "perdida_diaria": np.zeros(P, bool),
                                  "take_profit": np.zeros(P, bool), "costo_cobertura": np.zeros(P),
                                  "peor_dia": np.zeros(P)}
    alive = np.ones(P, bool)  # la cuenta sigue operando (no se cortó por pérdida diaria)
    state = []
    for book in books:
        structs = []
        for st in book.structures:
            structs.append({"st": st, "open": np.ones(P, bool), "realized": np.zeros(P)})
        kopt = np.zeros((P, U))
        for u, ho in book.gamma_options.items():
            kopt[:, u] = ho.leg.strike
        state.append({"book": book, "structs": structs, "shares": np.zeros((P, U)), "nopt": np.zeros((P, U)),
                      "kopt": kopt, "cash": np.zeros(P), "pnl": np.zeros(P)})

    def hedge_option(ho: HedgeOption, strikes: np.ndarray, day: int, model: str):
        """Precio de mercado y griegas del modelo de la opción de gamma con ``strikes`` por trayectoria."""
        leg = ho.leg
        t = calendar[day]
        tau = max(leg.tau0 - t, 0.0)
        divs = tuple((ti - t, di) for ti, di in leg.dividends if ti > t)
        shift = paths.atm_vol[:, day, leg.u] - paths.atm_vol[:, 0, leg.u]
        s = paths.spot[:, day, leg.u]
        market, _, _ = vec_crr(s, strikes, tau, rate, np.maximum(ho.sigma("crr", strikes) + shift, 0.05),
                               "c", divs, steps)
        sig_m = np.maximum(ho.sigma(model, strikes) + shift, 0.05)
        if model == "crr":
            _, delta, gamma = vec_crr(s, strikes, tau, rate, sig_m, "c", divs, steps)
        else:
            _, delta, gamma = vec_bs(s, strikes, tau, rate, sig_m, "c", divs)
        return market, delta, gamma

    def liquidation(st: Structure, day: int, at_mid: bool = False) -> np.ndarray:
        value = np.zeros(P)
        for leg in st.legs:
            mkt, _, _ = legdata(leg, "crr")
            side = 0.0 if at_mid else (-1.0 if leg.units > 0 else 1.0)  # largo al bid, corto al ask
            value += leg.units * 100.0 * (mkt[:, day] + side * leg.half * f)
        return value

    risk_at_mid = rules.mark == "mid"

    def hedge_value(s: dict, day: int) -> np.ndarray:
        value = s["cash"] + (s["shares"] * paths.spot[:, day, :]).sum(axis=1)
        for u, ho in s["book"].gamma_options.items():
            if np.any(s["nopt"][:, u] != 0):
                mkt, _, _ = hedge_option(ho, s["kopt"][:, u], day, "crr")
                value += s["nopt"][:, u] * 100.0 * mkt
        return value

    def rebalance(s: dict, day: int, flatten: np.ndarray) -> None:
        """Ajusta acciones y opciones de cobertura del libro al cierre de ``day``."""
        book = s["book"]
        if hedge == "sin_cobertura":
            return
        delta = np.zeros((P, U))
        gamma = np.zeros((P, U))
        for item in s["structs"]:
            mask = item["open"] & ~flatten
            for leg in item["st"].legs:
                _, d, g = legdata(leg, book.model)
                delta[:, leg.u] += np.where(mask, leg.units * 100.0 * d[:, day], 0.0)
                gamma[:, leg.u] += np.where(mask, leg.units * 100.0 * g[:, day], 0.0)
        for u in range(U):
            spot = paths.spot[:, day, u]
            hdelta = np.zeros(P)
            if hedge == "delta_gamma" and u in book.gamma_options:
                ho = book.gamma_options[u]
                per = 100.0 * ho.leg.half * f + rules.option_commission
                # Rotación: la opción que se alejó más de 5% del dinero se vende y se
                # compra la del strike de la grilla más cercano al spot.
                old_k = s["kopt"][:, u]
                rotate = np.abs(np.log(old_k / spot)) > ho.max_moneyness
                if np.any(rotate & (s["nopt"][:, u] != 0)):
                    mkt_old, _, _ = hedge_option(ho, old_k, day, "crr")
                    sell = np.where(rotate, s["nopt"][:, u], 0.0)
                    s["cash"] += sell * 100.0 * (mkt_old - ho.leg.half * f) - np.abs(sell) * rules.option_commission
                    out["costo_cobertura"] += np.abs(sell) * per
                    s["nopt"][:, u] = np.where(rotate, 0.0, s["nopt"][:, u])
                s["kopt"][:, u] = np.where(rotate, np.round(spot / ho.strike_step) * ho.strike_step, old_k)
                mkt, dh, gh = hedge_option(ho, s["kopt"][:, u], day, book.model)
                n_star = -gamma[:, u] / np.maximum(100.0 * gh, 1e-12)
                target_n = np.clip(np.round(n_star), 0, rules.max_option_contracts)
                target_n = np.where(flatten, 0.0, target_n)
                trade = target_n - s["nopt"][:, u]
                price = mkt + np.sign(trade) * ho.leg.half * f
                s["cash"] -= trade * 100.0 * price + np.abs(trade) * rules.option_commission
                out["costo_cobertura"] += np.abs(trade) * per
                s["nopt"][:, u] = target_n
                hdelta = target_n * 100.0 * dh
            target = np.where(flatten, 0.0, -np.round(delta[:, u] + hdelta))
            trade = target - s["shares"][:, u]
            do = (np.abs(trade) >= rules.band_shares) | ((target == 0) & (trade != 0))
            trade = np.where(do, trade, 0.0)
            cost = np.abs(trade) * spot * rules.stock_cost_bps * 1e-4
            s["cash"] -= trade * spot + cost
            out["costo_cobertura"] += cost
            s["shares"][:, u] += trade

    # Día 0: estructuras a su precio de entrada y coberturas armadas.
    for s in state:
        for item in s["structs"]:
            s["cash"] += item["st"].entry_cash  # el efectivo de la entrada queda en la cuenta
        rebalance(s, 0, np.zeros(P, bool))
    prev_total = sum(hedge_value(s, 0) + sum(np.where(i["open"], liquidation(i["st"], 0, risk_at_mid), 0.0)
                                              for i in s["structs"]) for s in state)
    dt_cash = np.diff(calendar)
    for day in range(1, D1):
        flatten_today = np.zeros(P, bool)
        for s in state:
            s["cash"] *= math.exp(rate * dt_cash[day - 1])
            s["cash"] += (s["shares"] * paths.dividend_paid[day][None, :]).sum(axis=1)
        # Valor de la cuenta al cierre, con las estructuras a liquidación.
        values, risk_values = {}, {}
        total = np.zeros(P)
        for s in state:
            hv = hedge_value(s, day)
            sv = np.zeros(P)
            for item in s["structs"]:
                liq = liquidation(item["st"], day)
                values[id(item)] = liq
                risk_values[id(item)] = liquidation(item["st"], day, True) if risk_at_mid else liq
                sv += np.where(item["open"], risk_values[id(item)], 0.0)
            total += hv + sv
        day_pnl = total - prev_total
        out["peor_dia"] = np.minimum(out["peor_dia"], day_pnl)
        # Pérdida diaria de la cuenta: se cierra todo.
        breach = alive & (day_pnl < -rules.daily_loss_usd)
        out["perdida_diaria"] |= breach
        flatten_today |= breach
        for s in state:
            for item in s["structs"]:
                liq = values[id(item)]
                unreal = item["st"].entry_cash + liq
                open_ = item["open"] & alive
                # Kill-switch con la marca de riesgo; take-profit sólo si la ganancia
                # existe después de pagar la salida (como en producción).
                kill = open_ & (item["st"].entry_cash + risk_values[id(item)] < -rules.kill_switch_usd)
                tp = open_ & (unreal >= rules.take_profit_fraction * item["st"].predicted_edge)
                close = open_ & (kill | tp | flatten_today)
                out["kill"] |= kill
                out["take_profit"] |= tp & ~kill
                # Al cerrar, la estructura pasa a efectivo a su valor de liquidación.
                s["cash"] += np.where(close, liq, 0.0)
                item["open"] &= ~close
        alive &= ~flatten_today
        for s in state:
            rebalance(s, day, ~alive)
        prev_total = np.zeros(P)
        for s in state:
            prev_total += hedge_value(s, day) + sum(np.where(i["open"], risk_values[id(i)], 0.0)
                                                    for i in s["structs"])
    # Fin del horizonte: P&L a liquidación, con la cobertura desarmada al bid/ask.
    last = D1 - 1
    out["pnl_mid"] = np.zeros(P)
    for s in state:
        open_mid = sum(np.where(i["open"], liquidation(i["st"], last, True), 0.0) for i in s["structs"])
        gamma_mid = {u: hedge_option(ho, s["kopt"][:, u], last, "crr")[0] for u, ho in s["book"].gamma_options.items()}
        book_mid = s["cash"] + open_mid + (s["shares"] * paths.spot[:, last, :]).sum(axis=1) + sum(
            s["nopt"][:, u] * 100.0 * gamma_mid[u] for u in gamma_mid)
        out[f"pnl_mid_{s['book'].model}"] = book_mid
        out["pnl_mid"] += book_mid
        value = s["cash"] + sum(np.where(i["open"], values[id(i)], 0.0) for i in s["structs"])
        spot = paths.spot[:, last, :]
        value += (s["shares"] * spot).sum(axis=1) - (np.abs(s["shares"]) * spot).sum(axis=1) * rules.stock_cost_bps * 1e-4
        for u, ho in s["book"].gamma_options.items():
            n = s["nopt"][:, u]
            value += n * 100.0 * (gamma_mid[u] - ho.leg.half * f) - np.abs(n) * rules.option_commission
        out[f"pnl_{s['book'].model}"] = value
        out["pnl"] += value
    return out


def risk_metrics(pnl: np.ndarray) -> dict[str, float]:
    """Media, desvío, VaR y ES al 95% y 99% (pérdidas en positivo)."""
    pnl = np.asarray(pnl, float)
    q95, q99 = np.quantile(pnl, [0.05, 0.01])
    return {
        "media": float(pnl.mean()), "desvio": float(pnl.std(ddof=1)),
        "var95": float(-q95), "es95": float(-pnl[pnl <= q95].mean()),
        "var99": float(-q99), "es99": float(-pnl[pnl <= q99].mean()),
        "peor": float(-pnl.min()),
    }
