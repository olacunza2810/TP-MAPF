r"""Valor relativo contra el modelo calibrado: la estrategia que opera en paper.

Por qué existe
--------------
Con bid/ask observado, los detectores model-free casi no disparan: en la foto
diaria de DoltHub, monotonicidad, spread vertical, butterfly y calendario dan
cero violaciones ejecutables (ver informe, sección de backtest). Esperar un
arbitraje puro dejaría la cuenta paper sin posiciones, y la consigna pide
estrategias en producción sobre las que la Parte 2 compare Black-Scholes contra
el binomial. La devolución lo dice explícito: empezar por butterflies o
verticales, de payoff acotado, con orden multipata límite y tamaño mínimo.

Qué hace
--------
Para cada subyacente y vencimiento:

1. Invierte la volatilidad implícita **del modelo elegido** (CRR americano o
   BSM europeo) en las opciones fuera del dinero, que son las líquidas: puts
   con :math:`K < F` y calls con :math:`K \ge F`.
2. Ajusta una sonrisa cuadrática en log-moneyness,
   :math:`\sigma(k) = a + b\,k + c\,k^2` con :math:`k = \ln(K/F)`, ponderada
   por vega y por la inversa del spread.
3. Valúa con ese modelo y esa sonrisa los verticales de strikes contiguos y las
   butterflies simétricas cerca del dinero.
4. Arma la orden según ``execution``:

   - ``taker``: compra al ask neto si el modelo la valúa por encima, o vende
     al bid neto si la valúa por debajo. Con bid/ask observado esto casi nunca
     ocurre: en la foto de DoltHub de agosto y septiembre, ninguna de 584
     estructuras evaluadas tenía edge después de cruzar el spread.
   - ``maker`` (por defecto): publica un límite **dentro** del spread al
     precio del modelo menos un margen, :math:`L = V - m` para comprar o
     :math:`L = V + m` para vender. Si la orden se llena, el edge contra el
     modelo es :math:`m` por construcción; si no se llena, no hay posición.
     Es proveer liquidez a la cotización del modelo, y la tasa de llenado es
     parte de lo que se mide.

La señal tiene el mismo formato que la de los detectores de arbitraje
(``leg_spec``, ``gross_credit``, ``commission``, ``net_edge_usd``), así que la
consumen sin cambios el control de riesgo previo, el router ``mleg``, el ledger
y :func:`ita_options.arbitrage.execution_edge`.

Qué NO es
---------
No es un arbitraje. El edge depende de que la sonrisa ajustada sea el precio
justo, y se realiza si el mercado converge a ella antes del vencimiento. Por eso
la salida natural es la **convergencia** (``take_profit_fraction`` del edge
anunciado) además del kill-switch y el cierre previo al vencimiento, y por eso
el tamaño es mínimo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ..arbitrage import ExecutionCosts
from ..volatility import bsm_price, crr_american_price, implied_volatility

_LOG = logging.getLogger(__name__)

Model = Literal["crr", "bsm"]


@dataclass(frozen=True, slots=True)
class RVSettings:
    """Parámetros de la estrategia de valor relativo.

    Attributes:
        model: ``crr`` (binomial americano, Parte 1) o ``bsm`` (Parte 2).
        execution: ``taker`` cruza el spread; ``maker`` publica el límite al
            precio del modelo menos el margen, dentro del spread.
        min_dte, max_dte: Ventana de vencimientos operables.
        moneyness_band: :math:`|\\ln(K/F)|` máximo de los strikes de la estructura.
        min_edge_usd: Edge neto mínimo por estructura, en USD.
        min_edge_to_width: Edge mínimo como fracción del ancho bid-ask de la
            estructura: una señal que es más chica que el spread es ruido.
        max_leg_spread_rel: Spread relativo máximo de cada pata.
        min_fit_points: Opciones fuera del dinero mínimas para ajustar la sonrisa.
        steps: Pasos del árbol CRR.
        max_signals: Señales devueltas por llamada.
    """

    model: Model = "crr"
    execution: Literal["taker", "maker"] = "maker"
    min_dte: int = 21
    max_dte: int = 75
    moneyness_band: float = 0.10
    min_edge_usd: float = 10.0
    min_edge_to_width: float = 0.25
    max_leg_spread_rel: float = 0.25
    min_fit_points: int = 5
    steps: int = 150
    max_signals: int = 2


def _price(model: Model, spot: float, strike: float, tau: float, rate: float,
           q: float, sigma: float, flag: str, steps: int) -> float:
    if model == "crr":
        return crr_american_price(spot, strike, tau, rate, q, sigma, flag, steps=steps)
    return bsm_price(spot, strike, tau, rate, q, sigma, flag)


@dataclass(frozen=True, slots=True)
class Smile:
    """Sonrisa cuadrática en log-moneyness para un vencimiento."""

    a: float
    b: float
    c: float
    forward: float
    n_points: int
    rmse_iv: float

    def sigma(self, strike: float) -> float:
        """Volatilidad del modelo para ``strike``, acotada a un rango razonable."""
        k = np.log(strike / self.forward)
        return float(np.clip(self.a + self.b * k + self.c * k * k, 0.03, 3.0))


def fit_smile(slice_: pd.DataFrame, model: Model, settings: RVSettings) -> Smile | None:
    """Ajusta la sonrisa del modelo sobre las opciones fuera del dinero de un vencimiento.

    Args:
        slice_: Cadena de un subyacente y un vencimiento, con ``bid``, ``ask``,
            ``strike``, ``option_type``, ``underlying_price``, ``tau``,
            ``risk_free_rate`` y ``dividend_yield``.
        model: Modelo con el que se invierte la IV.
        settings: Parámetros.

    Returns:
        La sonrisa, o ``None`` si no hay puntos suficientes.
    """
    if slice_.empty:
        return None
    row0 = slice_.iloc[0]
    spot, tau = float(row0["underlying_price"]), float(row0["tau"])
    rate, q = float(row0["risk_free_rate"]), float(row0.get("dividend_yield", 0.0) or 0.0)
    if not (spot > 0 and tau > 0):
        return None
    forward = spot * np.exp((rate - q) * tau)
    otm = slice_.loc[
        (slice_["bid"] > 0)
        & (((slice_["option_type"] == "p") & (slice_["strike"] < forward))
           | ((slice_["option_type"] == "c") & (slice_["strike"] >= forward)))
    ]
    ks, ivs, weights = [], [], []
    for r in otm.itertuples(index=False):
        mid = (r.bid + r.ask) / 2.0
        iv = implied_volatility(mid, spot, float(r.strike), tau, rate, q, str(r.option_type),
                                american=(model == "crr"), steps=settings.steps)
        if not np.isfinite(iv) or not 0.03 < iv < 3.0:
            continue
        k = np.log(float(r.strike) / forward)
        vega_proxy = np.exp(-0.5 * (k / (iv * np.sqrt(tau))) ** 2)
        ks.append(k)
        ivs.append(iv)
        weights.append(vega_proxy / max(r.ask - r.bid, 0.01))
    if len(ks) < settings.min_fit_points:
        return None
    k_arr, iv_arr, w = np.array(ks), np.array(ivs), np.sqrt(np.array(weights))
    design = np.column_stack([np.ones_like(k_arr), k_arr, k_arr**2])
    coef, *_ = np.linalg.lstsq(design * w[:, None], iv_arr * w, rcond=None)
    resid = iv_arr - design @ coef
    return Smile(float(coef[0]), float(coef[1]), float(coef[2]), float(forward), len(ks),
                 float(np.sqrt(np.mean(resid**2))))


def _structure_signal(
    legs: list[tuple[pd.Series, float]], detector: str, smile: Smile, model: Model,
    costs: ExecutionCosts, settings: RVSettings,
) -> dict[str, object] | None:
    """Evalúa una estructura en las dos direcciones y devuelve la mejor señal, si la hay."""
    first = legs[0][0]
    spot, tau = float(first["underlying_price"]), float(first["tau"])
    rate, q = float(first["risk_free_rate"]), float(first.get("dividend_yield", 0.0) or 0.0)
    model_value = 0.0
    width = 0.0
    for row, qty in legs:
        sigma = smile.sigma(float(row["strike"]))
        model_value += qty * _price(model, spot, float(row["strike"]), tau, rate, q, sigma,
                                    str(row["option_type"]), settings.steps)
        width += abs(qty) * (float(row["ask"]) - float(row["bid"]))
    n_contracts = sum(abs(qty) for _, qty in legs)
    commission = costs.commission_per_contract * n_contracts

    mids = [(float(r["bid"]) + float(r["ask"])) / 2.0 for r, _ in legs]
    best = None
    for direction in (+1.0, -1.0):
        signed_qty = [direction * qty for _, qty in legs]
        # Precio neto por acción de la estructura: positivo es débito.
        ask_net = sum(q * (float(r["ask"]) if q > 0 else float(r["bid"])) for (r, _), q in zip(legs, signed_qty))
        bid_net = sum(q * (float(r["bid"]) if q > 0 else float(r["ask"])) for (r, _), q in zip(legs, signed_qty))
        value = direction * model_value
        margin = (settings.min_edge_usd + commission) / costs.multiplier
        if settings.execution == "taker":
            limit = ask_net
        else:
            limit = min(value - margin, ask_net)
            if limit <= bid_net + 0.01:
                continue  # el límite del modelo queda fuera del spread: no hay dónde pararse
        # Precios por pata que suman exactamente el límite: se reparte la
        # distancia al mid en proporción a |cantidad|.
        mid_net = sum(q * m for q, m in zip(signed_qty, mids))
        delta = (limit - mid_net) / sum(abs(q) for q in signed_qty)
        exec_legs = []
        for (row, _), q, m in zip(legs, signed_qty, mids):
            exec_legs.append({
                "kind": "option", "symbol": row["symbol"], "qty": float(q),
                "price": round(m + delta * np.sign(q), 4),
                "strike": float(row["strike"]), "option_type": str(row["option_type"]),
                "expiration": row["expiration"],
            })
        cash = -sum(leg["qty"] * leg["price"] for leg in exec_legs)
        gross_credit = value + cash
        net_edge_usd = gross_credit * costs.multiplier - commission
        if net_edge_usd < settings.min_edge_usd - 1e-9:
            continue
        if settings.execution == "taker" and gross_credit < settings.min_edge_to_width * width:
            continue
        # Agresividad: qué tan cerca del lado contrario queda el límite (1 = cruza).
        aggressiveness = (limit - bid_net) / max(ask_net - bid_net, 1e-9)
        if best is None or aggressiveness > best["aggressiveness"]:
            strikes = "/".join(f"{float(r['strike']):g}" for r, _ in legs)
            side = "compra" if direction > 0 else "venta"
            kind = "call" if str(first["option_type"]) == "c" else "put"
            best = {
                "detector": f"rv_{model}_{detector}",
                "underlying": first["underlying"],
                "timestamp": first.get("timestamp"),
                "option_type": str(first["option_type"]),
                "expiration": first["expiration"],
                "strikes": strikes,
                "legs": (f"{side} {detector} {kind} {strikes} a {limit:+.2f} "
                         f"(bid {bid_net:+.2f} / ask {ask_net:+.2f}, {model} {value:+.2f})"),
                "gross_credit": gross_credit,
                "commission": commission,
                "net_edge": gross_credit - commission / costs.multiplier,
                "net_edge_usd": net_edge_usd,
                "leg_spec": tuple(exec_legs),
                "model_value": value,
                "limit_net": limit,
                "aggressiveness": aggressiveness,
                "smile_rmse_iv": smile.rmse_iv,
                "max_spread_rel": max(float(r["spread_rel"]) for r, _ in legs),
                "symbols": "|".join(str(r["symbol"]) for r, _ in legs),
                "min_volume": np.nan, "min_open_interest": np.nan,
            }
    return best


def detect_model_rv(
    chain: pd.DataFrame, costs: ExecutionCosts | None = None,
    settings: RVSettings | None = None,
) -> pd.DataFrame:
    """Señales de valor relativo en verticales y butterflies contra el modelo calibrado.

    Args:
        chain: Cadena curada de un instante (``build_live_chain`` o DoltHub).
        costs: Costos de ejecución.
        settings: Parámetros de la estrategia.

    Returns:
        Señales ordenadas por edge neto, como máximo ``settings.max_signals``.
    """
    costs = costs or ExecutionCosts()
    settings = settings or RVSettings()
    if chain.empty:
        return pd.DataFrame()
    base = chain.loc[chain["is_tradable"]] if "is_tradable" in chain else chain
    base = base.loc[
        (base["bid"] > 0) & (base["ask"] > base["bid"])
        & (base["spread_rel"] <= settings.max_leg_spread_rel)
    ]
    if "dte" not in base:
        # Sin días al vencimiento no se puede respetar la ventana operable: mejor
        # fallar que operar vencimientos que la estrategia excluye.
        raise KeyError("La cadena no trae 'dte': no se puede aplicar la ventana de vencimientos.")
    base = base.loc[base["dte"].between(settings.min_dte, settings.max_dte)]
    signals: list[dict[str, object]] = []
    for (_, _), slice_ in base.groupby(["underlying", "expiration"]):
        smile = fit_smile(chain.loc[(chain["underlying"] == slice_["underlying"].iloc[0])
                                    & (chain["expiration"] == slice_["expiration"].iloc[0])],
                          settings.model, settings)
        if smile is None:
            continue
        near = slice_.loc[np.abs(np.log(slice_["strike"] / smile.forward)) <= settings.moneyness_band]
        for flag in ("c", "p"):
            side = near.loc[near["option_type"] == flag].sort_values("strike").reset_index(drop=True)
            rows = [side.iloc[i] for i in range(len(side))]
            for i in range(len(rows) - 1):
                sig = _structure_signal([(rows[i], +1.0), (rows[i + 1], -1.0)], "vertical",
                                        smile, settings.model, costs, settings)
                if sig:
                    signals.append(sig)
            for i in range(1, len(rows) - 1):
                lo, mid, hi = rows[i - 1], rows[i], rows[i + 1]
                if not np.isclose(mid["strike"] - lo["strike"], hi["strike"] - mid["strike"]):
                    continue
                sig = _structure_signal([(lo, +1.0), (mid, -2.0), (hi, +1.0)], "butterfly",
                                        smile, settings.model, costs, settings)
                if sig:
                    signals.append(sig)
    if not signals:
        return pd.DataFrame()
    key = "aggressiveness" if settings.execution == "maker" else "net_edge_usd"
    out = pd.DataFrame(signals).sort_values(key, ascending=False)
    return out.head(settings.max_signals).reset_index(drop=True)


class CompetingModels:
    r"""Fuente de señales con el binomial y Black-Scholes compitiendo en la misma cuenta.

    Es la implementación de "Black-Scholes puesto a competir en producción
    contra el binomial ya en ejecución" (consigna, Parte 2). Los dos modelos
    corren la misma estrategia (:func:`detect_model_rv`) con los mismos
    parámetros; sólo cambia el modelo con el que se calibra la sonrisa y se
    valúa cada estructura. Así, la diferencia de resultados es atribuible al
    modelo y no a la regla.

    - **Cupo por modelo.** Cada modelo puede tener a lo sumo ``cap_per_model``
      estructuras abiertas; uno no le quita lugar al otro.
    - **Una estructura, un dueño.** Si los dos modelos eligen las mismas patas,
      o patas que ya están abiertas por cualquiera de los dos, se ejecuta una
      sola vez. La prioridad entre modelos alterna en cada ciclo, para que
      ninguno tenga ventaja sistemática, y la señal descartada queda registrada
      en el ledger (``senal_duplicada``).
    - **Atribución.** El detector de cada estrategia lleva el modelo
      (``rv_crr_vertical``, ``rv_bsm_butterfly``), así que el ledger separa el
      P&L de cada uno.
    - **Registro de cada ciclo.** Aunque no haya llenados, cada ciclo guarda
      cuántas señales dio cada modelo (``senales_por_modelo``): es parte de la
      comparación en producción.
    """

    def __init__(
        self,
        ledger: object,
        costs: ExecutionCosts,
        settings: dict[str, RVSettings],
        cap_per_model: int = 3,
    ) -> None:
        self.ledger = ledger
        self.costs = costs
        self.settings = settings
        self.cap_per_model = cap_per_model
        self._turn = 0

    @staticmethod
    def model_of(detector: str) -> str | None:
        """Modelo de un detector ``rv_<modelo>_<estructura>``, o ``None``."""
        parts = str(detector).split("_")
        return parts[1] if len(parts) >= 3 and parts[0] == "rv" else None

    @staticmethod
    def legs_key(legs) -> frozenset[str]:
        """Identidad por patas, sin el detector: lo que comparte la cuenta."""
        return frozenset(str(leg["symbol"]) for leg in legs if leg.get("kind", "option") == "option")

    def __call__(self, chain: pd.DataFrame) -> pd.DataFrame:
        open_strategies = self.ledger.open_strategies()
        counts = {model: 0 for model in self.settings}
        held: set[frozenset[str]] = set()
        for strategy in open_strategies:
            model = self.model_of(strategy["detector"])
            if model in counts:
                counts[model] += 1
            held.add(self.legs_key(strategy["legs"]))

        order = list(self.settings)
        shift = self._turn % len(order)
        order = order[shift:] + order[:shift]
        self._turn += 1

        candidates: dict[str, list[dict[str, object]]] = {}
        for model in order:
            room = self.cap_per_model - counts[model]
            if room <= 0:
                candidates[model] = []
                continue
            found = detect_model_rv(chain, self.costs, self.settings[model])
            candidates[model] = found.to_dict("records") if not found.empty else []
        self.ledger.event("senales_por_modelo", {
            "prioridad": order[0],
            **{m: len(c) for m, c in candidates.items()},
            **{f"abiertas_{m}": counts[m] for m in counts},
        })

        chosen: list[dict[str, object]] = []
        taken = set(held)
        # Ronda alternada: la mejor de cada modelo, después la segunda, etc.
        for rank in range(max((len(c) for c in candidates.values()), default=0)):
            for model in order:
                if rank >= len(candidates[model]):
                    continue
                if sum(1 for s in chosen if self.model_of(s["detector"]) == model) + counts[model] >= self.cap_per_model:
                    continue
                signal = candidates[model][rank]
                key = self.legs_key(signal["leg_spec"])
                if key in taken:
                    self.ledger.event("senal_duplicada", {
                        "modelo": model, "detector": signal["detector"],
                        "patas": sorted(key), "edge_usd": float(signal["net_edge_usd"]),
                    })
                    continue
                taken.add(key)
                chosen.append(signal)
        return pd.DataFrame(chosen)
