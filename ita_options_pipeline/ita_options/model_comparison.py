r"""Black-Scholes contra el binomial sobre la cadena observada (Parte 2, paso 3).

La consigna pide que los dos modelos compitan. Antes de comparar su desempeño
operando hay que saber **dónde** se separan, y si esa separación es más grande
que el costo de cruzar el spread. Este módulo mide eso con la misma convención
de dividendos para los dos (spot escrowed, ver :mod:`ita_options.greeks`), así
que toda diferencia entre ellos es el derecho de ejercicio anticipado.

Prima de ejercicio anticipado
-----------------------------
Con la misma volatilidad :math:`\sigma(K)` en los dos modelos,

.. math::

    \mathrm{EEP} = V^{\mathrm{CRR}}_{\mathrm{am}}(\sigma) - V^{\mathrm{CRR}}_{\mathrm{eu}}(\sigma) \;\ge\; 0 ,

medida entre el árbol americano y el mismo árbol sin ejercicio anticipado (el
europeo del árbol converge a Black-Scholes; restarlo del mismo árbol deja afuera
el error de discretización).

La teoría dice dónde tiene que aparecer:

- **calls:** sólo antes de una fecha ex. Ejercer en la víspera cobra el
  dividendo :math:`d` y pierde el valor tiempo; conviene si :math:`d` supera al
  valor extrínseco que queda, es decir en calls muy ITM con la fecha ex cerca.
  Sin dividendos un call americano vale lo mismo que el europeo (Merton, 1973).
- **puts:** ejercer cobra :math:`K` antes y gana el interés
  :math:`K(1-e^{-r\tau})`; la prima crece con lo ITM, con el plazo y con
  :math:`r`, y un dividendo próximo la reduce (después de la fecha ex la acción
  vale menos y conviene esperar).

Repricing
---------
Cada modelo se calibra como se calibraría en la práctica: invierte la IV de las
opciones **fuera del dinero** con su propio precio y ajusta la sonrisa
:math:`\sigma(k)=a+bk+ck^2` en :math:`k=\ln(K/F)`, con :math:`F=(S-D)e^{r\tau}`.
Las opciones dentro del dinero no entran al ajuste: valuarlas con la sonrisa
de las OTM del mismo strike es una prueba fuera de muestra en el corte
transversal, y es justo donde un modelo europeo tendría que fallar.

Diebold-Mariano
---------------
Para cada grupo de contratos se arma la serie diaria del diferencial de pérdidas

.. math::

    d_t = \frac{1}{n_t}\sum_{i \in t} \big( L^{\mathrm{BS}}_{i} - L^{\mathrm{CRR}}_{i} \big),

con :math:`L = |\hat V - m|/(s/2)` (error en medios spreads) o
:math:`L = (\hat V - m)^2` en USD. Promediar dentro del día evita tratar como
independientes a contratos que comparten el mismo shock. El estadístico usa la
varianza de largo plazo de Newey-West y la corrección de muestra chica de
Harvey, Leybourne y Newbold (1997), contra una :math:`t` de Student con
:math:`T-1` grados de libertad:

.. math::

    DM = \frac{\bar d}{\sqrt{\hat\Omega_{NW}/T}}, \qquad
    DM^* = DM\,\sqrt{\frac{T-1}{T}} .

:math:`\bar d > 0` significa que el binomial erra menos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .dolthub import projected_dividends
from .execution.model_rv import Smile
from .escrowed import equivalent_yield, implied_vol, model_price
from .greeks import crr_price, pv_dividends

__all__ = [
    "ComparisonSettings",
    "contract_dividends",
    "model_price",
    "implied_vol",
    "equivalent_yield",
    "fit_escrowed_smile",
    "value_slice",
    "add_buckets",
    "diebold_mariano",
    "dm_table",
    "bucket_summary",
]

Model = Literal["crr", "bsm", "crr_q"]
Dividends = Sequence[tuple[float, float]]
#: ``crr``: árbol americano escrowed; ``bsm``: Black-Scholes europeo escrowed;
#: ``crr_q``: árbol americano con yield continuo equivalente (el pricer que usa
#: hoy la estrategia de valor relativo), calibrado con su propia sonrisa.
MODELS: tuple[Model, ...] = ("crr", "bsm", "crr_q")


@dataclass(frozen=True, slots=True)
class ComparisonSettings:
    """Parámetros del estudio.

    Attributes:
        steps: Pasos del árbol (los mismos 100 que la estrategia y la calibración).
        min_dte, max_dte: Plazos incluidos.
        min_fit_points: Mínimo de opciones OTM con IV válida para ajustar la sonrisa.
        atm_band: Semiancho de la banda ATM en :math:`|k|`.
        itm_deep: Umbral de "muy dentro del dinero" en :math:`|k|`.
    """

    steps: int = 100
    min_dte: int = 7
    max_dte: int = 120
    min_fit_points: int = 5
    atm_band: float = 0.03
    itm_deep: float = 0.10


# --------------------------------------------------------------------------- #
# Precios e IV con spot escrowed
# --------------------------------------------------------------------------- #


def contract_dividends(dividends: pd.DataFrame, underlying: str, asof: date, expiration: date) -> list[tuple[float, float]]:
    """Dividendos proyectados point-in-time en la vida del contrato, como ``(años, monto)``."""
    flows = projected_dividends(dividends, underlying, asof, expiration)
    return [((d - asof).days / 365.0, a) for d, a in flows]


def fit_escrowed_smile(slice_: pd.DataFrame, model: Model, dividends: Dividends,
                       settings: ComparisonSettings = ComparisonSettings()) -> tuple[Smile, dict] | None:
    r"""Sonrisa cuadrática del modelo ajustada sobre las opciones OTM de un vencimiento.

    Mismos pesos que :func:`ita_options.execution.model_rv.fit_smile` (proxy de
    vega sobre el spread), pero con la IV del modelo escrowed.

    Returns:
        ``(sonrisa, ivs)`` con ``ivs`` = IV observada por índice de fila OTM, o
        ``None`` si no alcanzan los puntos.
    """
    row0 = slice_.iloc[0]
    spot, tau, rate = float(row0["underlying_price"]), float(row0["tau"]), float(row0["risk_free_rate"])
    forward = (spot - pv_dividends(dividends, rate, tau)) * math.exp(rate * tau)
    otm = slice_.loc[(slice_["bid"] > 0)
                     & (((slice_["option_type"] == "p") & (slice_["strike"] < forward))
                        | ((slice_["option_type"] == "c") & (slice_["strike"] >= forward)))]
    ks, ivs, weights, by_index = [], [], [], {}
    for idx, r in zip(otm.index, otm.itertuples(index=False)):
        mid = 0.5 * (r.bid + r.ask)
        iv = implied_vol(model, mid, spot, float(r.strike), tau, rate, str(r.option_type), dividends, settings.steps)
        if not np.isfinite(iv):
            continue
        k = math.log(float(r.strike) / forward)
        ks.append(k)
        ivs.append(iv)
        weights.append(math.exp(-0.5 * (k / (iv * math.sqrt(tau))) ** 2) / max(r.ask - r.bid, 0.01))
        by_index[idx] = iv
    if len(ks) < settings.min_fit_points:
        return None
    k_arr, iv_arr, w = np.array(ks), np.array(ivs), np.sqrt(np.array(weights))
    design = np.column_stack([np.ones_like(k_arr), k_arr, k_arr ** 2])
    coef, *_ = np.linalg.lstsq(design * w[:, None], iv_arr * w, rcond=None)
    resid = iv_arr - design @ coef
    smile = Smile(float(coef[0]), float(coef[1]), float(coef[2]), float(forward), len(ks),
                  float(np.sqrt(np.mean(resid ** 2))))
    return smile, by_index


def value_slice(slice_: pd.DataFrame, dividends: Dividends,
                settings: ComparisonSettings = ComparisonSettings(),
                spot_column: str = "underlying_price") -> pd.DataFrame:
    r"""Valúa con los tres modelos todos los contratos de un subyacente y vencimiento.

    Args:
        spot_column: Columna con el spot. ``underlying_price`` es el cierre de la
            acción; ``implied_spot`` el spot implícito por paridad en los pares
            ATM de la misma foto, que corrige el desfase horario entre la foto de
            opciones y el cierre (para una opción muy ITM, con :math:`\Delta\approx 1`,
            ese desfase pasa entero al error).

    Columnas de salida por contrato:

    - ``sigma_<m>``, ``v_<m>``: sonrisa de cada modelo en el strike y precio con
      **su** sonrisa (repricing), para ``m`` en ``crr``, ``bsm`` y ``crr_q``.
    - ``eep``: prima de ejercicio anticipado a la misma :math:`\sigma`
      (la del binomial): árbol americano menos árbol europeo.
    - ``en_ajuste``: si el contrato es OTM y entró al ajuste de la sonrisa.
    - ``dias_a_ex``: días a la primera fecha ex dentro de la vida (``nan`` si no hay).
    """
    if spot_column != "underlying_price":
        slice_ = slice_.assign(underlying_price=slice_[spot_column].astype(float))
    fits = {m: fit_escrowed_smile(slice_, m, dividends, settings) for m in MODELS}
    if any(f is None for f in fits.values()):
        return pd.DataFrame()
    smiles = {m: fits[m][0] for m in MODELS}
    in_fit = fits["crr"][1]
    row0 = slice_.iloc[0]
    spot, tau, rate = float(row0["underlying_price"]), float(row0["tau"]), float(row0["risk_free_rate"])
    D = pv_dividends(dividends, rate, tau)
    first_ex = min((ti for ti, _ in dividends if 0 < ti <= tau), default=float("nan"))
    rows = []
    for idx, r in zip(slice_.index, slice_.itertuples(index=False)):
        K, flag = float(r.strike), str(r.option_type)
        row = {
            "trade_date": r.trade_date, "underlying": r.underlying, "expiration": r.expiration,
            "option_type": flag, "strike": K, "dte": int(r.dte), "tau": tau, "spot": spot,
            "forward": smiles["crr"].forward, "k": math.log(K / smiles["crr"].forward),
            "bid": float(r.bid), "ask": float(r.ask), "mid": 0.5 * (r.bid + r.ask),
            "half": max(0.5 * (r.ask - r.bid), 0.005), "pv_div": D,
            "dias_a_ex": first_ex * 365.0 if np.isfinite(first_ex) else np.nan,
            "en_ajuste": idx in in_fit,
            "intrinseco": max(spot - K, 0.0) if flag == "c" else max(K - spot, 0.0),
        }
        try:
            for m in MODELS:
                sig = smiles[m].sigma(K)
                row[f"sigma_{m}"] = sig
                row[f"v_{m}"] = model_price(m, spot, K, tau, rate, sig, flag, dividends, settings.steps)
            # La prima se mide entre el árbol americano y el mismo árbol europeo: así
            # no la contamina el error de discretización del árbol frente a BS.
            row["eep"] = row["v_crr"] - crr_price(spot, K, tau, rate, row["sigma_crr"], flag, dividends,
                                                  steps=settings.steps, american=False)
        except ValueError:
            continue
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Grupos, métricas y test
# --------------------------------------------------------------------------- #


def add_buckets(frame: pd.DataFrame, settings: ComparisonSettings = ComparisonSettings()) -> pd.DataFrame:
    r"""Agrega los grupos de análisis y los errores de cada modelo.

    - ``dinero``: con :math:`x = -k` para calls y :math:`x = k` para puts
      (:math:`x>0` es dentro del dinero): ``OTM``, ``ATM``, ``ITM`` y ``ITM profundo``.
    - ``plazo``: 7-30, 31-60 y 61-120 días.
    - ``fecha_ex``: sin fecha ex en la vida, ex en 7 días o menos, 8-30 y más de 30.
    """
    out = frame.copy()
    x = np.where(out["option_type"] == "c", -out["k"], out["k"])
    out["dinero"] = np.select(
        [x < -settings.atm_band, x <= settings.atm_band, x <= settings.itm_deep],
        ["OTM", "ATM", "ITM"], default="ITM profundo")
    out["plazo"] = pd.cut(out["dte"], [settings.min_dte - 1, 30, 60, settings.max_dte],
                          labels=["7-30 días", "31-60 días", "61-120 días"]).astype(str)
    out["fecha_ex"] = np.select(
        [out["dias_a_ex"].isna(), out["dias_a_ex"] <= 7, out["dias_a_ex"] <= 30],
        ["sin fecha ex", "ex en <=7 días", "ex en 8-30 días"], default="ex en >30 días")
    for m in MODELS:
        err = out[f"v_{m}"] - out["mid"]
        out[f"err_{m}"] = err
        out[f"L_{m}"] = err.abs() / out["half"]
        out[f"L2_{m}"] = err ** 2
        out[f"dentro_{m}"] = (out[f"v_{m}"] >= out["bid"]) & (out[f"v_{m}"] <= out["ask"])
    out["eep_en_medios_spreads"] = out["eep"] / out["half"]
    # Efecto del tratamiento del dividendo, cada árbol con su propia calibración.
    out["efecto_div"] = out["v_crr"] - out["v_crr_q"]
    return out


def diebold_mariano(daily: pd.Series, lags: int | None = None) -> dict[str, float]:
    r"""Test de Diebold-Mariano (con corrección HLN) sobre una serie de diferenciales.

    Args:
        daily: Serie temporal de :math:`d_t` (una observación por fecha).
        lags: Rezagos de Newey-West; por defecto :math:`\lfloor 4 (T/100)^{2/9} \rfloor`.

    Returns:
        ``T``, ``d_medio``, ``dm`` (corregido) y ``p_valor`` a dos colas.
    """
    d = np.asarray(pd.Series(daily).dropna(), dtype=float)
    T = len(d)
    if T < 5:
        return {"T": T, "d_medio": float(np.mean(d)) if T else np.nan, "dm": np.nan, "p_valor": np.nan}
    if lags is None:
        lags = int(math.floor(4 * (T / 100) ** (2 / 9)))
    dc = d - d.mean()
    omega = dc @ dc / T
    for lag in range(1, lags + 1):
        gamma = dc[lag:] @ dc[:-lag] / T
        omega += 2 * (1 - lag / (lags + 1)) * gamma
    if omega <= 0:
        return {"T": T, "d_medio": float(d.mean()), "dm": np.nan, "p_valor": np.nan}
    dm = d.mean() / math.sqrt(omega / T) * math.sqrt((T - 1) / T)
    p = 2 * student_t.sf(abs(dm), df=T - 1)
    return {"T": T, "d_medio": float(d.mean()), "dm": float(dm), "p_valor": float(p)}


def dm_table(frame: pd.DataFrame, by: Sequence[str] = (), loss: str = "L",
             pair: tuple[str, str] = ("bsm", "crr")) -> pd.DataFrame:
    """Diebold-Mariano por grupo entre ``pair = (a, b)``; ``d > 0`` significa que ``b`` erra menos.

    Por defecto BS contra el binomial: ``d > 0`` es a favor del binomial.
    """
    a, b = pair
    groups = frame.groupby(list(by), observed=True) if by else [((), frame)]
    rows = []
    for key, part in groups:
        key = key if isinstance(key, tuple) else (key,)
        diff = (part[f"{loss}_{a}"] - part[f"{loss}_{b}"]).groupby(part["trade_date"]).mean().sort_index()
        res = diebold_mariano(diff)
        rows.append({"comparacion": f"{a} vs {b}", **dict(zip(by, key)), "contratos": len(part),
                     "perdida": loss, **res})
    return pd.DataFrame(rows)


def bucket_summary(frame: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    """Prima de ejercicio anticipado y errores de cada modelo por grupo."""
    g = frame.groupby(list(by), observed=True)
    out = g.agg(
        contratos=("eep", "size"),
        eep_media_usd=("eep", "mean"),
        eep_p90_usd=("eep", lambda s: float(s.quantile(0.9))),
        eep_media_medios_spreads=("eep_en_medios_spreads", "mean"),
        pct_eep_mayor_medio_spread=("eep_en_medios_spreads", lambda s: float((s > 1).mean())),
        efecto_div_medio_usd=("efecto_div", "mean"),
        mae_crr=("err_crr", lambda s: float(s.abs().mean())),
        mae_bsm=("err_bsm", lambda s: float(s.abs().mean())),
        mae_crr_q=("err_crr_q", lambda s: float(s.abs().mean())),
        sesgo_crr=("err_crr", "mean"),
        sesgo_bsm=("err_bsm", "mean"),
        sesgo_crr_q=("err_crr_q", "mean"),
        L_mediana_crr=("L_crr", "median"),
        L_mediana_bsm=("L_bsm", "median"),
        dentro_crr=("dentro_crr", "mean"),
        dentro_bsm=("dentro_bsm", "mean"),
        dentro_crr_q=("dentro_crr_q", "mean"),
    )
    below = frame.assign(_bajo=frame["v_bsm"] < frame["intrinseco"] - 1e-9).groupby(list(by), observed=True)["_bajo"].mean()
    out["pct_bsm_bajo_intrinseco"] = below
    return out.reset_index()
