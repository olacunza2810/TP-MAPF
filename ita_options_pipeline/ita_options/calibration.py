r"""Calibración del mercado sintético a la distribución real de cada subyacente.

Este módulo existe para responder una objeción concreta: **la muestra sintética
debe reproducir la distribución de precios de los activos reales, y cada
parámetro debe estar justificado.** No alcanza con "elegimos números que se ven
razonables": cada supuesto acá abajo sale de una medición pública del activo y
está documentado con su fuente y su lógica financiera.

Universo del trabajo (constituyentes individuales del ETF ITA, no el ETF):

- **RTX** — RTX Corporation (ex-Raytheon). Defensa + aeroespacial comercial.
- **BA**  — The Boeing Company. Aeroespacial comercial + defensa.
- **LMT** — Lockheed Martin. Defensa pura (F-35, misiles).

Por qué LMT como tercer activo, y no GE / GD:

- **GE Aerospace** es la mayor ponderación del ITA (~21%) y tiene las opciones
  más líquidas del sector, pero su serie histórica está contaminada: en abril de
  2024 GE se escindió en tres empresas (GE Aerospace, GE Vernova, GE HealthCare).
  La serie previa a la escisión **no corresponde a la misma entidad**, así que
  calibrar una distribución de retornos sobre ella sería metodológicamente
  incorrecto — justo lo contrario de lo que pide la consigna.
- **LMT** es un *pure-play* de defensa con historia corporativa continua, opciones
  listadas profundas y —esto es lo valioso para poner a prueba los detectores—
  un **régimen de volatilidad distinto** al de RTX y BA: base más baja pero con
  saltos episódicos grandes (cargos por programas, p. ej. la caída por los
  charges de 2025). Así la muestra cubre tres regímenes: BA (vol alta, colas muy
  gordas, sin dividendo), RTX (vol media, dividendo bajo) y LMT (vol de base
  baja, dividendo alto, saltos raros pero severos).

Modelo de retornos: **difusión con saltos de Merton (jump-diffusion)** bajo la
medida riesgo-neutral. Un GBM gaussiano puro subestima las colas; los retornos
de acciones —y BA es el ejemplo de manual, con el 737 MAX y la pandemia— tienen
curtosis en exceso positiva (colas gordas) y asimetría negativa (los cracks son
más bruscos que las subas). Merton reproduce ambos hechos con pocos parámetros
interpretables y **mantiene el pricing coherente**: la volatilidad total del
proceso se fija igual a la IV at-the-money con la que se valúan las opciones, de
modo que la vol realizada del camino iguala a la implícita (el defecto central
del generador anterior era que no lo hacía).

Fuentes de los valores objetivo (consultadas el 13/09/2026; ver informe para el
detalle). Los niveles son de mercado y **se re-estiman con datos propios en
cuanto el recorder de Alpaca tenga historia**: la gracia del módulo es que todo
supuesto vive en un solo lugar, tipado y trazable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "AssetProfile",
    "ASSET_PROFILES",
    "RISK_FREE_RATE",
    "TRADING_DAYS",
    "diffusion_sigma",
    "simulate_daily_path",
    "simulate_snapshot_path",
    "distribution_report",
]

#: Tasa libre de riesgo continua anualizada. Consistente con ``config.py``
#: (~UST a plazos cortos, 2026).
RISK_FREE_RATE: Final[float] = 0.0425

#: Días hábiles bursátiles por año, para anualizar retornos diarios.
TRADING_DAYS: Final[int] = 252


@dataclass(frozen=True, slots=True)
class AssetProfile:
    r"""Perfil calibrado de un subyacente.

    Cada campo es un objetivo medido sobre el activo real. El proceso simulado se
    construye para reproducirlos; :func:`distribution_report` verifica que lo haga.

    Attributes:
        ticker: Símbolo.
        spot: Precio de referencia (nivel actual de mercado). Fija dónde arranca
            el camino; no afecta la *forma* de la distribución de retornos.
        dividend_yield: Dividend yield continuo :math:`q`. Entra tanto en el
            drift riesgo-neutral del camino como en la valuación CRR. **Es
            específico del activo y no un placeholder**: BA lo tiene suspendido
            desde 2020 (0%), RTX paga ~1.6% y LMT ~2.6%.
        annual_vol: Volatilidad total anualizada objetivo (difusión + saltos).
            Se usa como IV at-the-money para valuar y como varianza total del
            proceso, de modo que **implícita = realizada** por construcción.
        jump_intensity: :math:`\lambda`, saltos esperados por año (Poisson).
        jump_mean: :math:`\mu_J`, media del salto en log-retorno. Negativa: los
            saltos de equities son mayoritariamente a la baja.
        jump_vol: :math:`\delta_J`, desvío del tamaño del salto.
        smile_skew: Pendiente del skew de la superficie de IV, :math:`\beta_1` en
            :math:`\sigma(k)=\sigma_{atm}(1-\beta_1 k+\beta_2 k^2)`. Más alto =
            skew más pronunciado (BA > RTX ≈ LMT).
        smile_curv: Curvatura (sonrisa), :math:`\beta_2`.
        source: Nota de trazabilidad de los valores.
    """

    ticker: str
    spot: float
    dividend_yield: float
    annual_vol: float
    jump_intensity: float
    jump_mean: float
    jump_vol: float
    smile_skew: float
    smile_curv: float
    source: str = ""

    def jump_variance(self) -> float:
        r"""Varianza anual aportada por los saltos: :math:`\lambda(\mu_J^2+\delta_J^2)`."""
        return self.jump_intensity * (self.jump_mean**2 + self.jump_vol**2)

    def diffusion_sigma(self) -> float:
        r"""Vol difusiva :math:`\sigma_{diff}` tal que la vol total iguala ``annual_vol``.

        .. math::

            \sigma_{\text{total}}^2 = \sigma_{diff}^2 + \lambda(\mu_J^2+\delta_J^2)
            \;\Rightarrow\;
            \sigma_{diff} = \sqrt{\sigma_{\text{total}}^2 - \lambda(\mu_J^2+\delta_J^2)}

        Si los saltos ya explican toda la varianza objetivo, se deja un piso
        difusivo mínimo para no anular el componente browniano.
        """
        residual = self.annual_vol**2 - self.jump_variance()
        return math.sqrt(max(residual, (0.05 * self.annual_vol) ** 2))


def diffusion_sigma(profile: AssetProfile) -> float:
    """Alias funcional de :meth:`AssetProfile.diffusion_sigma`."""
    return profile.diffusion_sigma()


#: Perfiles calibrados. Valores objetivo tomados de fuentes públicas (13/09/2026).
#:
#: Volatilidades: mezcla de vol realizada trailing y IV 30 días, para que el
#: objetivo represente el régimen del activo y no una foto de un solo día.
#:   RTX: IV30 ~30-32%, realizada trailing ~25-27%  -> objetivo 28%
#:   BA : IV30 ~34%, colas históricamente mucho más gordas -> objetivo 36%
#:   LMT: IV30 ~32% (elevada por charges 2025), base histórica ~20-22% -> 26%
#: Dividendos (yield continuo): RTX 2.92 USD / ~174 ≈ 1.6%; BA suspendido = 0%;
#:   LMT 13.80 USD / ~524 ≈ 2.6%.
#: Saltos: calibrados al horizonte diario, donde se observan (earnings, cracks).
#:   BA: los más frecuentes y grandes (737 MAX, pandemia, pérdidas trimestrales).
#:   RTX: episódicos (p. ej. el recall del motor GTF de Pratt & Whitney, 2023).
#:   LMT: raros pero severos (cargos por programas fixed-price).
ASSET_PROFILES: Final[Mapping[str, AssetProfile]] = {
    "RTX": AssetProfile(
        ticker="RTX",
        spot=174.0,
        dividend_yield=0.016,
        annual_vol=0.28,
        jump_intensity=6.0,
        jump_mean=-0.025,
        jump_vol=0.035,
        smile_skew=0.35,
        smile_curv=0.90,
        source="IV30~30-32%, realizada~26%; div 2.92 USD/~174≈1.6%; jump: shock GTF 2023",
    ),
    "BA": AssetProfile(
        ticker="BA",
        spot=210.0,
        dividend_yield=0.0,
        annual_vol=0.36,
        jump_intensity=8.0,
        jump_mean=-0.035,
        jump_vol=0.055,
        smile_skew=0.55,
        smile_curv=1.30,
        source="IV30~34%, colas gordas (737 MAX/COVID); dividendo suspendido desde 2020",
    ),
    "LMT": AssetProfile(
        ticker="LMT",
        spot=524.0,
        dividend_yield=0.026,
        annual_vol=0.26,
        jump_intensity=4.0,
        jump_mean=-0.030,
        jump_vol=0.045,
        smile_skew=0.35,
        smile_curv=0.85,
        source="IV30~32% (charges 2025), base ~20-22%; div 13.80 USD/~524≈2.6%; jumps raros/severos",
    ),
}


def _simulate_log_returns(
    profile: AssetProfile,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Retornos logarítmicos de un paso bajo Merton, medida riesgo-neutral.

    Para cada paso :math:`\Delta t`:

    .. math::

        r = \Big(\underbrace{k - q - \lambda\kappa - \tfrac12\sigma_{diff}^2}_{\text{drift RN}}\Big)\Delta t
            + \sigma_{diff}\sqrt{\Delta t}\,Z
            + \sum_{i=1}^{N_{\Delta t}} J_i

    con :math:`Z\sim\mathcal N(0,1)`, :math:`N_{\Delta t}\sim\text{Poisson}(\lambda\Delta t)`,
    :math:`J_i\sim\mathcal N(\mu_J,\delta_J^2)` y la corrección de martingala
    :math:`\kappa = e^{\mu_J+\delta_J^2/2}-1`, que asegura que el descuento del
    activo sea una martingala pese a los saltos.
    """
    sigma = profile.diffusion_sigma()
    lam = profile.jump_intensity
    kappa = math.exp(profile.jump_mean + 0.5 * profile.jump_vol**2) - 1.0
    drift = (RISK_FREE_RATE - profile.dividend_yield - lam * kappa
             - 0.5 * sigma**2) * dt
    diffusion = sigma * math.sqrt(dt) * rng.standard_normal(n_steps)

    counts = rng.poisson(lam * dt, n_steps)
    jumps = np.zeros(n_steps)
    for i, c in enumerate(counts):
        if c:
            jumps[i] = rng.normal(profile.jump_mean, profile.jump_vol, c).sum()
    return drift + diffusion + jumps


def simulate_daily_path(
    profile: AssetProfile,
    n_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Camino de precios diario para validar la distribución del activo.

    Es el objeto que debe "parecerse" al activo real: sobre sus retornos diarios
    se miden vol anualizada, skew y curtosis y se comparan con el objetivo.
    """
    r = _simulate_log_returns(profile, n_days, 1.0 / TRADING_DAYS, rng)
    return profile.spot * np.exp(np.cumsum(r))


def simulate_snapshot_path(
    profile: AssetProfile,
    n_snapshots: int,
    interval_minutes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Camino intradía a la cadencia de los snapshots del chain.

    Usa el mismo proceso calibrado que :func:`simulate_daily_path`. A cadencia de
    minutos los saltos aparecen poco (son eventos mayormente overnight/earnings),
    así que la vol realizada intradía queda **entre** la difusiva
    :math:`\sigma_{diff}` y la total ``annual_vol``: para activos con saltos
    grandes (BA) se acerca a la total; para los demás, a la difusiva. En todos los
    casos queda anclada al nivel del activo, no a un número arbitrario.
    """
    minutes_per_year = TRADING_DAYS * 6.5 * 60.0
    dt = interval_minutes / minutes_per_year
    r = _simulate_log_returns(profile, n_snapshots, dt, rng)
    return profile.spot * np.exp(np.cumsum(np.concatenate([[0.0], r]))[: n_snapshots])


def distribution_report(
    n_days: int = 2520,
    n_reps: int = 24,
    seed: int = 7,
    profiles: Mapping[str, AssetProfile] | None = None,
) -> pd.DataFrame:
    """Compara los momentos simulados contra el objetivo calibrado, por activo.

    Los momentos de las colas (skew, curtosis) son ruidosos en una sola
    realización, así que se **promedian sobre ``n_reps`` simulaciones
    independientes** de ``n_days`` retornos diarios cada una. La tabla resultante
    es representativa y no depende de una semilla afortunada — es la evidencia
    directa de que la muestra reproduce la distribución de cada activo.
    """
    profiles = profiles or ASSET_PROFILES
    rows: list[dict[str, object]] = []
    for offset, (tk, prof) in enumerate(profiles.items()):
        vols: list[float] = []
        skews: list[float] = []
        kurts: list[float] = []
        for rep in range(n_reps):
            # Semilla determinista por (activo, réplica): sin hash(), que Python
            # aleatoriza por proceso y rompería la reproducibilidad.
            rng = np.random.default_rng(seed + 1_000 * (offset + 1) + rep)
            r = np.diff(np.log(simulate_daily_path(prof, n_days, rng)))
            vols.append(float(np.std(r, ddof=1) * math.sqrt(TRADING_DAYS)))
            skews.append(float(pd.Series(r).skew()))
            kurts.append(float(pd.Series(r).kurt()))
        rows.append({
            "activo": tk,
            "vol_objetivo": prof.annual_vol,
            "vol_simulada": float(np.mean(vols)),
            "sigma_difusiva": prof.diffusion_sigma(),
            "aporte_saltos_vol": math.sqrt(prof.jump_variance()),
            "skew_simulado": float(np.mean(skews)),
            "curtosis_exceso_simulada": float(np.mean(kurts)),
            "div_yield": prof.dividend_yield,
        })
    return pd.DataFrame(rows)
