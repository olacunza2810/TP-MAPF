"""Evaluación de resultados y control de data snooping.

El motor de backtest produce P&L. Este módulo responde la pregunta difícil: ese
P&L, ¿es señal o es el resultado de haber probado muchas configuraciones hasta
que una funcionó?

La respuesta formal es el **Deflated Sharpe Ratio** (Bailey y López de Prado,
2014). La intuición: si se prueban :math:`N` estrategias sin ningún poder
predictivo, la mejor de ellas igual va a mostrar un Sharpe positivo por puro
azar, y ese máximo esperado crece con :math:`N`. El DSR compara el Sharpe
observado contra ese máximo esperado bajo la hipótesis nula, en vez de contra
cero.

Esto es exactamente lo que la consigna pide cuando exige control contra data
snooping, y es difícil de argumentar sin un número.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .backtest import ArbitrageBacktester, BacktestResult, StrategyParams

_LOG = logging.getLogger(__name__)

__all__ = [
    "PerformanceMetrics",
    "compute_metrics",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "split_in_sample",
    "GridSearchResult",
    "grid_search",
    "trade_breakdown",
    "signal_funnel",
]

_EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Métricas de una curva de P&L.

    Attributes:
        n_trades: Operaciones cerradas.
        total_pnl: P&L total en USD.
        mean_pnl: P&L medio por operación.
        sharpe: Sharpe anualizado de los retornos por período.
        max_drawdown: Máxima caída desde un pico, en USD.
        hit_rate: Fracción de operaciones ganadoras.
        edge_capture: Fracción del edge anunciado que se realizó.
        skew: Asimetría de los retornos por período.
        kurtosis: Curtosis (no exceso) de los retornos por período.
        n_periods: Períodos de la curva de equity.
    """

    n_trades: int
    total_pnl: float
    mean_pnl: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    edge_capture: float
    skew: float
    kurtosis: float
    n_periods: int


def compute_metrics(
    result: BacktestResult, periods_per_year: float = 252.0
) -> PerformanceMetrics:
    """Calcula las métricas de una corrida.

    El Sharpe se computa sobre las variaciones de P&L total, realizado más no
    realizado. Usar sólo el realizado produce una serie escalonada que subestima
    fuertemente la volatilidad: las pérdidas latentes no aparecen hasta que se
    cierra la posición.

    Args:
        result: Resultado del backtest.
        periods_per_year: Períodos por año para anualizar.

    Returns:
        Las métricas de la corrida.
    """
    equity = result.equity
    if equity.empty:
        nan = float("nan")
        return PerformanceMetrics(0, 0.0, nan, nan, nan, nan, nan, nan, nan, 0)

    total = equity["realized_pnl"] + equity["unrealized_pnl"]
    changes = total.diff().dropna()

    if len(changes) > 1 and changes.std(ddof=1) > 0:
        sharpe = float(
            changes.mean() / changes.std(ddof=1) * np.sqrt(periods_per_year)
        )
        skew = float(changes.skew())
        kurt = float(changes.kurtosis() + 3.0)
    else:
        sharpe = skew = kurt = float("nan")

    drawdown = float((total.cummax() - total).max()) if len(total) else float("nan")

    return PerformanceMetrics(
        n_trades=len(result.trades),
        total_pnl=result.total_pnl,
        mean_pnl=(
            float(result.trades["pnl"].mean()) if not result.trades.empty
            else float("nan")
        ),
        sharpe=sharpe,
        max_drawdown=drawdown,
        hit_rate=result.hit_rate,
        edge_capture=result.edge_capture,
        skew=skew,
        kurtosis=kurt,
        n_periods=len(changes),
    )


_BREAKDOWN_COLUMNS = (
    "operaciones", "pnl_total", "pnl_medio", "hit_rate",
    "edge_anunciado", "edge_al_ejecutar", "edge_capture",
)

_FUNNEL_LABELS = {
    "detected": "detectadas",
    "queued": "encoladas",
    "filled": "ejecutadas",
    "rejected_edge_gone": "edge_desaparecido",
    "rejected_no_quote": "sin_precio",
    "rejected_capacity": "sin_capacidad",
    "pending_at_end": "pendientes_al_final",
}


def trade_breakdown(result: BacktestResult, by: str | Sequence[str]) -> pd.DataFrame:
    """Agrega las operaciones cerradas por detector, motivo de salida u otra columna.

    Es la tabla que explica de dónde sale el P&L: un total negativo puede venir
    de un solo detector, o de las salidas por stop-loss y no del vencimiento.

    Args:
        result: Resultado del backtest.
        by: Columna o columnas de ``result.trades`` por las que agrupar.

    Returns:
        Una fila por grupo, ordenada del peor al mejor P&L total.
    """
    columns = [by] if isinstance(by, str) else list(by)
    if result.trades.empty:
        return pd.DataFrame(columns=[*columns, *_BREAKDOWN_COLUMNS])
    trades = result.trades
    if "execution_edge" not in trades.columns:
        trades = trades.assign(execution_edge=np.nan)
    table = (
        trades.groupby(columns, dropna=False)
        .agg(
            operaciones=("pnl", "size"),
            pnl_total=("pnl", "sum"),
            pnl_medio=("pnl", "mean"),
            hit_rate=("pnl", lambda s: float((s > 0).mean())),
            edge_anunciado=("predicted_edge", "sum"),
            edge_al_ejecutar=("execution_edge", "sum"),
        )
        .reset_index()
    )
    announced = table["edge_anunciado"].where(table["edge_anunciado"] != 0)
    table["edge_capture"] = table["pnl_total"] / announced
    return table.sort_values("pnl_total", ignore_index=True)


def signal_funnel(result: BacktestResult) -> pd.DataFrame:
    """Embudo de señales por detector, con una fila de total.

    ``detectadas`` cuenta todas las oportunidades que superaron el umbral;
    ``encoladas``, las que entraron por el tope ``max_positions_per_signal``.
    Cada encolada termina en exactamente una de: ejecutada, edge desaparecido,
    sin precio, sin capacidad o pendiente al final de los datos.
    """
    by_detector = result.diagnostics.get("by_detector", {})
    labels = list(_FUNNEL_LABELS.values())
    if not by_detector:
        return pd.DataFrame(columns=["detector", *labels])
    rows = [
        {"detector": detector,
         **{label: int(counts.get(key, 0)) for key, label in _FUNNEL_LABELS.items()}}
        for detector, counts in sorted(by_detector.items())
    ]
    table = pd.DataFrame(rows)
    total = {"detector": "TOTAL", **table[labels].sum().astype(int).to_dict()}
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark: float = 0.0,
) -> float:
    r"""Probabilidad de que el Sharpe verdadero supere un umbral.

    .. math::

        \widehat{\text{PSR}} = \Phi\!\left(
            \frac{(\hat{SR} - SR^*)\sqrt{n-1}}
                 {\sqrt{1 - \gamma_3 \hat{SR}
                        + \frac{\gamma_4 - 1}{4}\hat{SR}^2}}
        \right)

    Corrige por asimetría :math:`\gamma_3` y curtosis :math:`\gamma_4`, que
    importan mucho acá: una estrategia de arbitraje tiene retornos con
    asimetría negativa (muchas ganancias chicas, pérdidas ocasionales grandes),
    y el Sharpe clásico la sobrevalora.

    Args:
        sharpe: Sharpe observado, no anualizado.
        n_obs: Cantidad de observaciones.
        skew: Asimetría de los retornos.
        kurtosis: Curtosis no excedente.
        benchmark: Sharpe de referencia.

    Returns:
        Probabilidad en :math:`[0,1]`, o ``nan`` si no es computable.
    """
    if n_obs < 2 or not np.isfinite(sharpe):
        return float("nan")
    denominator = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if denominator <= 0:
        return float("nan")
    z = (sharpe - benchmark) * np.sqrt(n_obs - 1) / np.sqrt(denominator)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    r"""Sharpe máximo esperado bajo la hipótesis nula.

    Si se prueban :math:`N` configuraciones sin poder predictivo, el máximo
    Sharpe observado converge a

    .. math::

        \mathbb{E}[\max \hat{SR}] \approx \sqrt{V}\left[
            (1-\gamma)\,\Phi^{-1}\!\left(1 - \tfrac{1}{N}\right)
            + \gamma\,\Phi^{-1}\!\left(1 - \tfrac{1}{Ne}\right)
        \right]

    con :math:`\gamma` la constante de Euler-Mascheroni y :math:`V` la varianza
    de los Sharpe entre configuraciones. Es el umbral que hay que superar para
    poder afirmar que se encontró algo.

    Args:
        n_trials: Configuraciones probadas.
        sharpe_variance: Varianza de los Sharpe entre configuraciones.

    Returns:
        El Sharpe máximo esperado por azar.
    """
    if n_trials < 2 or sharpe_variance <= 0:
        return 0.0
    term = (1.0 - _EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n_trials)
    term += _EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sharpe_variance) * term)


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    n_trials: int,
    sharpe_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Sharpe deflactado por la cantidad de configuraciones probadas.

    Es la PSR evaluada contra el máximo esperado por azar en vez de contra cero.
    Un DSR por debajo de 0.95 indica que el resultado no se distingue de lo que
    produciría la búsqueda sobre ruido.

    Args:
        sharpe: Sharpe observado, no anualizado.
        n_obs: Observaciones de la serie.
        n_trials: Configuraciones probadas en la grilla.
        sharpe_variance: Varianza de los Sharpe entre configuraciones.
        skew: Asimetría de los retornos.
        kurtosis: Curtosis no excedente.

    Returns:
        El DSR en :math:`[0,1]`.
    """
    threshold = expected_max_sharpe(n_trials, sharpe_variance)
    return probabilistic_sharpe_ratio(sharpe, n_obs, skew, kurtosis, threshold)


def split_in_sample(
    quotes: pd.DataFrame, fraction: float = 0.6
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el dataset en in-sample y out-of-sample de forma cronológica.

    El corte es **temporal**, nunca aleatorio. Un split aleatorio sobre series
    financieras deja observaciones del futuro en el conjunto de entrenamiento y
    convierte el out-of-sample en una ficción.

    Args:
        quotes: Dataset curado con ``timestamp``.
        fraction: Fracción del período que va a in-sample.

    Returns:
        Tupla ``(in_sample, out_of_sample)``.

    Raises:
        ValueError: si la fracción no está en :math:`(0,1)`.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("La fracción in-sample debe estar entre 0 y 1.")
    stamps = np.sort(quotes["timestamp"].unique())
    cutoff = stamps[int(len(stamps) * fraction)]
    _LOG.info("Corte in-sample / out-of-sample en %s", cutoff)
    return (
        quotes.loc[quotes["timestamp"] < cutoff].copy(),
        quotes.loc[quotes["timestamp"] >= cutoff].copy(),
    )


@dataclass
class GridSearchResult:
    """Resultado de la búsqueda de hiperparámetros.

    Attributes:
        trials: Una fila por configuración, con hiperparámetros y métricas.
        best_params: Configuración de mayor Sharpe in-sample.
        n_trials: Configuraciones evaluadas.
        sharpe_variance: Varianza de los Sharpe, insumo del DSR.
    """

    trials: pd.DataFrame
    best_params: StrategyParams
    n_trials: int
    sharpe_variance: float

    def deflated(self, oos_metrics: PerformanceMetrics) -> float:
        """Calcula el DSR del resultado out-of-sample.

        Se deflacta con la cantidad de configuraciones probadas **in-sample**,
        que es donde ocurrió la búsqueda. Ese es el punto: el out-of-sample no
        es limpio por definición si la configuración se eligió mirando una
        grilla grande.
        """
        return deflated_sharpe_ratio(
            sharpe=oos_metrics.sharpe / np.sqrt(252.0),
            n_obs=oos_metrics.n_periods,
            n_trials=self.n_trials,
            sharpe_variance=self.sharpe_variance,
            skew=oos_metrics.skew if np.isfinite(oos_metrics.skew) else 0.0,
            kurtosis=(
                oos_metrics.kurtosis if np.isfinite(oos_metrics.kurtosis) else 3.0
            ),
        )


def grid_search(
    quotes: pd.DataFrame,
    underlying: pd.DataFrame,
    grid: dict[str, Sequence[Any]],
    base: StrategyParams | None = None,
    objective: str = "sharpe",
) -> GridSearchResult:
    """Calibra hiperparámetros por búsqueda exhaustiva sobre la grilla.

    Registra **todas** las configuraciones probadas, no sólo la ganadora. Ese
    registro es lo que permite deflactar después: reportar el mejor resultado
    de una grilla de 200 combinaciones sin decir que se probaron 200 es la
    definición operativa de data snooping.

    Args:
        quotes: Dataset in-sample.
        underlying: Barras del subyacente.
        grid: Diccionario de nombre de hiperparámetro a valores a probar.
        base: Configuración base sobre la que se varían los parámetros.
        objective: Métrica a maximizar: ``sharpe``, ``total_pnl`` o
            ``edge_capture``.

    Returns:
        El resultado de la búsqueda.

    Raises:
        ValueError: si la grilla está vacía o el objetivo es desconocido.
    """
    if not grid:
        raise ValueError("La grilla no puede estar vacía.")
    if objective not in {"sharpe", "total_pnl", "edge_capture"}:
        raise ValueError(f"Objetivo desconocido: {objective}")

    base = base or StrategyParams()
    names = list(grid)
    combinations = list(itertools.product(*(grid[n] for n in names)))
    _LOG.info("Grid search: %d configuraciones.", len(combinations))

    engine = ArbitrageBacktester(quotes, underlying)
    rows: list[dict[str, Any]] = []
    best_score, best_params = -np.inf, base

    for values in combinations:
        overrides = dict(zip(names, values, strict=True))
        params = StrategyParams(**{**base.as_dict_for_ctor(), **overrides}) if hasattr(
            base, "as_dict_for_ctor"
        ) else _replace(base, overrides)
        result = engine.run(params)
        metrics = compute_metrics(result)
        score = getattr(metrics, objective)
        rows.append({**overrides, **metrics.__dict__})
        if np.isfinite(score) and score > best_score:
            best_score, best_params = score, params

    trials = pd.DataFrame(rows)
    sharpes = trials["sharpe"].replace([np.inf, -np.inf], np.nan).dropna()
    variance = float(sharpes.var(ddof=1) / 252.0) if len(sharpes) > 1 else 0.0

    return GridSearchResult(
        trials=trials,
        best_params=best_params,
        n_trials=len(combinations),
        sharpe_variance=variance,
    )


def _replace(params: StrategyParams, overrides: dict[str, Any]) -> StrategyParams:
    """Devuelve una copia de los hiperparámetros con los campos sustituidos."""
    import dataclasses

    return dataclasses.replace(params, **overrides)
