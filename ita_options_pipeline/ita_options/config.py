"""Configuración tipada del pipeline de datos de opciones.

Todo parámetro que afecte al backtest vive acá y se serializa junto al dataset,
de modo que cualquier corrida sea reproducible a partir del artefacto parquet.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Final, Literal, Mapping

__all__ = [
    "AlpacaCredentials",
    "LiquidityThresholds",
    "PricingAssumptions",
    "DailyBarAssumptions",
    "IngestionSettings",
    "ExecutionSettings",
    "PipelineConfig",
]

OptionsFeedName = Literal["opra", "indicative"]
DataMode = Literal["nbbo", "daily"]

#: Universo del trabajo práctico. Subyacentes individuales pertenecientes al ITA.
#: El ETF NO es subyacente válido según la consigna. Se eligen tres regímenes de
#: volatilidad distintos: BA (alta, colas gordas, sin dividendo), RTX (media,
#: dividendo bajo) y LMT (base baja, dividendo alto, saltos raros y severos).
#: LMT se prefiere a GE porque GE se escindió en tres empresas en 2024 y su serie
#: histórica no es representativa de la entidad actual. Ver ``calibration.py``.
DEFAULT_UNDERLYINGS: Final[tuple[str, ...]] = ("RTX", "BA", "LMT")


@dataclass(frozen=True, slots=True)
class AlpacaCredentials:
    """Credenciales de Alpaca leídas de variables de entorno.

    Nunca hardcodear claves en el repositorio: el entregable incluye el repo de
    GitHub y las claves quedarían versionadas.
    """

    api_key: str
    secret_key: str
    paper: bool = True

    @classmethod
    def from_env(cls) -> "AlpacaCredentials":
        """Construye credenciales desde ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY``.

        Raises:
            RuntimeError: si alguna variable de entorno no está definida.
        """
        key = os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError(
                "Faltan ALPACA_API_KEY / ALPACA_SECRET_KEY en el entorno."
            )
        paper = os.environ.get("ALPACA_PAPER", "true").lower() != "false"
        return cls(api_key=key, secret_key=secret, paper=paper)


@dataclass(frozen=True, slots=True)
class LiquidityThresholds:
    """Umbrales de explotabilidad de un contrato.

    Un arbitraje detectado sobre un contrato que no se puede ejecutar no es un
    arbitraje: es un error de datos. Estos filtros son la primera línea de
    defensa contra falsos positivos.

    Attributes:
        min_volume: Volumen negociado estrictamente mayor a este valor.
        min_open_interest: Open interest mínimo (usando el OI rezagado un día).
        max_relative_spread: Cota superior de ``(ask - bid) / mid``.
        min_bid: Bid estrictamente mayor a este valor. Un bid de cero implica que
            no hay contraparte compradora y la pata short es inejecutable.
        min_mid_price: Descarta contratos de precio ínfimo, donde el tick de
            0.01 USD domina cualquier señal del modelo.
        max_dte: Horizonte máximo. Vencimientos muy largos tienen quotes
            indicativos poco confiables.
        min_dte: Horizonte mínimo. Bajo 7 días el gamma y el pin risk dominan.
        data_mode: ``nbbo`` para cotizaciones con bid/ask real (recorder de
            Alpaca). ``daily`` para velas diarias (Polygon gratuito), donde no
            hay spread observado ni open interest histórico: la liquidez se
            juzga por el volumen de la rueda anterior.
        min_prev_day_volume: Sólo en modo ``daily``. Volumen de la rueda
            anterior estrictamente mayor a este valor.
    """

    min_volume: int = 0
    min_open_interest: int = 50
    max_relative_spread: float = 0.15
    min_bid: float = 0.0
    min_mid_price: float = 0.05
    max_dte: int = 180
    min_dte: int = 7
    data_mode: DataMode = "nbbo"
    min_prev_day_volume: float = 0.0

    @classmethod
    def for_daily_bars(cls, **overrides: object) -> "LiquidityThresholds":
        """Umbrales para velas diarias: sin open interest ni spread."""
        base: dict[str, object] = {"data_mode": "daily", "min_open_interest": 0}
        return cls(**{**base, **overrides})  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class PricingAssumptions:
    """Supuestos del modelo de valuación usados para invertir la IV.

    Attributes:
        risk_free_rate: Tasa libre de riesgo continua anualizada. Si se provee
            ``risk_free_curve_path`` se ignora y se interpola por DTE.
        dividend_yield: Dividend yield continuo por ticker, calibrado a mercado
            (2026): BA suspendió su dividendo (0%), RTX paga ~1.6% y LMT ~2.6%.
            Deben recalibrarse contra el mercado cuando haya datos propios.
        risk_free_curve_path: CSV opcional con columnas ``date,tenor_days,rate``.
        binomial_steps: Pasos del árbol CRR para la inversión de IV americana.
        day_count: Convención de conteo de días. 365 calendario para consistencia
            con el uso de mercado en volatilidad implícita.
    """

    risk_free_rate: float = 0.0425
    dividend_yield: Mapping[str, float] = field(
        default_factory=lambda: {"RTX": 0.016, "BA": 0.0000, "LMT": 0.026}
    )
    risk_free_curve_path: Path | None = None
    binomial_steps: int = 256
    day_count: Literal[365, 360, 252] = 365


@dataclass(frozen=True, slots=True)
class DailyBarAssumptions:
    """Supuestos para operar con velas diarias en lugar de NBBO.

    Una vela diaria no trae bid ni ask. Para que los detectores y el backtest
    sigan cruzando el spread, se construyen lados sintéticos alrededor de un
    precio de referencia. Todo resultado obtenido así depende de
    ``assumed_relative_spread`` y debe reportarse con su sensibilidad.

    Attributes:
        price_source: ``close`` (último trade de la rueda) o ``vwap`` (precio
            medio ponderado por volumen). Si falta el VWAP se usa el cierre.
        assumed_relative_spread: Spread relativo supuesto, ``(ask - bid) / mid``.
        min_half_spread: Medio spread mínimo en USD: un tick de 0.01.
    """

    price_source: Literal["close", "vwap"] = "close"
    assumed_relative_spread: float = 0.05
    min_half_spread: float = 0.01


@dataclass(frozen=True, slots=True)
class IngestionSettings:
    """Parámetros de extracción y control de tasa.

    Attributes:
        feed: ``opra`` requiere suscripción Algo Trader Plus. ``indicative`` es
            gratuito pero entrega quotes modificados y con 15 minutos de demora,
            lo que lo hace inservible para arbitraje en vivo y marginal para
            calibración. Debe declararse en la presentación.
        requests_per_minute: Límite del plan. 200/min en el plan gratuito.
        max_concurrency: Corrutinas simultáneas contra el thread pool.
        snapshot_batch_size: Alpaca acepta hasta 100 símbolos por request de
            snapshot.
        max_retries: Reintentos con backoff exponencial ante 429/5xx.
        strike_window: Fracción del spot que delimita los strikes a descargar.
            0.25 significa strikes entre 75% y 125% del spot.
    """

    feed: OptionsFeedName = "indicative"
    requests_per_minute: int = 190
    max_concurrency: int = 8
    snapshot_batch_size: int = 100
    max_retries: int = 5
    strike_window: float = 0.25


@dataclass(frozen=True, slots=True)
class ExecutionSettings:
    """Parámetros del piloto de paper trading.

    Attributes:
        contracts: Unidades por estructura (100 acciones por unidad en la paridad).
        min_edge_usd: Edge neto mínimo por unidad, en detección y al ejecutar.
        max_open_strategies: Tope de estructuras abiertas simultáneas.
        max_loss_per_trade: Tope de pérdida máxima al vencimiento por operación.
        kill_switch_usd: Pérdida no realizada por estructura que fuerza el cierre.
        daily_loss_limit_usd: Pérdida diaria (realizada + no realizada) que
            detiene las entradas y cierra todo.
        cycle_seconds: Segundos entre ciclos con el mercado abierto.
        entry_start_minutes: Minutos después de la apertura antes de entrar.
        entry_stop_minutes: Minutos antes del cierre en que se dejan de abrir
            estructuras y se cierran las que vencen ese día.
        stock_fill_timeout_s: Espera máxima del fill completo de las acciones.
        option_fill_timeout_s: Espera máxima del fill de una ``mleg``.
        unwind_timeout_s: Espera máxima para deshacer acciones.
        order_poll_s: Intervalo entre consultas de estado de una orden.
        coverage_checks: Consultas de posición antes de declarar la cobertura
            en acciones no confirmada.
        stock_limit_tolerance: USD por acción que el límite de la pata de
            acción cede sobre el precio de la señal para ser marcable.
        max_reprices: Re-precios de una ``mleg`` no llenada.
        reprice_step: USD por unidad escalada que empeora cada re-precio.
        close_cushion: USD por unidad escalada que se cede al cerrar.
        max_quote_age_s: Antigüedad máxima de un quote para considerarlo.
        strike_window: Fracción del spot que delimita el universo en vivo.
        max_dte: Horizonte máximo del universo en vivo.
        detectors: Detectores habilitados en vivo.
        kill_file: Si este archivo existe, se detienen las entradas y se cierra todo.
        reconcile_tolerance_cycles: Ciclos seguidos con diferencias de
            reconciliación antes de detener las entradas.
        max_consecutive_errors: Errores de datos o API seguidos que detienen
            las entradas.
    """

    contracts: int = 1
    min_edge_usd: float = 5.0
    max_open_strategies: int = 5
    max_loss_per_trade: float = 2000.0
    kill_switch_usd: float = 2000.0
    daily_loss_limit_usd: float = 3000.0
    cycle_seconds: float = 90.0
    entry_start_minutes: int = 15
    entry_stop_minutes: int = 30
    stock_fill_timeout_s: float = 30.0
    option_fill_timeout_s: float = 60.0
    unwind_timeout_s: float = 30.0
    order_poll_s: float = 2.0
    coverage_checks: int = 5
    stock_limit_tolerance: float = 0.05
    max_reprices: int = 1
    reprice_step: float = 0.05
    close_cushion: float = 0.10
    max_quote_age_s: float = 1200.0
    strike_window: float = 0.10
    max_dte: int = 60
    detectors: tuple[str, ...] = (
        "monotonicity", "vertical_bound", "butterfly", "put_call_parity",
    )
    kill_file: Path = Path("KILL")
    reconcile_tolerance_cycles: int = 2
    max_consecutive_errors: int = 3


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configuración raíz.

    Attributes:
        underlyings: Tickers subyacentes. Deben ser acciones individuales
            listadas en NYSE/Nasdaq pertenecientes al ITA.
        start: Inicio de la ventana de backfill histórico.
        end: Fin de la ventana de backfill histórico.
        data_root: Directorio raíz del data lake local.
    """

    credentials: AlpacaCredentials
    underlyings: tuple[str, ...] = DEFAULT_UNDERLYINGS
    start: date = field(default_factory=lambda: date.today() - timedelta(days=365))
    end: date = field(default_factory=date.today)
    data_root: Path = Path("data")
    liquidity: LiquidityThresholds = field(default_factory=LiquidityThresholds)
    pricing: PricingAssumptions = field(default_factory=PricingAssumptions)
    daily: DailyBarAssumptions = field(default_factory=DailyBarAssumptions)
    ingestion: IngestionSettings = field(default_factory=IngestionSettings)

    def manifest(self) -> dict[str, object]:
        """Devuelve un dict serializable con todos los supuestos, sin secretos."""
        payload = asdict(self)
        payload.pop("credentials", None)
        payload["data_root"] = str(self.data_root)
        payload["start"] = self.start.isoformat()
        payload["end"] = self.end.isoformat()
        curve = payload["pricing"].get("risk_free_curve_path")
        payload["pricing"]["risk_free_curve_path"] = str(curve) if curve else None
        payload["underlyings"] = list(self.underlyings)
        return payload
