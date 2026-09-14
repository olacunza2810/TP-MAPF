"""Motor de backtest event-driven para estrategias de arbitraje de opciones.

Principio de diseño
-------------------
El control anti-lookahead no se implementa por convención sino por
**construcción**. La clase :class:`PointInTimeView` es la única puerta de acceso
a los datos durante el loop, y levanta excepción si se le pide información
posterior al reloj actual. Un backtest que se apoya en "tuvimos cuidado de no
mirar el futuro" no es verificable; uno que no puede acceder al futuro sí.

Los tres sesgos de la consigna se atacan así:

**Lookahead.** :class:`PointInTimeView` corta el acceso, y toda ejecución ocurre
con un retardo configurable respecto de la señal. Detectar una dislocación en el
snapshot de las 15:00 y ejecutar contra ese mismo quote supone latencia cero.

**Survivorship.** Las posiciones no desaparecen cuando el contrato deja de
cotizar. Se marcan a su último valor conocido y se liquidan por payoff al
vencimiento. Un backtest que descarta las posiciones sin quote elimina
justamente los casos donde la liquidez se evaporó.

**Data snooping.** El motor no lo resuelve; lo resuelve :mod:`evaluation` con el
split in-sample / out-of-sample y el ajuste por pruebas múltiples. Acá sólo se
registra cada corrida con sus hiperparámetros para que ese ajuste sea posible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Iterator, Literal, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .arbitrage import ExecutionCosts, execution_edge, run_all_detectors

_LOG = logging.getLogger(__name__)
_NY = ZoneInfo("America/New_York")


def _expiry_instant(expiration: Any) -> pd.Timestamp:
    """Instante de vencimiento: 16:00 de Nueva York del día de expiración, en UTC.

    Tomar la medianoche UTC como vencimiento cerraba la posición con el spot de
    la rueda anterior. Con velas diarias las posiciones sí llegan a vencer, así
    que la diferencia se vuelve visible.
    """
    stamp = pd.Timestamp(expiration)
    if stamp.tz is not None:
        stamp = stamp.tz_convert(_NY).tz_localize(None)
    return (stamp.normalize() + pd.Timedelta(hours=16)).tz_localize(_NY).tz_convert("UTC")


def _session_open(stamp: pd.Timestamp) -> pd.Timestamp:
    """Apertura (09:30 de Nueva York) de la rueda que contiene ``stamp``, en UTC."""
    local = pd.Timestamp(stamp).tz_convert(_NY)
    opening = local.tz_localize(None).normalize() + pd.Timedelta(hours=9, minutes=30)
    return opening.tz_localize(_NY).tz_convert("UTC")

__all__ = [
    "PointInTimeView",
    "LookaheadError",
    "StrategyParams",
    "Position",
    "BacktestResult",
    "ArbitrageBacktester",
]


class LookaheadError(RuntimeError):
    """Se intentó acceder a datos posteriores al reloj del backtest."""


class PointInTimeView:
    """Vista de los datos restringida al instante actual del backtest.

    Es deliberadamente restrictiva: no expone el DataFrame completo, sólo
    cortes hasta ``now``. Cualquier consulta hacia adelante levanta
    :class:`LookaheadError` en vez de devolver datos.
    """

    def __init__(self, quotes: pd.DataFrame, underlying: pd.DataFrame) -> None:
        """Inicializa la vista.

        Args:
            quotes: Dataset curado de opciones, con ``timestamp``.
            underlying: Barras del subyacente, con ``timestamp`` y ``close``.
        """
        self._quotes = quotes.sort_values("timestamp").reset_index(drop=True)
        self._underlying = underlying.sort_values("timestamp").reset_index(drop=True)
        self._now: pd.Timestamp | None = None

    @property
    def now(self) -> pd.Timestamp:
        """Reloj actual del backtest.

        Raises:
            LookaheadError: si se consulta antes de posicionar el reloj.
        """
        if self._now is None:
            raise LookaheadError("El reloj del backtest no fue inicializado.")
        return self._now

    def advance_to(self, timestamp: pd.Timestamp) -> None:
        """Mueve el reloj hacia adelante.

        Args:
            timestamp: Nuevo instante.

        Raises:
            LookaheadError: si se intenta retroceder el reloj.
        """
        if self._now is not None and timestamp < self._now:
            raise LookaheadError(
                f"El reloj no puede retroceder: {self._now} -> {timestamp}"
            )
        self._now = timestamp

    def chain_at(self, timestamp: pd.Timestamp | None = None) -> pd.DataFrame:
        """Devuelve la cadena vigente en un instante.

        Args:
            timestamp: Instante consultado. Por defecto, el reloj actual.

        Returns:
            Las filas cuyo ``timestamp`` coincide exactamente con el pedido.

        Raises:
            LookaheadError: si el instante es posterior al reloj.
        """
        target = timestamp if timestamp is not None else self.now
        if target > self.now:
            raise LookaheadError(
                f"Acceso al futuro: se pidió {target} con reloj en {self.now}."
            )
        return self._quotes.loc[self._quotes["timestamp"] == target]

    def last_quote(self, symbol: str) -> pd.Series | None:
        """Última cotización conocida de un contrato, no posterior al reloj.

        Devuelve ``None`` si el contrato nunca cotizó hasta ahora. Que un
        contrato deje de cotizar no lo elimina de la cartera: esa es la
        diferencia entre marcar una posición ilíquida y borrarla del backtest.
        """
        visible = self._quotes.loc[
            (self._quotes["symbol"] == symbol)
            & (self._quotes["timestamp"] <= self.now)
        ]
        return None if visible.empty else visible.iloc[-1]

    def quote_now(self, symbol: str) -> pd.Series | None:
        """Fila del contrato sellada exactamente en el reloj actual.

        A diferencia de :meth:`last_quote`, no arrastra una fila vieja: si el
        contrato no tiene dato en este instante devuelve ``None``. Es lo que
        necesita la ejecución a la apertura, que sólo es válida si el contrato
        operó en esa rueda.
        """
        rows = self._quotes.loc[
            (self._quotes["symbol"] == symbol) & (self._quotes["timestamp"] == self.now)
        ]
        return None if rows.empty else rows.iloc[-1]

    def underlying_field(self, underlying: str, column: str) -> float:
        """Columna de la barra del subyacente sellada en el reloj actual, o ``NaN``."""
        rows = self._underlying.loc[
            (self._underlying["symbol"] == underlying)
            & (self._underlying["timestamp"] == self.now)
        ]
        if rows.empty or column not in rows.columns:
            return float("nan")
        return float(rows.iloc[-1][column])

    def spot(self, underlying: str, timestamp: pd.Timestamp | None = None) -> float:
        """Último precio conocido del subyacente, no posterior al reloj."""
        target = timestamp if timestamp is not None else self.now
        if target > self.now:
            raise LookaheadError(f"Acceso al futuro: {target} > {self.now}.")
        visible = self._underlying.loc[
            (self._underlying["symbol"] == underlying)
            & (self._underlying["timestamp"] <= target)
        ]
        return float("nan") if visible.empty else float(visible.iloc[-1]["close"])

    def settlement_spot(self, underlying: str, expiration: pd.Timestamp) -> float:
        """Precio de liquidación al vencimiento.

        Es el único acceso legítimo a un dato posterior al reloj, y sólo se
        habilita cuando el reloj ya pasó el vencimiento: liquidar una posición
        vencida requiere el precio de ese día.

        Raises:
            LookaheadError: si el vencimiento todavía no ocurrió.
        """
        expiry = _expiry_instant(expiration)
        if expiry > self.now:
            raise LookaheadError(
                f"Liquidación anticipada: vencimiento {expiry} > reloj {self.now}."
            )
        return self.spot(underlying, expiry)

    def timestamps(self) -> list[pd.Timestamp]:
        """Instantes disponibles en el dataset, en orden."""
        return sorted(self._quotes["timestamp"].unique())


@dataclass(frozen=True, slots=True)
class StrategyParams:
    """Hiperparámetros de la estrategia, calibrables en in-sample.

    Cada campo acá es un grado de libertad, y cada grado de libertad es una
    oportunidad de sobreajustar. La cantidad de combinaciones probadas se
    registra y se usa después para deflactar el Sharpe.

    Attributes:
        min_net_edge: Ganancia neta mínima por contrato, en USD.
        max_spread_rel: Spread relativo máximo tolerado en la peor pata.
        min_open_interest: Open interest mínimo en la peor pata.
        min_volume: Volumen mínimo en la peor pata.
        max_dte: Días al vencimiento máximos.
        max_concurrent_positions: Tope de posiciones abiertas simultáneas.
        max_positions_per_signal: Cuántas oportunidades tomar por instante.
        detectors: Detectores habilitados. ``None`` habilita todos.
        execution_lag: Retardo entre señal y ejecución.
        stop_loss_usd: Pérdida no realizada que fuerza el cierre. ``None``
            desactiva el stop.
        contracts_per_trade: Tamaño de la posición.
        require_edge_at_execution: Si ``True``, al ejecutar se recalcula el
            edge con los precios de ese instante y la operación se descarta si
            quedó por debajo del umbral. Con ``False`` se reproduce el
            comportamiento anterior: se ejecuta aunque el edge haya desaparecido.
        min_edge_at_execution: Umbral en USD por contrato para ese control.
            ``None`` usa ``min_net_edge``.
        execution_price: ``quote`` ejecuta contra el bid/ask del corte en que
            vence el retardo. ``open`` (sólo para velas diarias) ejecuta contra
            ``bid_open``/``ask_open`` de esa rueda, es decir, en la apertura: la
            orden se llena a las 09:30 y la posición ya queda sujeta al
            vencimiento y al stop-loss del cierre de la misma rueda.
    """

    min_net_edge: float = 5.0
    max_spread_rel: float = 0.15
    min_open_interest: float = 50.0
    min_volume: float = 1.0
    max_dte: int = 180
    max_concurrent_positions: int = 20
    max_positions_per_signal: int = 3
    detectors: tuple[str, ...] | None = None
    execution_lag: timedelta = timedelta(minutes=5)
    stop_loss_usd: float | None = 500.0
    contracts_per_trade: float = 1.0
    require_edge_at_execution: bool = True
    min_edge_at_execution: float | None = None
    execution_price: Literal["quote", "open"] = "quote"

    def as_dict(self) -> dict[str, Any]:
        """Serializa los hiperparámetros para el registro de la corrida."""
        return {
            "min_net_edge": self.min_net_edge,
            "max_spread_rel": self.max_spread_rel,
            "min_open_interest": self.min_open_interest,
            "min_volume": self.min_volume,
            "max_dte": self.max_dte,
            "max_concurrent_positions": self.max_concurrent_positions,
            "max_positions_per_signal": self.max_positions_per_signal,
            "detectors": self.detectors,
            "execution_lag_s": self.execution_lag.total_seconds(),
            "stop_loss_usd": self.stop_loss_usd,
            "contracts_per_trade": self.contracts_per_trade,
            "require_edge_at_execution": self.require_edge_at_execution,
            "min_edge_at_execution": self.min_edge_at_execution,
            "execution_price": self.execution_price,
        }


@dataclass(slots=True)
class Position:
    """Posición multi-pata abierta por el backtest.

    Attributes:
        position_id: Identificador correlativo.
        detector: Detector que la originó.
        underlying: Ticker del subyacente.
        legs: Patas con signo, precio de entrada y contrato.
        opened_at: Instante de ejecución, ya con el retardo aplicado.
        entry_cash: Flujo de caja neto al abrir. Positivo es crédito.
        entry_commission: Comisiones pagadas al abrir.
        predicted_edge: Edge en USD que anunciaba el detector en la señal.
        expiration: Vencimiento más lejano de las patas.
        closed_at: Instante de cierre, si ya cerró.
        exit_cash: Flujo de caja neto al cerrar.
        exit_reason: ``expiry``, ``stop_loss``, ``converged`` o ``end_of_data``.
        signal_at: Instante en que el detector generó la señal.
        execution_edge: Edge en USD recalculado con los precios de ejecución,
            neto de la misma comisión que usó el detector.
    """

    position_id: int
    detector: str
    underlying: str
    legs: tuple[dict[str, Any], ...]
    opened_at: pd.Timestamp
    entry_cash: float
    entry_commission: float
    predicted_edge: float
    expiration: pd.Timestamp
    closed_at: pd.Timestamp | None = None
    exit_cash: float = 0.0
    exit_commission: float = 0.0
    exit_reason: str = ""
    signal_at: pd.Timestamp | None = None
    execution_edge: float = float("nan")

    @property
    def is_open(self) -> bool:
        """Indica si la posición sigue abierta."""
        return self.closed_at is None

    @property
    def realized_pnl(self) -> float:
        """P&L realizado, neto de comisiones de entrada y salida."""
        return (
            self.entry_cash
            + self.exit_cash
            - self.entry_commission
            - self.exit_commission
        )


@dataclass
class BacktestResult:
    """Resultado de una corrida.

    Attributes:
        trades: Una fila por posición cerrada.
        equity: Curva de P&L acumulado por instante.
        params: Hiperparámetros usados.
        diagnostics: Contadores del loop.
    """

    trades: pd.DataFrame
    equity: pd.DataFrame
    params: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def total_pnl(self) -> float:
        """P&L total realizado."""
        return float(self.trades["pnl"].sum()) if not self.trades.empty else 0.0

    @property
    def hit_rate(self) -> float:
        """Fracción de operaciones con P&L positivo."""
        if self.trades.empty:
            return float("nan")
        return float((self.trades["pnl"] > 0).mean())

    @property
    def edge_capture(self) -> float:
        """Fracción del edge anunciado que efectivamente se realizó.

        Es la métrica más informativa del backtest. Un detector puede tener
        razón sobre la existencia de la dislocación y aun así capturar una
        fracción pequeña, porque entre la señal y el fill el mercado se movió.
        Valores muy por debajo de 1 indican que la estrategia es correcta en
        teoría e inejecutable en la práctica.
        """
        if self.trades.empty:
            return float("nan")
        predicted = self.trades["predicted_edge"].sum()
        return float(self.trades["pnl"].sum() / predicted) if predicted else float("nan")


class ArbitrageBacktester:
    """Loop event-driven sobre los instantes del dataset."""

    def __init__(
        self,
        quotes: pd.DataFrame,
        underlying: pd.DataFrame,
        costs: ExecutionCosts | None = None,
    ) -> None:
        """Inicializa el backtester.

        Args:
            quotes: Dataset curado de opciones.
            underlying: Barras del subyacente.
            costs: Costos de ejecución.
        """
        self._quotes = quotes
        self._underlying = underlying
        self._costs = costs or ExecutionCosts()

    # ------------------------------------------------------------------ #
    # Valuación
    # ------------------------------------------------------------------ #

    def _leg_close_price(
        self, leg: dict[str, Any], view: PointInTimeView
    ) -> float | None:
        """Precio al que se puede cerrar una pata, cruzando el spread.

        Cerrar una pata larga implica vender al bid; cerrar una corta, comprar
        al ask. Marcar al mid subestima el costo de salida y es una de las
        formas más comunes de inflar un backtest de arbitraje.
        """
        if leg["kind"] == "stock":
            spot = view.spot(str(leg["symbol"]))
            return None if not np.isfinite(spot) else spot

        quote = view.last_quote(str(leg["symbol"]))
        if quote is None:
            return None
        side = "bid" if leg["qty"] > 0 else "ask"
        price = float(quote[side])
        return price if np.isfinite(price) and price > 0 else None

    def _settlement_value(
        self, leg: dict[str, Any], view: PointInTimeView, underlying: str
    ) -> float:
        """Valor de liquidación de una pata al vencimiento.

        Las opciones liquidan por valor intrínseco; la acción, a su precio.
        """
        if leg["kind"] == "stock":
            return view.spot(underlying)
        spot = view.settlement_spot(underlying, leg["expiration"])
        strike = float(leg["strike"])
        if not np.isfinite(spot):
            return 0.0
        return max(spot - strike, 0.0) if leg["option_type"] == "c" else max(
            strike - spot, 0.0
        )

    def _mark_to_market(
        self, position: Position, view: PointInTimeView
    ) -> float | None:
        """Valor de cierre de la posición completa, o ``None`` si no es marcable."""
        total = 0.0
        for leg in position.legs:
            price = self._leg_close_price(leg, view)
            if price is None:
                return None
            total += leg["qty"] * price
        return total * self._costs.multiplier

    # ------------------------------------------------------------------ #
    # Ejecución
    # ------------------------------------------------------------------ #

    def _execute(
        self,
        signal: pd.Series,
        view: PointInTimeView,
        params: StrategyParams,
        position_id: int,
        signal_at: pd.Timestamp | None = None,
    ) -> tuple[Position | None, str]:
        r"""Abre una posición ejecutando contra los quotes del instante de fill.

        La señal se genera en :math:`t` y se ejecuta en el primer instante
        disponible a partir de :math:`t + \text{lag}`, re-cotizando cada pata.
        Si alguna pata dejó de cotizar en el ínterin, la operación se descarta
        entera: no se ejecutan estrategias parciales, porque una mariposa a la
        que le falta un ala no tiene payoff acotado.

        **Control del edge.** Con los precios de ejecución se recalcula el edge
        con la misma fórmula del detector. El crédito de cada detector tiene una
        parte de caja (lo que se cobra al armar las patas) y, en algunos, una
        parte estructural que no cambia con los precios: el ancho del spread
        vertical, el strike descontado de la paridad o el valor del modelo. Esa
        parte se obtiene de la señal,

        .. math::

            \text{estructural} = \text{crédito bruto} - \text{caja}_{\text{señal}},

        y el edge al ejecutar es
        :math:`\text{caja}_{\text{ejecución}} + \text{estructural} - \text{comisión}`,
        con la misma comisión que usó el detector. Si los precios no se movieron,
        coincide exactamente con el edge anunciado.

        Returns:
            ``(posición, "filled")``, o ``(None, motivo)`` con motivo
            ``no_quote`` o ``edge_gone``.
        """
        legs = tuple(signal["leg_spec"])
        if not legs:
            return None, "no_quote"

        at_open = params.execution_price == "open"
        cash = 0.0
        refreshed: list[dict[str, Any]] = []
        for leg in legs:
            if leg["kind"] == "stock":
                spot = (
                    view.underlying_field(str(leg["symbol"]), "open")
                    if at_open
                    else view.spot(str(leg["symbol"]))
                )
                if not np.isfinite(spot):
                    return None, "no_quote"
                fill = spot
            else:
                # A la apertura sólo vale la vela de esta rueda: arrastrar la
                # de una rueda anterior sería ejecutar a un precio que ya no
                # existía cuando se envió la orden.
                symbol = str(leg["symbol"])
                quote = view.quote_now(symbol) if at_open else view.last_quote(symbol)
                if quote is None:
                    return None, "no_quote"
                side = "ask" if leg["qty"] > 0 else "bid"
                column = f"{side}_open" if at_open else side
                fill = float(quote[column]) if column in quote.index else float("nan")
                if not np.isfinite(fill) or fill <= 0:
                    return None, "no_quote"
            cash -= leg["qty"] * fill * self._costs.multiplier
            refreshed.append({**leg, "price": fill})

        residual_edge = execution_edge(
            signal, [leg["price"] for leg in refreshed], self._costs,
            params.contracts_per_trade,
        )
        if params.require_edge_at_execution:
            threshold = (
                params.min_edge_at_execution
                if params.min_edge_at_execution is not None
                else params.min_net_edge
            ) * params.contracts_per_trade
            if not residual_edge >= threshold:
                return None, "edge_gone"

        n_legs = len(refreshed) * params.contracts_per_trade
        commission = self._costs.commission_per_contract * n_legs
        expirations = [
            pd.Timestamp(l["expiration"])
            for l in refreshed
            if l["kind"] == "option" and pd.notna(l["expiration"])
        ]
        return Position(
            position_id=position_id,
            detector=str(signal["detector"]),
            underlying=str(signal["underlying"]),
            legs=tuple(refreshed),
            opened_at=_session_open(view.now) if at_open else view.now,
            entry_cash=cash * params.contracts_per_trade,
            entry_commission=commission,
            predicted_edge=float(signal["net_edge_usd"]) * params.contracts_per_trade,
            expiration=max(expirations) if expirations else pd.NaT,
            signal_at=signal_at,
            execution_edge=residual_edge,
        ), "filled"

    def _close(
        self,
        position: Position,
        view: PointInTimeView,
        reason: str,
        params: StrategyParams,
    ) -> None:
        """Cierra una posición, por mercado o por liquidación al vencimiento."""
        if reason == "expiry":
            value = sum(
                leg["qty"] * self._settlement_value(leg, view, position.underlying)
                for leg in position.legs
            ) * self._costs.multiplier
            commission = 0.0
        else:
            marked = self._mark_to_market(position, view)
            if marked is None:
                marked = 0.0
                reason = f"{reason}_sin_quote"
            value = marked
            commission = self._costs.commission_per_contract * len(position.legs)

        position.exit_cash = value * params.contracts_per_trade
        position.exit_commission = commission * params.contracts_per_trade
        position.closed_at = view.now
        position.exit_reason = reason

    # ------------------------------------------------------------------ #
    # Loop principal
    # ------------------------------------------------------------------ #

    def run(self, params: StrategyParams) -> BacktestResult:
        """Corre el backtest completo.

        Args:
            params: Hiperparámetros de la estrategia.

        Returns:
            El resultado con operaciones, curva de equity y diagnósticos.
        """
        view = PointInTimeView(self._quotes, self._underlying)
        stamps = view.timestamps()
        if not stamps:
            raise ValueError("El dataset no tiene instantes.")

        open_positions: list[Position] = []
        closed: list[Position] = []
        equity_rows: list[dict[str, Any]] = []
        # (instante de ejecución, instante de la señal, señal)
        pending: list[tuple[pd.Timestamp, pd.Timestamp, pd.Series]] = []
        counters: dict[str, int] = {
            "signals": 0, "queued": 0, "filled": 0, "rejected_no_quote": 0,
            "rejected_edge_gone": 0, "rejected_capacity": 0, "pending_at_end": 0,
        }
        by_detector: dict[str, dict[str, int]] = {}
        next_id = 0

        def bump(detector: str, key: str, amount: int = 1) -> None:
            """Suma al contador total y al del detector."""
            if key != "detected":
                counters[key] += amount
            row = by_detector.setdefault(
                detector, {k: 0 for k in ("detected", *counters) if k != "signals"}
            )
            row[key] += amount

        signal_costs = ExecutionCosts(
            commission_per_contract=self._costs.commission_per_contract,
            multiplier=self._costs.multiplier,
            min_net_edge=params.min_net_edge,
            stock_borrow_rate=self._costs.stock_borrow_rate,
        )

        if params.execution_price not in ("quote", "open"):
            raise ValueError(f"execution_price desconocido: {params.execution_price}")
        at_open = params.execution_price == "open"

        def execute_ready(stamp: pd.Timestamp) -> None:
            """Ejecuta las señales cuyo retardo ya venció.

            En modo ``open`` corre antes de cerrar posiciones: la orden se llena
            en la apertura, así que la capacidad disponible es la que había a
            esa hora, y la posición nueva ya queda sujeta al vencimiento y al
            stop-loss del cierre de la misma rueda.
            """
            nonlocal pending, next_id, open_positions
            ready = [item for item in pending if stamp >= item[0]]
            pending = [item for item in pending if stamp < item[0]]
            for index, (_, signal_at, signal) in enumerate(ready):
                if len(open_positions) >= params.max_concurrent_positions:
                    for _, _, skipped in ready[index:]:
                        bump(str(skipped["detector"]), "rejected_capacity")
                    break
                position, outcome = self._execute(
                    signal, view, params, next_id, signal_at
                )
                if position is None:
                    bump(str(signal["detector"]), f"rejected_{outcome}")
                    continue
                open_positions.append(position)
                bump(position.detector, "filled")
                next_id += 1

        for stamp in stamps:
            view.advance_to(stamp)
            if at_open:
                execute_ready(stamp)

            # 1. Cerrar lo que corresponda.
            still_open: list[Position] = []
            for position in open_positions:
                if pd.notna(position.expiration):
                    if stamp >= _expiry_instant(position.expiration):
                        self._close(position, view, "expiry", params)
                        closed.append(position)
                        continue
                if params.stop_loss_usd is not None:
                    marked = self._mark_to_market(view=view, position=position)
                    if marked is not None:
                        unrealized = (
                            position.entry_cash + marked - position.entry_commission
                        )
                        if unrealized < -abs(params.stop_loss_usd):
                            self._close(position, view, "stop_loss", params)
                            closed.append(position)
                            continue
                still_open.append(position)
            open_positions = still_open

            # 2. Ejecutar señales cuyo retardo ya venció (en modo open ya se hizo).
            if not at_open:
                execute_ready(stamp)

            # 3. Generar señales nuevas sobre la cadena visible.
            chain = view.chain_at()
            if not chain.empty:
                filtered = self._apply_param_filters(chain, params)
                if not filtered.empty:
                    signals = run_all_detectors(filtered, signal_costs)
                    if params.detectors is not None and not signals.empty:
                        signals = signals.loc[
                            signals["detector"].isin(params.detectors)
                        ]
                    counters["signals"] += len(signals)
                    for detector, count in signals["detector"].value_counts().items():
                        bump(str(detector), "detected", int(count))
                    for _, signal in signals.head(
                        params.max_positions_per_signal
                    ).iterrows():
                        pending.append((stamp + params.execution_lag, stamp, signal))
                        bump(str(signal["detector"]), "queued")

            # 4. Registrar equity.
            unrealized_total = 0.0
            for position in open_positions:
                marked = self._mark_to_market(position, view)
                if marked is not None:
                    unrealized_total += (
                        position.entry_cash + marked - position.entry_commission
                    )
            equity_rows.append(
                {
                    "timestamp": stamp,
                    "realized_pnl": sum(p.realized_pnl for p in closed),
                    "unrealized_pnl": unrealized_total,
                    "open_positions": len(open_positions),
                }
            )

        # Cierre forzado al final de los datos.
        for position in open_positions:
            self._close(position, view, "end_of_data", params)
            closed.append(position)

        # Señales encoladas cuyo retardo cae después del último dato.
        for _, _, signal in pending:
            bump(str(signal["detector"]), "pending_at_end")

        return BacktestResult(
            trades=self._trades_frame(closed),
            equity=pd.DataFrame(equity_rows),
            params=params.as_dict(),
            diagnostics={**counters, "by_detector": by_detector},
        )

    @staticmethod
    def _apply_param_filters(
        chain: pd.DataFrame, params: StrategyParams
    ) -> pd.DataFrame:
        """Aplica los filtros calibrables antes de correr los detectores."""
        out = chain
        if "spread_rel" in out.columns:
            out = out.loc[out["spread_rel"].fillna(np.inf) <= params.max_spread_rel]
        if "open_interest" in out.columns:
            out = out.loc[
                out["open_interest"].fillna(0.0) >= params.min_open_interest
            ]
        # Con velas diarias el volumen conocible antes de operar es el de la
        # rueda anterior. En datos NBBO la columna existe por el esquema pero
        # viene vacía, y ahí se sigue usando ``volume``.
        volume_column = (
            "volume_prev_day"
            if "volume_prev_day" in out.columns and out["volume_prev_day"].notna().any()
            else "volume"
        )
        if volume_column in out.columns:
            out = out.loc[out[volume_column].fillna(0.0) >= params.min_volume]
        if "dte" in out.columns:
            out = out.loc[out["dte"] <= params.max_dte]
        return out

    @staticmethod
    def _trades_frame(positions: Sequence[Position]) -> pd.DataFrame:
        """Convierte las posiciones cerradas en un DataFrame de operaciones."""
        if not positions:
            return pd.DataFrame(
                columns=[
                    "position_id", "detector", "underlying", "signal_at",
                    "opened_at", "closed_at", "entry_cash", "exit_cash",
                    "commission", "pnl", "predicted_edge", "execution_edge",
                    "edge_capture", "exit_reason",
                ]
            )
        rows = [
            {
                "position_id": p.position_id,
                "detector": p.detector,
                "underlying": p.underlying,
                "signal_at": p.signal_at,
                "opened_at": p.opened_at,
                "closed_at": p.closed_at,
                "entry_cash": p.entry_cash,
                "exit_cash": p.exit_cash,
                "commission": p.entry_commission + p.exit_commission,
                "pnl": p.realized_pnl,
                "predicted_edge": p.predicted_edge,
                "execution_edge": p.execution_edge,
                "edge_capture": (
                    p.realized_pnl / p.predicted_edge if p.predicted_edge else np.nan
                ),
                "exit_reason": p.exit_reason,
            }
            for p in positions
        ]
        return pd.DataFrame(rows).sort_values("opened_at").reset_index(drop=True)
