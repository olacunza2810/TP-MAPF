"""Diagnóstico de conectividad y disponibilidad de datos contra Alpaca.

Corre las llamadas del pipeline en el mismo orden en que las necesita, y reporta
cuál es la primera que falla. La idea es que un problema de credenciales, de
permisos de opciones o de feed se detecte en dos minutos y no en medio de una
presentación.

Cada chequeo devuelve, además de si pasó, una medición útil: cuántos contratos
trae el universo, qué cobertura de open interest hay, cuántos snapshots vienen
con NBBO real y cuál es el spread mediano. Eso último es lo que determina si la
estrategia es viable: si el spread mediano es del 40%, no hay arbitraje que
sobreviva a cruzarlo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Any, Callable, Coroutine

import numpy as np
import pandas as pd

from .clients import AsyncAlpacaGateway
from .config import PipelineConfig

_LOG = logging.getLogger(__name__)

__all__ = ["CheckResult", "run_diagnostics", "format_report"]


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Resultado de un chequeo individual.

    Attributes:
        name: Nombre legible del chequeo.
        passed: Si el chequeo pasó.
        detail: Medición o mensaje de error.
        blocking: Si un fallo acá impide seguir con los chequeos siguientes.
    """

    name: str
    passed: bool
    detail: str
    blocking: bool = True


async def _probe(
    name: str,
    coro_factory: Callable[[], Coroutine[Any, Any, str]],
    blocking: bool = True,
) -> CheckResult:
    """Ejecuta un chequeo capturando cualquier excepción como fallo.

    Args:
        name: Nombre del chequeo.
        coro_factory: Función sin argumentos que devuelve la corrutina a correr.
            Debe devolver un string con la medición.
        blocking: Si un fallo corta la secuencia.

    Returns:
        El resultado del chequeo.
    """
    try:
        detail = await coro_factory()
        return CheckResult(name, True, detail, blocking)
    except Exception as exc:  # noqa: BLE001 - el diagnóstico reporta, no propaga
        return CheckResult(name, False, f"{type(exc).__name__}: {exc}", blocking)


async def run_diagnostics(config: PipelineConfig) -> list[CheckResult]:
    """Corre la batería completa de chequeos.

    Args:
        config: Configuración del pipeline, con credenciales.

    Returns:
        Lista de resultados en orden de ejecución. Si un chequeo bloqueante
        falla, los posteriores no se ejecutan y se reportan como omitidos.
    """
    # Un diagnóstico debe fallar rápido: reintentar cinco veces con backoff
    # exponencial ante credenciales inválidas sólo hace esperar al usuario.
    fast_fail = replace(config.ingestion, max_retries=1)
    gateway = AsyncAlpacaGateway(config.credentials, fast_fail)
    results: list[CheckResult] = []
    state: dict[str, Any] = {}

    async def check_auth() -> str:
        import asyncio

        account = await asyncio.to_thread(gateway._trading.get_account)
        return (
            f"cuenta {getattr(account, 'account_number', '?')}, "
            f"estado {getattr(account, 'status', '?')}, "
            f"{'paper' if config.credentials.paper else 'LIVE'}"
        )

    results.append(await _probe("1. Autenticación", check_auth))
    if not results[-1].passed:
        return _skip_rest(results, 5)

    async def check_stock() -> str:
        bars = await gateway.fetch_underlying_bars(
            symbols=config.underlyings,
            start=date.today() - timedelta(days=7),
            end=date.today(),
        )
        frame = getattr(bars, "df", pd.DataFrame())
        if frame.empty:
            raise RuntimeError("sin barras del subyacente en los últimos 7 días")
        latest = frame.reset_index().sort_values("timestamp").groupby("symbol").last()
        state["spots"] = {str(k): float(v) for k, v in latest["close"].items()}
        precios = ", ".join(f"{k}={v:.2f}" for k, v in state["spots"].items())
        return f"{len(frame)} barras; último cierre: {precios}"

    results.append(await _probe("2. Precios del subyacente", check_stock))
    if not results[-1].passed:
        return _skip_rest(results, 4)

    async def check_universe() -> str:
        contracts = await gateway.fetch_option_contracts(
            underlyings=config.underlyings,
            expiration_gte=date.today(),
            expiration_lte=date.today() + timedelta(days=config.liquidity.max_dte),
        )
        if not contracts:
            raise RuntimeError(
                "universo vacío: revisar si la cuenta tiene opciones habilitadas"
            )
        state["contracts"] = contracts
        with_oi = sum(
            1 for c in contracts if getattr(c, "open_interest", None) not in (None, "")
        )
        return (
            f"{len(contracts)} contratos; "
            f"{with_oi} ({with_oi / len(contracts):.0%}) con open interest"
        )

    results.append(await _probe("3. Universo de contratos", check_universe))
    if not results[-1].passed:
        return _skip_rest(results, 3)

    async def check_snapshots() -> str:
        symbols = [c.symbol for c in state["contracts"][:100]]
        snapshots = await gateway.fetch_option_snapshots(symbols)
        if not snapshots:
            raise RuntimeError("el feed no devolvió ningún snapshot")

        spreads: list[float] = []
        with_quote = 0
        for snap in snapshots.values():
            quote = getattr(snap, "latest_quote", None)
            if quote is None:
                continue
            bid, ask = float(quote.bid_price or 0), float(quote.ask_price or 0)
            if bid > 0 and ask > 0 and ask >= bid:
                with_quote += 1
                spreads.append((ask - bid) / ((ask + bid) / 2))

        state["quote_coverage"] = with_quote / max(len(symbols), 1)
        state["median_spread"] = float(np.median(spreads)) if spreads else float("nan")
        if not spreads:
            raise RuntimeError("ningún contrato con NBBO utilizable (bid y ask > 0)")
        return (
            f"feed '{config.ingestion.feed}': {with_quote}/{len(symbols)} con NBBO "
            f"({state['quote_coverage']:.0%}); spread relativo mediano "
            f"{state['median_spread']:.1%}"
        )

    results.append(await _probe("4. NBBO en vivo (modo record)", check_snapshots))

    async def check_bars() -> str:
        symbols = [c.symbol for c in state["contracts"][:20]]
        bars = await gateway.fetch_option_bars(
            symbols, date.today() - timedelta(days=30), date.today()
        )
        frame = getattr(bars, "df", pd.DataFrame())
        if frame.empty:
            raise RuntimeError("sin barras históricas para estos contratos")
        return (
            f"{len(frame)} barras sobre {frame.reset_index()['symbol'].nunique()} "
            "contratos (sin bid/ask: el spread se estima)"
        )

    results.append(
        await _probe("5. Barras históricas (modo backfill)", check_bars, blocking=False)
    )

    def check_viability() -> CheckResult:
        spread = state.get("median_spread", float("nan"))
        threshold = config.liquidity.max_relative_spread
        if not np.isfinite(spread):
            return CheckResult(
                "6. Viabilidad de la estrategia", False, "sin datos de spread", False
            )
        passed = spread < threshold
        verdict = (
            f"spread mediano {spread:.1%} vs umbral {threshold:.0%}: "
            + ("hay margen para operar" if passed else "el spread se come el edge")
        )
        return CheckResult("6. Viabilidad de la estrategia", passed, verdict, False)

    results.append(check_viability())
    return results


def _skip_rest(results: list[CheckResult], remaining: int) -> list[CheckResult]:
    """Marca como omitidos los chequeos que no se ejecutaron."""
    names = [
        "2. Precios del subyacente",
        "3. Universo de contratos",
        "4. NBBO en vivo (modo record)",
        "5. Barras históricas (modo backfill)",
        "6. Viabilidad de la estrategia",
    ]
    for name in names[-remaining:]:
        results.append(CheckResult(name, False, "omitido: falló un paso anterior", False))
    return results


def format_report(results: list[CheckResult]) -> str:
    """Arma el reporte legible del diagnóstico.

    Args:
        results: Resultados de :func:`run_diagnostics`.

    Returns:
        El reporte listo para imprimir, con veredicto y sugerencia final.
    """
    lines = ["", "DIAGNÓSTICO DEL PIPELINE", "=" * 70]
    for result in results:
        mark = "OK  " if result.passed else "FALLA"
        lines.append(f"  [{mark}] {result.name}")
        lines.append(f"          {result.detail}")
    lines.append("=" * 70)

    blocking_failures = [r for r in results if not r.passed and r.blocking]
    if blocking_failures:
        lines.append(f"  Bloqueado en: {blocking_failures[0].name}")
        lines.append("  Resolver eso antes de seguir. El resto no se pudo evaluar.")
    else:
        soft = [r for r in results if not r.passed]
        lines.append("  Conectividad OK. El pipeline puede correr.")
        if soft:
            lines.append(
                f"  Advertencias ({len(soft)}): "
                + "; ".join(r.name for r in soft)
            )
        lines.append("")
        lines.append("  Siguiente paso:  python -m ita_options.pipeline record")
    lines.append("")
    return "\n".join(lines)
