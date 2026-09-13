"""Orquestador del pipeline y CLI.

Uso::

    python -m ita_options.pipeline universe
    python -m ita_options.pipeline record --interval 300
    python -m ita_options.pipeline backfill --start 2025-09-01 --end 2026-09-01
    python -m ita_options.pipeline enrich --start 2026-09-01
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from .clients import AsyncAlpacaGateway
from .config import AlpacaCredentials, PipelineConfig
from .demo import run_offline_demo
from .doctor import format_report, run_diagnostics
from .enrich import (
    align_underlying,
    attach_lagged_open_interest,
    compute_implied_volatility,
    compute_mid_and_spread,
    compute_time_to_expiry,
)
from .arbitrage import ExecutionCosts, run_all_detectors
from .filters import FilterReport, apply_liquidity_filters
from .ingest import HistoricalBackfill, SnapshotRecorder, UniverseBuilder
from .storage import ParquetStore

_LOG = logging.getLogger(__name__)

__all__ = ["OptionsPipeline", "main"]


class OptionsPipeline:
    """Coordina universo, ingesta, enriquecimiento, filtrado y persistencia."""

    def __init__(self, config: PipelineConfig) -> None:
        """Inicializa el pipeline y sus dependencias."""
        self.config = config
        self.gateway = AsyncAlpacaGateway(config.credentials, config.ingestion)
        self.quotes = ParquetStore(config.data_root, "option_quotes")
        self.master = ParquetStore(config.data_root, "contract_master")
        self.curated = ParquetStore(config.data_root, "curated")

    async def _spot_prices(self) -> dict[str, float]:
        """Obtiene el último cierre diario de cada subyacente."""
        bars = await self.gateway.fetch_underlying_bars(
            symbols=self.config.underlyings,
            start=date.today() - timedelta(days=10),
            end=date.today(),
        )
        frame = getattr(bars, "df", pd.DataFrame())
        if frame.empty:
            return {}
        latest = frame.reset_index().sort_values("timestamp").groupby("symbol").last()
        return {str(k): float(v) for k, v in latest["close"].items()}

    async def refresh_universe(self) -> pd.DataFrame:
        """Descarga el universo del día y lo agrega al maestro append-only."""
        spots = await self._spot_prices()
        master = await UniverseBuilder(self.gateway, self.config).build(spots)
        if master.empty:
            _LOG.warning("Universo vacío.")
            return master
        master["trade_date"] = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
        path = self.master.path / f"universe_{date.today():%Y%m%d}.parquet"
        master.to_parquet(path, compression="zstd", index=False)
        _LOG.info("Maestro persistido: %d contratos en %s", len(master), path)
        return master

    async def live_chain(self) -> pd.DataFrame:
        """Arma la cadena curada del instante actual, lista para los detectores.

        Encadena las cuatro etapas del pipeline en una sola pasada: universo,
        snapshot del NBBO, enriquecimiento y filtros de liquidez. Es el camino
        que usa el subcomando ``detect`` y el que conviene mostrar en vivo,
        porque va de la API a las oportunidades sin pasos manuales.

        Returns:
            La cadena con ``is_tradable`` ya calculado.

        Raises:
            RuntimeError: si el universo o el snapshot vienen vacíos.
        """
        master = await self.refresh_universe()
        if master.empty:
            raise RuntimeError("Universo vacío: correr 'doctor' para diagnosticar.")

        symbols = master.loc[master["tradable"], "symbol"].tolist()
        recorder = SnapshotRecorder(self.gateway, self.config)
        quotes = await recorder.capture(symbols)
        if quotes.empty:
            raise RuntimeError("El feed no devolvió cotizaciones utilizables.")

        quotes = quotes.merge(
            master[
                ["symbol", "underlying", "expiration", "strike", "option_type",
                 "style", "multiplier"]
            ],
            on="symbol",
            how="left",
        )

        window_start = pd.to_datetime(quotes["timestamp"]).min() - timedelta(days=5)
        bars = await self.gateway.fetch_underlying_bars(
            symbols=self.config.underlyings,
            start=window_start,
            end=datetime.now(),
        )
        underlying = getattr(bars, "df", pd.DataFrame()).reset_index()

        frame = compute_mid_and_spread(quotes)
        frame = align_underlying(frame, underlying, tolerance=timedelta(minutes=30))
        frame = compute_time_to_expiry(frame, self.config.pricing.day_count)
        frame = attach_lagged_open_interest(frame, master)
        frame = compute_implied_volatility(frame, self.config.pricing)
        frame, report = apply_liquidity_filters(frame, self.config.liquidity)
        _LOG.info(
            "Cadena en vivo: %d contratos, %d explotables.",
            len(frame),
            report.surviving_rows,
        )
        return frame

    async def record(self, interval_seconds: int, iterations: int | None) -> None:
        """Corre el recorder de snapshots sobre el universo vigente."""
        master = await self.refresh_universe()
        if master.empty:
            return
        symbols = master.loc[master["tradable"], "symbol"].tolist()
        await SnapshotRecorder(self.gateway, self.config).run_forever(
            symbols=symbols,
            sink=self.quotes,
            interval_seconds=interval_seconds,
            max_iterations=iterations,
        )

    async def backfill(self, start: date, end: date) -> int:
        """Descarga bars históricos del universo vigente y los persiste."""
        master = await self.refresh_universe()
        if master.empty:
            return 0
        symbols = master["symbol"].tolist()
        frame = await HistoricalBackfill(self.gateway, self.config).fetch(
            symbols, start, end
        )
        if frame.empty:
            return 0
        frame = frame.merge(
            master[["symbol", "underlying", "expiration", "strike", "option_type",
                    "style", "multiplier"]],
            on="symbol",
            how="left",
        )
        return self.quotes.write(frame)

    async def enrich(
        self, start: str | None, end: str | None
    ) -> tuple[pd.DataFrame, FilterReport]:
        """Alinea, calcula IV, filtra y persiste el dataset curado.

        Args:
            start: Fecha mínima ``YYYY-MM-DD``.
            end: Fecha máxima ``YYYY-MM-DD``.

        Returns:
            Tupla ``(dataset curado, reporte de filtros)``.
        """
        raw = self.quotes.read(
            underlyings=self.config.underlyings, start=start, end=end
        )
        if raw.empty:
            raise RuntimeError("No hay quotes crudos en el rango solicitado.")

        window_start = pd.to_datetime(raw["timestamp"]).min() - timedelta(days=1)
        window_end = pd.to_datetime(raw["timestamp"]).max() + timedelta(days=1)
        bars = await self.gateway.fetch_underlying_bars(
            symbols=self.config.underlyings, start=window_start, end=window_end
        )
        underlying = getattr(bars, "df", pd.DataFrame()).reset_index()

        frame = compute_mid_and_spread(raw)
        frame = align_underlying(frame, underlying)
        frame = compute_time_to_expiry(frame, self.config.pricing.day_count)

        master_files = sorted(self.master.path.glob("universe_*.parquet"))
        if master_files:
            latest_master = pd.read_parquet(master_files[-1])
            frame = attach_lagged_open_interest(frame, latest_master)

        frame = compute_implied_volatility(frame, self.config.pricing)
        frame, report = apply_liquidity_filters(frame, self.config.liquidity, drop=False)

        self.curated.write(frame)
        self.curated.write_manifest(
            {
                **self.config.manifest(),
                "rows": len(frame),
                "tradable_rows": int(frame["is_tradable"].sum()),
                "generated_at": datetime.now().isoformat(),
            }
        )
        return frame, report


def _build_parser() -> argparse.ArgumentParser:
    """Construye el parser de la CLI."""
    parser = argparse.ArgumentParser(description="Pipeline de datos de opciones ITA.")
    parser.add_argument("--data-root", default="data", help="Raíz del data lake.")
    parser.add_argument(
        "--tickers", nargs="+", default=["RTX", "BA"], help="Subyacentes."
    )
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Diagnostica conectividad y datos disponibles.")
    sub.add_parser("demo", help="Corre el pipeline completo offline, sin credenciales.")
    sub.add_parser("universe", help="Refresca el maestro de contratos.")

    det = sub.add_parser("detect", help="Detecta arbitrajes sobre la cadena en vivo.")
    det.add_argument("--min-edge", type=float, default=5.0,
                     help="Ganancia neta mínima por contrato, en USD.")
    det.add_argument("--include-model", action="store_true",
                     help="Incluir el detector dependiente del binomial calibrado.")

    rec = sub.add_parser("record", help="Graba snapshots point-in-time.")
    rec.add_argument("--interval", type=int, default=300)
    rec.add_argument("--iterations", type=int, default=None)

    back = sub.add_parser("backfill", help="Descarga bars históricos.")
    back.add_argument("--start", required=True)
    back.add_argument("--end", required=True)

    enr = sub.add_parser("enrich", help="Alinea, calcula IV y filtra.")
    enr.add_argument("--start", default=None)
    enr.add_argument("--end", default=None)
    return parser


async def _run(args: argparse.Namespace) -> None:
    """Despacha el subcomando elegido."""

    credentials = (
        AlpacaCredentials(api_key="offline", secret_key="offline")
        if args.command == "demo"
        else AlpacaCredentials.from_env()
    )
    config = PipelineConfig(
        credentials=credentials,
        underlyings=tuple(args.tickers),
        data_root=Path(args.data_root),
    )
    pipeline = OptionsPipeline(config)

    if args.command == "demo":
        run_offline_demo(Path(args.data_root))
        return
    if args.command == "doctor":
        print(format_report(await run_diagnostics(config)))
    elif args.command == "detect":
        chain = await pipeline.live_chain()
        found = run_all_detectors(
            chain,
            costs=ExecutionCosts(min_net_edge=args.min_edge),
            include_model=args.include_model,
        )
        if found.empty:
            print("Sin arbitrajes por encima del umbral. "
                  "La cadena es consistente, que es lo normal.")
        else:
            columns = ["detector", "underlying", "strikes", "legs", "net_edge_usd",
                       "min_volume", "max_spread_rel"]
            print(found[columns].to_string(index=False))
    elif args.command == "universe":
        await pipeline.refresh_universe()
    elif args.command == "record":
        await pipeline.record(args.interval, args.iterations)
    elif args.command == "backfill":
        rows = await pipeline.backfill(
            date.fromisoformat(args.start), date.fromisoformat(args.end)
        )
        _LOG.info("Backfill: %d filas.", rows)
    elif args.command == "enrich":
        _, report = await pipeline.enrich(args.start, args.end)
        print(report.to_frame().to_string(index=False))
        print(f"\nTasa de supervivencia: {report.survival_rate:.2%}")


def main() -> None:
    """Punto de entrada de la CLI."""
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_run(args))
    except RuntimeError as exc:
        # Falta de credenciales o de datos: es un problema de configuración del
        # usuario, no un bug. Un traceback acá sólo esconde el mensaje útil.
        raise SystemExit(f"\n{exc}\n")


if __name__ == "__main__":
    main()
