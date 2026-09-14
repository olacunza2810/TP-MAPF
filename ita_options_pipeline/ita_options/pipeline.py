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
import json
import logging
import sys
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from .clients import AsyncAlpacaGateway
from .config import (
    AlpacaCredentials,
    DailyBarAssumptions,
    LiquidityThresholds,
    PipelineConfig,
)
from .daily import daily_strategy_params, run_backtest_daily, run_enrich_daily
from .demo import run_offline_demo
from .evaluation import compute_metrics, signal_funnel, trade_breakdown
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

__all__ = ["OptionsPipeline", "main", "parse_args"]

_COMMANDS = (
    "doctor", "demo", "universe", "detect", "record", "backfill", "enrich",
    "enrich-daily", "backtest-daily", "paper-run",
)


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

    end_ = sub.add_parser(
        "enrich-daily",
        help="Cura las velas diarias de Polygon (bid/ask sintético, sin red).",
    )
    end_.add_argument("--start", default=None)
    end_.add_argument("--end", default=None)
    end_.add_argument("--price-source", choices=["close", "vwap"], default="close")
    end_.add_argument("--assumed-spread", type=float, default=0.05,
                      help="Spread relativo supuesto para construir bid y ask.")
    end_.add_argument("--min-prev-volume", type=float, default=0.0,
                      help="Volumen mínimo (exclusivo) de la rueda anterior.")

    btd = sub.add_parser("backtest-daily", help="Backtest sobre el dataset curado diario.")
    btd.add_argument("--start", default=None)
    btd.add_argument("--end", default=None)
    btd.add_argument("--lag-sessions", type=int, nargs="+", default=[1],
                     help="Ruedas entre señal y ejecución; varios valores comparan.")
    btd.add_argument("--min-edge", type=float, default=5.0)
    btd.add_argument("--stop-loss", type=float, default=500.0,
                     help="Pérdida no realizada que cierra la posición; 0 lo desactiva.")
    btd.add_argument("--execute-at", choices=["open", "close"], default="open",
                     help="Llenar la orden en la apertura (default) o al cierre de la rueda.")
    btd.add_argument("--no-edge-check", dest="edge_check", action="store_false",
                     help="Ejecutar aunque el edge haya desaparecido (comportamiento previo).")
    btd.add_argument("--report-dir", default=None,
                     help="Carpeta donde guardar resumen, operaciones y desgloses en CSV.")

    paper = sub.add_parser(
        "paper-run",
        help="Loop intradía de paper trading en Alpaca (dry-run salvo --live).",
    )
    paper.add_argument("--live", action="store_true",
                       help="Enviar órdenes a la cuenta paper. Sin esto sólo registra el plan.")
    paper.add_argument("--max-cycles", type=int, default=None)
    paper.add_argument("--cycle-seconds", type=float, default=90.0)
    paper.add_argument("--contracts", type=int, default=1)
    paper.add_argument("--max-open", type=int, default=5)
    paper.add_argument("--kill-switch", type=float, default=2000.0)
    paper.add_argument("--daily-loss", type=float, default=3000.0)
    paper.add_argument("--kill-file", default="KILL",
                       help="Si este archivo existe, se detienen entradas y se cierra todo.")
    paper.add_argument("--ledger", default=None,
                       help="Ledger SQLite (por defecto <data-root>/paper/ledger.sqlite).")
    paper.add_argument("--report-dir", default="reportes/paper")
    paper.add_argument("--detectors", nargs="+",
                       default=["monotonicity", "vertical_bound", "butterfly", "put_call_parity"])

    # ``--tickers`` también dentro de cada subcomando. SUPPRESS evita que el
    # default del subcomando pise el valor pasado antes del subcomando.
    for command in sub.choices.values():
        command.add_argument("--tickers", nargs="+", default=argparse.SUPPRESS,
                             help="Subyacentes (también separados por comas).")
    return parser


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    """Evita que ``--tickers`` se coma el nombre del subcomando.

    ``--tickers`` acepta varios valores, así que argparse le asigna todo lo que
    sigue, incluido ``enrich-daily``, y después interpreta mal el resto. Acá se
    juntan sus valores en un único ``--tickers=RTX,BA,LMT`` que termina en la
    primera opción o subcomando. El separador ``--`` antes del subcomando, que
    era la forma de esquivar el problema, se sigue aceptando.
    """
    tokens = list(argv)
    out: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--tickers":
            values = []
            index += 1
            while (index < len(tokens) and not tokens[index].startswith("-")
                   and tokens[index] not in _COMMANDS):
                values.append(tokens[index])
                index += 1
            out.append(f"--tickers={','.join(values)}" if values else token)
            continue
        if token == "--" and index + 1 < len(tokens) and tokens[index + 1] in _COMMANDS:
            index += 1
            continue
        out.append(token)
        index += 1
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parsea la línea de comandos con ``--tickers`` antes o después del subcomando."""
    raw = sys.argv[1:] if argv is None else argv
    args = _build_parser().parse_args(_normalize_argv(raw))
    args.tickers = [
        ticker.strip().upper()
        for value in args.tickers
        for ticker in str(value).split(",")
        if ticker.strip()
    ]
    return args


def _latest_daily_manifest(data_root: Path) -> dict[str, object]:
    """Supuestos de la última curación diaria (fuente de precio, spread supuesto)."""
    manifests = sorted(Path(data_root).glob("manifest_daily_*.json"))
    return json.loads(manifests[-1].read_text("utf-8")) if manifests else {}


def _print_table(title: str, table: "pd.DataFrame") -> None:
    print(f"\n{title}")
    if table.empty:
        print("  (sin filas)")
        return
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


def _run_daily(args: argparse.Namespace) -> None:
    """Subcomandos del modo diario: trabajan sólo con archivos locales."""
    data_root = Path(args.data_root)
    tickers = list(args.tickers)
    if args.command == "enrich-daily":
        frame, report = run_enrich_daily(
            data_root, tickers, args.start, args.end,
            liquidity=LiquidityThresholds.for_daily_bars(
                min_prev_day_volume=args.min_prev_volume
            ),
            assumptions=DailyBarAssumptions(
                price_source=args.price_source,
                assumed_relative_spread=args.assumed_spread,
            ),
        )
        print(report.to_frame().to_string(index=False))
        print(f"\nTasa de supervivencia: {report.survival_rate:.2%}")
        iv = frame["iv_american"].dropna()
        if not iv.empty:
            print(f"IV americana: mediana {iv.median():.1%} sobre {len(iv):,} filas "
                  f"(bid/ask sintético, spread supuesto {args.assumed_spread:.0%})")
        return

    assumptions = _latest_daily_manifest(data_root).get("daily_assumptions", {})
    price_source = assumptions.get("price_source", "?")
    spread = assumptions.get("assumed_relative_spread", float("nan"))
    stop_loss = args.stop_loss or None
    report_dir = Path(args.report_dir) if args.report_dir else None
    if report_dir:
        report_dir.mkdir(parents=True, exist_ok=True)

    print("\nBACKTEST DIARIO — bid/ask sintético, ver advertencias en daily.py")
    print(f"fuente {price_source} | spread supuesto {spread:.1%} | min-edge "
          f"{args.min_edge:g} USD | stop-loss {stop_loss or 'desactivado'} | "
          f"edge al ejecutar: {'sí' if args.edge_check else 'no'} | "
          f"ejecución al {'open' if args.execute_at == 'open' else 'cierre'}")

    summary_rows = []
    for lag in args.lag_sessions:
        result = run_backtest_daily(
            data_root, tickers,
            daily_strategy_params(
                lag, min_net_edge=args.min_edge, stop_loss_usd=stop_loss,
                require_edge_at_execution=args.edge_check,
                execution_price="open" if args.execute_at == "open" else "quote",
            ),
            ExecutionCosts(min_net_edge=args.min_edge),
            args.start, args.end,
        )
        metrics = compute_metrics(result)
        funnel = signal_funnel(result)
        by_detector = trade_breakdown(result, "detector")
        by_exit = trade_breakdown(result, "exit_reason")

        print(f"\n=== Latencia {lag} rueda(s) ===")
        print(f"operaciones {metrics.n_trades} | P&L {metrics.total_pnl:,.0f} USD | "
              f"hit rate {metrics.hit_rate:.2f} | edge capturado {metrics.edge_capture:.2f} | "
              f"Sharpe {metrics.sharpe:.2f} | max drawdown {metrics.max_drawdown:,.0f} USD")
        _print_table("Embudo de señales", funnel)
        _print_table("P&L por detector", by_detector)
        _print_table("P&L por motivo de salida", by_exit)

        diagnostics = result.diagnostics
        summary_rows.append({
            "price_source": price_source, "assumed_spread": spread,
            "start": args.start, "end": args.end, "lag_sessions": lag,
            "min_edge": args.min_edge, "stop_loss": stop_loss,
            "edge_check": args.edge_check, "execute_at": args.execute_at,
            "n_trades": metrics.n_trades,
            "total_pnl": metrics.total_pnl, "hit_rate": metrics.hit_rate,
            "edge_capture": metrics.edge_capture, "sharpe": metrics.sharpe,
            "max_drawdown": metrics.max_drawdown,
            **{k: diagnostics.get(k, 0) for k in (
                "signals", "queued", "filled", "rejected_edge_gone",
                "rejected_no_quote", "rejected_capacity", "pending_at_end")},
        })
        if report_dir:
            tag = (f"{price_source}_spread{spread:g}_lag{lag}_sl{stop_loss or 0:g}_"
                   f"edge{'on' if args.edge_check else 'off'}_{args.execute_at}")
            result.trades.to_csv(report_dir / f"trades_{tag}.csv", index=False)
            funnel.to_csv(report_dir / f"funnel_{tag}.csv", index=False)
            by_detector.to_csv(report_dir / f"by_detector_{tag}.csv", index=False)
            by_exit.to_csv(report_dir / f"by_exit_{tag}.csv", index=False)

    if report_dir:
        # Se acumula entre corridas: una grilla sobre el spread supuesto requiere
        # re-curar entre backtests y todas las filas terminan en el mismo resumen.
        summary_path = report_dir / "summary.csv"
        pd.DataFrame(summary_rows).to_csv(
            summary_path, mode="a", header=not summary_path.exists(), index=False
        )
        print(f"\nReportes en {report_dir.resolve()}")


def _run_paper(args: argparse.Namespace) -> None:
    """Loop de paper trading. Rechaza credenciales de cuenta real."""
    from alpaca.trading.client import TradingClient

    from .config import ExecutionSettings
    from .execution.broker import AlpacaBroker
    from .execution.ledger import Ledger
    from .execution.live_data import AlpacaMarketData
    from .execution.live_runner import LiveRunner

    credentials = AlpacaCredentials.from_env()
    if not credentials.paper:
        raise RuntimeError("paper-run sólo opera la cuenta paper: ALPACA_PAPER=false detectado.")
    data_root = Path(args.data_root)
    settings = ExecutionSettings(
        contracts=args.contracts, max_open_strategies=args.max_open,
        kill_switch_usd=args.kill_switch, daily_loss_limit_usd=args.daily_loss,
        cycle_seconds=args.cycle_seconds, kill_file=Path(args.kill_file),
        detectors=tuple(args.detectors),
    )
    config = PipelineConfig(credentials=credentials, underlyings=tuple(args.tickers),
                            data_root=data_root)
    dividends_path = data_root / "polygon" / "dividends.parquet"
    dividends = pd.read_parquet(dividends_path) if dividends_path.exists() else None
    ledger = Ledger(Path(args.ledger) if args.ledger else data_root / "paper" / "ledger.sqlite")
    runner = LiveRunner(
        AlpacaBroker(TradingClient(credentials.api_key, credentials.secret_key, paper=True)),
        AlpacaMarketData(config, settings), ledger, settings,
        dividends=dividends, dry_run=not args.live, report_dir=Path(args.report_dir),
    )
    mode = "LIVE (paper)" if args.live else "dry-run"
    print(f"paper-run {mode}: {', '.join(args.tickers)} | detectores {', '.join(args.detectors)}"
          f" | kill-switch {args.kill_switch:g} USD | corte manual: crear '{args.kill_file}'")
    report = runner.run(max_cycles=args.max_cycles)
    print(f"Reporte de sesión: {report}")


async def _run(args: argparse.Namespace) -> None:
    """Despacha el subcomando elegido."""

    if args.command in {"enrich-daily", "backtest-daily"}:
        _run_daily(args)
        return
    if args.command == "paper-run":
        _run_paper(args)
        return

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
    args = parse_args()
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
