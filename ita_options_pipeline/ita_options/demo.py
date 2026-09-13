"""Mercado sintético y corrida offline del pipeline completo.

Genera una cadena de opciones con una superficie de volatilidad realista y la
hace pasar por **el mismo código** que procesaría datos de Alpaca: alineación
contra el subyacente, cálculo de IV, filtros de liquidez, escritura y lectura de
parquet, detección de arbitrajes, backtest y evaluación.

Sirve para dos cosas:

1. **Verificar que todo funciona sin credenciales.** El único código que queda
   sin ejercitar es el que habla con la red.
2. **Ensayar la demo de la presentación.** El comando imprime exactamente lo que
   imprimiría con datos reales, así se puede practicar el guion antes de tener
   las claves.

Los datos son sintéticos y el reporte lo dice en cada corrida. No sirven como
evidencia empírica de nada: sirven para probar la cañería.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import (
    ASSET_PROFILES,
    RISK_FREE_RATE,
    AssetProfile,
    simulate_snapshot_path,
)
from .config import LiquidityThresholds, PricingAssumptions
from .volatility import crr_american_price

_LOG = logging.getLogger(__name__)

__all__ = ["generate_market", "run_offline_demo"]

#: Subyacentes del mercado sintético: los tres constituyentes del ITA calibrados
#: en ``calibration.py`` (RTX, BA, LMT). Se deriva de los perfiles para que exista
#: una única fuente de verdad de los parámetros de cada activo.
_UNDERLYINGS: dict[str, AssetProfile] = dict(ASSET_PROFILES)
_RATE = RISK_FREE_RATE


def _smile(
    log_moneyness: np.ndarray,
    atm_vol: float,
    skew: float,
    curv: float,
) -> np.ndarray:
    r"""Superficie de volatilidad con skew y sonrisa, específica por activo.

    .. math::

        \sigma(k) = \sigma_{\text{atm}}\big(1 - \beta_1\,k + \beta_2\,k^2\big)

    donde :math:`k = \ln(K/S)`. El término lineal negativo produce el skew
    característico de acciones: las puts OTM cotizan con volatilidad implícita
    más alta que los calls OTM, porque el mercado paga por protección a la baja.
    El término cuadrático genera la sonrisa.

    A diferencia de la versión previa —que imponía el mismo :math:`\beta_1,
    \beta_2` a todos los activos— acá el skew y la curvatura salen del perfil de
    cada subyacente: BA tiene el skew más pronunciado (riesgo de crack), RTX y
    LMT uno más suave.

    Args:
        log_moneyness: :math:`\ln(K/S)` por contrato.
        atm_vol: Volatilidad at-the-money (= vol total calibrada del activo).
        skew: Coeficiente :math:`\beta_1` del perfil.
        curv: Coeficiente :math:`\beta_2` del perfil.

    Returns:
        Volatilidad por contrato.
    """
    return atm_vol * (1.0 - skew * log_moneyness + curv * log_moneyness**2)


def generate_market(
    n_snapshots: int = 12,
    interval_minutes: int = 5,
    seed: int = 42,
    inject_dislocations: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera cotizaciones sintéticas y barras del subyacente.

    Cada subyacente sigue el proceso **calibrado** de ``calibration.py``: una
    difusión con saltos de Merton bajo la medida riesgo-neutral, cuyos parámetros
    (vol, dividendo, intensidad y tamaño de saltos, skew del smile) reproducen la
    distribución del activo real. La volatilidad total del proceso se fija igual a
    la IV at-the-money con la que se valúan las opciones, así la vol realizada del
    camino coincide con la implícita. Las opciones se valúan con el árbol CRR
    sobre el smile de cada activo, de modo que la cadena es internamente
    consistente: **no contiene arbitrajes** salvo los que se inyectan a propósito.

    Args:
        n_snapshots: Cantidad de cortes temporales.
        interval_minutes: Minutos entre cortes.
        seed: Semilla, para que la corrida sea reproducible.
        inject_dislocations: Cuántas dislocaciones plantar. Sirven para
            comprobar que los detectores las encuentran.

    Returns:
        Tupla ``(cotizaciones crudas, barras del subyacente)``.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-09-04 13:35", tz="UTC")
    stamps = [start + timedelta(minutes=interval_minutes * i) for i in range(n_snapshots)]
    expiries = [pd.Timestamp("2026-10-16"), pd.Timestamp("2026-11-20")]

    quote_rows: list[dict[str, object]] = []
    bar_rows: list[dict[str, object]] = []

    for ticker, profile in _UNDERLYINGS.items():
        # Camino calibrado por activo (jump-diffusion riesgo-neutral). La semilla
        # se deriva del nombre para que cada activo tenga su propio camino y la
        # corrida siga siendo reproducible.
        asset_rng = np.random.default_rng(
            seed + 1_000 * (sorted(_UNDERLYINGS).index(ticker) + 1)
        )
        path = simulate_snapshot_path(
            profile, n_snapshots, interval_minutes, asset_rng
        )

        for index, stamp in enumerate(stamps):
            current = float(path[index])
            # La barra se etiqueta con la apertura del intervalo: su cierre
            # recién se conoce un minuto después. El pipeline lo corrige.
            bar_rows.append(
                {"timestamp": stamp - timedelta(minutes=1), "symbol": ticker,
                 "close": current}
            )

            for expiry in expiries:
                tau = max((expiry - stamp.tz_localize(None)).days / 365.0, 1 / 365)
                strikes = np.round(
                    np.arange(current * 0.85, current * 1.16, current * 0.025) / 2.5
                ) * 2.5
                log_k = np.log(strikes / current)
                vols = _smile(
                    log_k, profile.annual_vol, profile.smile_skew, profile.smile_curv
                )

                for strike, vol in zip(strikes, vols, strict=True):
                    for flag in ("c", "p"):
                        fair = crr_american_price(
                            current, float(strike), tau, _RATE,
                            profile.dividend_yield, float(vol), flag, 96,
                        )
                        if fair < 0.02:
                            continue
                        # Spread proporcional al precio, con piso de un tick, y
                        # más ancho lejos del dinero, como en el mercado real.
                        rel = 0.03 + 0.25 * abs(float(log_k[list(strikes).index(strike)]))
                        half = max(fair * rel / 2, 0.01)
                        bid, ask = round(max(fair - half, 0.01), 2), round(fair + half, 2)
                        quote_rows.append(
                            {
                                "symbol": (
                                    f"{ticker}{expiry:%y%m%d}{flag.upper()}"
                                    f"{int(strike * 1000):08d}"
                                ),
                                "underlying": ticker,
                                "timestamp": stamp,
                                "observed_at": stamp,
                                "expiration": expiry,
                                "strike": float(strike),
                                "option_type": flag,
                                "style": "american",
                                "multiplier": 100.0,
                                "bid": bid,
                                "ask": ask,
                                "bid_size": float(rng.integers(1, 80)),
                                "ask_size": float(rng.integers(1, 80)),
                                "last_trade_price": (bid + ask) / 2,
                                "volume": float(rng.integers(0, 900)),
                                "open_interest": float(rng.integers(0, 4000)),
                                "open_interest_date": (
                                    stamp.tz_localize(None).normalize()
                                    - timedelta(days=1)
                                ),
                                "spread_is_proxy": False,
                                "feed": "sintetico",
                                "source": "demo",
                                "trade_date": stamp.tz_localize(None).normalize(),
                            }
                        )

    quotes = pd.DataFrame(quote_rows)

    # Dislocaciones plantadas: encarecemos el bid de un contrato líquido en
    # algunos cortes, rompiendo la convexidad de forma detectable.
    liquid = quotes.loc[
        (quotes["volume"] > 100) & (quotes["open_interest"] > 500)
        & (quotes["bid"] > 1.0)
    ]
    if inject_dislocations and not liquid.empty:
        targets = rng.choice(liquid.index.to_numpy(), size=inject_dislocations,
                             replace=False)
        quotes.loc[targets, ["bid", "ask"]] *= 1.45

    return quotes, pd.DataFrame(bar_rows)


def run_offline_demo(data_root: Path, verbose: bool = True) -> dict[str, object]:
    """Corre el pipeline completo sobre el mercado sintético.

    Ejercita, en orden: ``enrich``, ``filters``, ``schemas``, ``storage``,
    ``arbitrage``, ``backtest`` y ``evaluation``. Lo único que no toca es la
    capa de red.

    Args:
        data_root: Carpeta donde escribir el data lake de prueba.
        verbose: Imprimir el reporte por pantalla.

    Returns:
        Diccionario con los resultados de cada etapa.
    """
    from .arbitrage import ExecutionCosts, run_all_detectors
    from .backtest import ArbitrageBacktester, StrategyParams
    from .enrich import (
        align_underlying,
        compute_implied_volatility,
        compute_mid_and_spread,
        compute_time_to_expiry,
    )
    from .evaluation import compute_metrics
    from .filters import apply_liquidity_filters
    from .storage import ParquetStore

    lines: list[str] = []
    def say(text: str = "") -> None:
        lines.append(text)
        if verbose:
            print(text)

    say()
    say("=" * 72)
    say("  CORRIDA OFFLINE — DATOS SINTÉTICOS, NO SON DATOS DE MERCADO")
    say("=" * 72)

    quotes, underlying = generate_market()
    say(f"\n1. Mercado generado")
    say(f"   {len(quotes):,} cotizaciones | "
        f"{quotes['timestamp'].nunique()} cortes | "
        f"{quotes['symbol'].nunique():,} contratos | "
        f"{quotes['underlying'].nunique()} subyacentes")

    frame = compute_mid_and_spread(quotes)
    frame = align_underlying(frame, underlying, bar_duration=timedelta(minutes=1))
    matched = frame["underlying_price"].notna().mean()
    say(f"\n2. Alineación contra el subyacente (anti-lookahead)")
    say(f"   {matched:.1%} de las cotizaciones con spot vigente | "
        f"rezago mediano {frame['alignment_lag_s'].median():.0f}s")

    frame = compute_time_to_expiry(frame)
    # 64 pasos alcanzan para la demo: el objetivo es ejercitar la cañería,
    # no calibrar con precisión de producción.
    frame = compute_implied_volatility(
        frame, PricingAssumptions(binomial_steps=64)
    )
    iv = frame["iv_american"].dropna()
    say(f"\n3. Volatilidad implícita (árbol CRR americano)")
    say(f"   {len(iv):,}/{len(frame):,} invertidas | "
        f"mediana {iv.median():.1%} | rango {iv.min():.1%}–{iv.max():.1%}")
    say(f"   fuera de cotas de no arbitraje: {int(frame['violates_bounds'].sum())}")

    frame, report = apply_liquidity_filters(frame, LiquidityThresholds())
    say(f"\n4. Filtros de liquidez")
    say(f"   {report.surviving_rows:,}/{report.total_rows:,} explotables "
        f"({report.survival_rate:.1%})")
    for _, row in report.to_frame().head(4).iterrows():
        say(f"     {row['criterio']:<32} {int(row['filas_excluidas']):>6,} "
            f"({row['pct_del_total']:.1f}%)")

    # Carpeta limpia por corrida: el data lake es append-only, así que reusarla
    # mezclaría estas filas con las de corridas anteriores.
    import shutil

    demo_root = Path(data_root) / "demo_run"
    shutil.rmtree(demo_root, ignore_errors=True)
    store = ParquetStore(demo_root, "demo_quotes")
    written = store.write(frame)
    reread = store.read(underlyings=list(_UNDERLYINGS))
    say(f"\n5. Persistencia en parquet")
    say(f"   escritas {written:,} | releídas {len(reread):,} | "
        f"columnas {len(reread.columns)} | "
        f"is_tradable sobrevive: {'is_tradable' in reread.columns}")

    snapshot = frame.loc[frame["timestamp"] == frame["timestamp"].max()]
    found = run_all_detectors(snapshot, ExecutionCosts(min_net_edge=5.0))
    say(f"\n6. Detección de arbitrajes (último corte)")
    if found.empty:
        say("   sin oportunidades sobre el umbral")
    else:
        say(f"   {len(found)} oportunidades | por tipo: "
            + ", ".join(f"{k}={v}" for k, v in found['detector'].value_counts().items()))
        top = found.iloc[0]
        say(f"   mejor: {top['detector']} {top['underlying']} "
            f"strikes {top['strikes']} → {top['net_edge_usd']:,.0f} USD")

    engine = ArbitrageBacktester(frame, underlying, ExecutionCosts(min_net_edge=5.0))
    say(f"\n7. Backtest — sensibilidad a la latencia")
    say(f"   {'latencia':>10} {'operaciones':>12} {'P&L USD':>12} {'edge capt.':>11}")
    results = {}
    for minutes in (1, 5, 15):
        outcome = engine.run(
            StrategyParams(
                min_net_edge=5.0,
                execution_lag=timedelta(minutes=minutes),
                stop_loss_usd=None,
            )
        )
        metrics = compute_metrics(outcome)
        results[minutes] = metrics
        say(f"   {minutes:>8}min {metrics.n_trades:>12} "
            f"{metrics.total_pnl:>12,.0f} {metrics.edge_capture:>11.2f}")
        if minutes == 1 and not outcome.trades.empty:
            motivos = outcome.trades["exit_reason"].value_counts().to_dict()
            results["exit_reasons"] = motivos

    if results.get("exit_reasons"):
        say(f"\n   Motivo de cierre: "
            + ", ".join(f"{k}={v}" for k, v in results["exit_reasons"].items()))
        say("   El mercado sintético abarca una hora y los vencimientos son a")
        say("   meses, así que las posiciones se cierran por fin de datos, no al")
        say("   vencimiento: se paga el spread dos veces sin cobrar el payoff.")
        say("   Por eso el P&L es negativo. Con datos reales de varias semanas la")
        say("   posición llega a expiry y el resultado es otro.")

    say()
    say("=" * 72)
    say("  Todo el pipeline corrió salvo la capa de red (clients / ingest).")
    say("  Recordatorio: los datos son sintéticos. No son evidencia de nada.")
    say("=" * 72)
    say()

    return {
        "quotes": len(quotes),
        "filter_report": report,
        "opportunities": found,
        "backtest": results,
        "report": "\n".join(lines),
    }
