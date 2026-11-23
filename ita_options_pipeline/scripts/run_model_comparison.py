r"""Paso 3 de la Parte 2: Black-Scholes contra el binomial sobre DoltHub.

Para cada rueda de marzo a noviembre de 2026, cada subyacente y cada
vencimiento de 7 a 120 días:

1. cada modelo invierte la IV de las OTM con su propio precio (dividendos
   discretos escrowed, proyectados point-in-time) y ajusta su sonrisa;
2. con esa sonrisa valúa **todos** los contratos operables, ITM incluidos
   (las ITM no entran al ajuste);
3. se mide la prima de ejercicio anticipado a la misma :math:`\sigma`, el
   error de cada modelo contra el mid y el efecto de tratar el dividendo como
   discreto en vez de yield continuo.

Salidas en ``reportes_dolthub/``:

- ``comparacion_contratos.parquet``: una fila por contrato y rueda;
- ``comparacion_eep_por_grupo.csv``: prima y errores por tipo, dinero, plazo y fecha ex;
- ``comparacion_dm.csv``: Diebold-Mariano por grupo, con las dos pérdidas;
- ``comparacion_resumen.json``: los números que cita el informe;
- ``comparacion_eep.png``: la prima en medios spreads por grupo.

Uso::

    python scripts/run_model_comparison.py            # todas las ruedas
    python scripts/run_model_comparison.py --weekly   # una foto por semana (rápido)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ita_options.dolthub import merge_dividends  # noqa: E402
from ita_options.model_comparison import (  # noqa: E402
    ComparisonSettings, add_buckets, bucket_summary, contract_dividends, dm_table, value_slice,
)

_LOG = logging.getLogger("comparacion")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#ecebe6"

_DIVIDENDS: pd.DataFrame | None = None
_SPOT = "implied_spot"


def _init(dividends: pd.DataFrame, spot_column: str) -> None:
    global _DIVIDENDS, _SPOT
    _DIVIDENDS = dividends
    _SPOT = spot_column


def _value_day(day_chain: pd.DataFrame) -> pd.DataFrame:
    settings = ComparisonSettings()
    out = []
    for (und, exp), slice_ in day_chain.groupby(["underlying", "expiration"]):
        divs = contract_dividends(_DIVIDENDS, und, slice_["trade_date"].iloc[0], exp)
        frame = value_slice(slice_, divs, settings, spot_column=_SPOT)
        if not frame.empty:
            out.append(frame)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def load(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    curated = pd.read_parquet(data_root / "dolthub" / "curated.parquet")
    curated["trade_date"] = pd.to_datetime(curated["trade_date"]).dt.date
    curated["expiration"] = pd.to_datetime(curated["expiration"]).dt.date
    settings = ComparisonSettings()
    base = curated.loc[curated["is_tradable"] & curated["dte"].between(settings.min_dte, settings.max_dte)]
    dividends = merge_dividends(pd.read_parquet(data_root / "dolthub" / "dividends.parquet"),
                                data_root / "dolthub" / "dividends_polygon.parquet")
    return base, dividends


def run(base: pd.DataFrame, dividends: pd.DataFrame, workers: int, spot_column: str) -> pd.DataFrame:
    days = [g for _, g in base.groupby("trade_date")]
    start = time.time()
    frames = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(dividends, spot_column)) as pool:
        for i, frame in enumerate(pool.map(_value_day, days, chunksize=2), 1):
            frames.append(frame)
            if i % 20 == 0:
                _LOG.info("%d/%d ruedas (%.0fs)", i, len(days), time.time() - start)
    return add_buckets(pd.concat(frames, ignore_index=True))


def plot(table: pd.DataFrame, out: Path) -> None:
    """Prima de ejercicio en medios spreads, por dinero y fecha ex, calls y puts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order_m = ["OTM", "ATM", "ITM", "ITM profundo"]
    order_ex = ["sin fecha ex", "ex en >30 días", "ex en 8-30 días", "ex en <=7 días"]
    colors = dict(zip(order_ex, ["#d4d3cd", "#86b6ef", "#2a78d6", "#104281"]))
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, facecolor=SURFACE)
    width = 0.2
    for ax, flag, title in zip(axes, ("c", "p"), ("Calls", "Puts")):
        ax.set_facecolor(SURFACE)
        sub = table.loc[table["option_type"] == flag]
        for j, ex in enumerate(order_ex):
            part = sub.loc[sub["fecha_ex"] == ex].set_index("dinero").reindex(order_m)
            x = np.arange(len(order_m)) + (j - 1.5) * (width + 0.02)
            ax.bar(x, part["eep_media_medios_spreads"].fillna(0), width=width, color=colors[ex], label=ex)
        ax.axhline(1.0, color=INK2, lw=1, ls="--", zorder=0)
        ax.text(len(order_m) - 0.5, 1.02, "1 medio spread", ha="right", va="bottom", color=INK2, fontsize=8)
        ax.set_xticks(np.arange(len(order_m)), order_m)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Prima de ejercicio anticipado\n(media, en medios spreads)")
    axes[1].legend(frameon=False, fontsize=8, title="Fecha ex en la vida", title_fontsize=8, loc="upper left")
    fig.suptitle("Dónde se separan el binomial americano y Black-Scholes europeo — RTX, BA, LMT, mar-nov 2026",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "comparacion_eep.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(ROOT / "reportes_dolthub"))
    parser.add_argument("--weekly", action="store_true", help="Una foto por semana en lugar de todas las ruedas.")
    parser.add_argument("--max-days", type=int, default=None, help="Sólo las primeras N ruedas (prueba).")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--spot", choices=["implied_spot", "underlying_price"], default="implied_spot",
                        help="Spot para valuar: implícito por paridad (principal) o cierre de la acción (robustez).")
    parser.add_argument("--tables-only", action="store_true", help="Rehace tablas y gráfico con el parquet guardado.")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.spot == "implied_spot" else "_spot_cierre"
    contracts_path = out / f"comparacion_contratos{suffix}.parquet"
    if args.tables_only:
        frame = pd.read_parquet(contracts_path)
    else:
        base, dividends = load(Path(args.data_root))
        days = sorted(base["trade_date"].unique())
        if args.weekly:
            weeks = pd.Series(days).map(lambda d: d.isocalendar()[:2])
            days = pd.Series(days).groupby(weeks).max().tolist()
        if args.max_days:
            days = days[: args.max_days]
        frame = run(base.loc[base["trade_date"].isin(days)], dividends, args.workers, args.spot)
        frame.to_parquet(contracts_path, index=False)

    groups = ["option_type", "dinero", "fecha_ex"]
    table = bucket_summary(frame, groups)
    table.to_csv(out / f"comparacion_eep_por_grupo{suffix}.csv", index=False)
    plazo = bucket_summary(frame, ["option_type", "dinero", "plazo"])
    plazo.to_csv(out / f"comparacion_eep_por_plazo{suffix}.csv", index=False)
    by_und = bucket_summary(frame, ["underlying", "option_type"])
    by_und.to_csv(out / f"comparacion_por_subyacente{suffix}.csv", index=False)

    # Dos preguntas: ¿importa el ejercicio anticipado? (BS vs binomial) y
    # ¿importa tratar el dividendo como discreto? (árbol con yield continuo vs escrowed).
    pairs = (("bsm", "crr"), ("crr_q", "crr"))
    dm = pd.concat([
        dm_table(frame, loss=loss, pair=pair).assign(grupo="total")
        for pair in pairs for loss in ("L", "L2")
    ] + [
        dm_table(frame, by=by, loss=loss, pair=pair).assign(grupo="+".join(by))
        for pair in pairs
        for by in (["en_ajuste"], ["option_type", "dinero"], ["option_type", "fecha_ex"], ["underlying"])
        for loss in ("L", "L2")
    ], ignore_index=True)
    dm.to_csv(out / f"comparacion_dm{suffix}.csv", index=False)

    itm = frame.loc[frame["dinero"].isin(["ITM", "ITM profundo"])]
    summary = {
        "spot": args.spot,
        "ruedas": int(frame["trade_date"].nunique()),
        "contratos": int(len(frame)),
        "contratos_itm": int(len(itm)),
        "eep_media_usd": float(frame["eep"].mean()),
        "pct_eep_mayor_medio_spread": float((frame["eep_en_medios_spreads"] > 1).mean()),
        "pct_eep_mayor_medio_spread_itm": float((itm["eep_en_medios_spreads"] > 1).mean()),
        "dentro_crr": float(frame["dentro_crr"].mean()),
        "dentro_bsm": float(frame["dentro_bsm"].mean()),
        "dentro_crr_itm": float(itm["dentro_crr"].mean()),
        "dentro_bsm_itm": float(itm["dentro_bsm"].mean()),
        "dentro_crr_q": float(frame["dentro_crr_q"].mean()),
        "dentro_crr_q_itm": float(itm["dentro_crr_q"].mean()),
        "pct_bsm_bajo_intrinseco_itm": float((itm["v_bsm"] < itm["intrinseco"] - 1e-9).mean()),
        "efecto_div_abs_medio_usd": float(frame["efecto_div"].abs().mean()),
        "dm_total": dm.loc[dm["grupo"] == "total"].to_dict(orient="records"),
    }
    # Universo de la estrategia de valor relativo: |k| <= 10%, 21 a 75 días.
    uni = frame.loc[frame["k"].abs().le(0.10) & frame["dte"].between(21, 75)]
    gap = (uni["v_crr"] - uni["v_bsm"]).abs()
    summary["universo_estrategia"] = {
        "contratos": int(len(uni)),
        "brecha_crr_bsm_mediana_usd_contrato": float(gap.median() * 100),
        "brecha_crr_bsm_p90_usd_contrato": float(gap.quantile(0.9) * 100),
        "pct_brecha_crr_bsm_mayor_10usd": float((gap > 0.10).mean()),
        "pct_brecha_crr_crrq_mayor_10usd": float(((uni["v_crr"] - uni["v_crr_q"]).abs() > 0.10).mean()),
        "pct_eep_mayor_medio_spread": float((uni["eep"] > uni["half"]).mean()),
        "pct_bsm_bajo_intrinseco": float((uni["v_bsm"] < uni["intrinseco"] - 1e-9).mean()),
        "sesgo_usd_contrato": {
            f"{t}_{d}": {m: float(g[f"err_{m}"].mean() * 100) for m in ("crr", "bsm", "crr_q")}
            for (t, d), g in uni.groupby(["option_type", "dinero"])
        },
    }
    (out / f"comparacion_resumen{suffix}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                                                 encoding="utf-8")
    if not suffix:
        plot(table, out)
    _LOG.info("\n%s", json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
