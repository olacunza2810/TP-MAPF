r"""Calibración del binomial a la superficie observada y error de repricing (punto 5).

La devolución pidió el cuadro de IV contra moneyness y vencimiento y el RMSE de
repricing. Con una IV invertida contrato por contrato el repricing da error cero
por construcción, así que el error que importa es el de una **superficie
paramétrica** ajustada y vuelta a valuar con el árbol:

.. math::

    \sigma(k) = a + b\,k + c\,k^2,\qquad k = \ln(K/F)

ajustada por vencimiento, por mínimos cuadrados ponderados por vega y por la
inversa del spread, sobre las opciones **fuera del dinero** (puts con
:math:`K<F`, calls con :math:`K\ge F`): son las líquidas y las que no cargan
prima de ejercicio anticipado.

Muestra: una foto por semana (el viernes, o la última rueda de la semana) de
marzo a noviembre de 2026.

Errores reportados, por subyacente y modelo:

- **in-sample:** se ajusta con la foto de :math:`t` y se valúa la misma foto;
- **out-of-sample:** con los parámetros de :math:`t` (sonrisa fija en
  moneyness, forward actualizado) se valúa la foto de la rueda siguiente;
- contra dos referencias: una **sigma constante** por subyacente y vencimiento
  (la IV ATM) y el **modelo europeo** (BSM) con el mismo ajuste.

Cada error se informa en USD, normalizado por el medio spread
(:math:`|\hat V - m|/(s/2)`, menor que 1 significa dentro del bid-ask) y como
fracción de precios que caen dentro del bid-ask.

Uso::

    python scripts/run_calibration.py --out reportes_dolthub
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ita_options.execution.model_rv import RVSettings, fit_smile  # noqa: E402
from ita_options.rv_backtest import _effective_yield  # noqa: E402
from ita_options.volatility import bsm_price, crr_american_price, implied_volatility  # noqa: E402

_LOG = logging.getLogger("calibration")
SURFACE = "#fcfcfb"
INK, INK2 = "#0b0b0b", "#52514e"
DTE_COLORS = {"7-30 días": "#86b6ef", "31-60 días": "#2a78d6", "61-120 días": "#104281"}


def weekly_days(days: list) -> list:
    frame = pd.Series(sorted(days))
    weeks = frame.map(lambda d: (d.isocalendar().year, d.isocalendar().week))
    return frame.groupby(weeks).max().tolist()


def _otm(slice_: pd.DataFrame, forward: float) -> pd.DataFrame:
    return slice_.loc[((slice_["option_type"] == "p") & (slice_["strike"] < forward))
                      | ((slice_["option_type"] == "c") & (slice_["strike"] >= forward))]


def _price(model: str, r, sigma: float, steps: int = 100) -> float:
    if model == "crr":
        return crr_american_price(r.underlying_price, r.strike, r.tau, r.risk_free_rate, r.q_eff, sigma, r.option_type, steps=steps)
    return bsm_price(r.underlying_price, r.strike, r.tau, r.risk_free_rate, r.q_eff, sigma, r.option_type)


def calibrate(curated: pd.DataFrame, models=("crr", "bsm")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ajusta y valúa. Devuelve ``(errores_por_contrato, ivs_observadas)``."""
    base = curated.loc[curated["is_tradable"] & curated["dte"].between(7, 120)].copy()
    base["q_eff"] = _effective_yield(base)
    base["dividend_yield"] = base["q_eff"]
    days = sorted(base["trade_date"].unique())
    sample = weekly_days(days)
    next_day = {d: days[i + 1] for i, d in enumerate(days[:-1])}
    errors, ivs = [], []
    for day in sample:
        today = base.loc[base["trade_date"] == day]
        tomorrow = base.loc[base["trade_date"] == next_day.get(day)]
        for (und, exp), slice_ in today.groupby(["underlying", "expiration"]):
            for model in models:
                settings = RVSettings(model=model, steps=100, min_fit_points=5)
                smile = fit_smile(slice_, model, settings)
                if smile is None:
                    continue
                otm = _otm(slice_, smile.forward)
                atm_sigma = smile.sigma(smile.forward)
                for r in otm.itertuples(index=False):
                    mid = (r.bid + r.ask) / 2
                    k = np.log(r.strike / smile.forward)
                    if model == "crr":
                        iv = implied_volatility(mid, r.underlying_price, r.strike, r.tau, r.risk_free_rate,
                                                r.q_eff, r.option_type, american=True, steps=100)
                        ivs.append({"trade_date": day, "underlying": und, "expiration": exp, "dte": r.dte,
                                    "k": k, "iv_crr": iv, "iv_proveedor": r.provider_iv,
                                    "option_type": r.option_type, "sigma_modelo": smile.sigma(r.strike)})
                    for label, sigma in (("superficie", smile.sigma(r.strike)), ("sigma_constante", atm_sigma)):
                        errors.append({"muestra": "in_sample", "modelo": model, "ajuste": label,
                                       "trade_date": day, "underlying": und, "dte": r.dte, "k": k,
                                       "mid": mid, "half": (r.ask - r.bid) / 2, "bid": r.bid, "ask": r.ask,
                                       "valor": _price(model, r, sigma)})
                # Fuera de muestra: la misma sonrisa en moneyness sobre la foto siguiente.
                nxt = tomorrow.loc[tomorrow["expiration"] == exp]
                nxt = nxt.loc[nxt["underlying"] == und]
                if nxt.empty:
                    continue
                row1 = nxt.iloc[0]
                fwd1 = row1["underlying_price"] * np.exp((row1["risk_free_rate"] - row1["q_eff"]) * row1["tau"])
                for r in _otm(nxt, fwd1).itertuples(index=False):
                    k = np.log(r.strike / fwd1)
                    shifted = float(np.clip(smile.a + smile.b * k + smile.c * k * k, 0.03, 3.0))
                    for label, sigma in (("superficie", shifted), ("sigma_constante", atm_sigma)):
                        mid = (r.bid + r.ask) / 2
                        errors.append({"muestra": "out_of_sample", "modelo": model, "ajuste": label,
                                       "trade_date": r.trade_date, "underlying": und, "dte": r.dte, "k": k,
                                       "mid": mid, "half": (r.ask - r.bid) / 2, "bid": r.bid, "ask": r.ask,
                                       "valor": _price(model, r, sigma)})
        _LOG.info("Calibrado %s", day)
    err = pd.DataFrame(errors)
    err["error"] = err["valor"] - err["mid"]
    err["error_en_medios_spreads"] = err["error"].abs() / err["half"].clip(lower=0.005)
    err["dentro_bid_ask"] = (err["valor"] >= err["bid"]) & (err["valor"] <= err["ask"])
    return err, pd.DataFrame(ivs)


def summarize(err: pd.DataFrame) -> pd.DataFrame:
    return (err.groupby(["muestra", "modelo", "ajuste", "underlying"])
            .agg(contratos=("error", "size"),
                 rmse_usd=("error", lambda e: float(np.sqrt(np.mean(e**2)))),
                 mae_usd=("error", lambda e: float(np.mean(np.abs(e)))),
                 error_mediano_en_medios_spreads=("error_en_medios_spreads", "median"),
                 pct_dentro_bid_ask=("dentro_bid_ask", "mean"))
            .reset_index())


def plot(ivs: pd.DataFrame, out: Path) -> None:
    """Gráficos de la sonrisa (forma relativa al ATM) y de la estructura temporal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ivs = ivs.loc[ivs["iv_crr"].between(0.05, 1.5)].copy()
    ivs["tramo"] = pd.cut(ivs["dte"], [6, 30, 60, 120], labels=list(DTE_COLORS))
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})

    # 1. Forma de la sonrisa: IV menos la IV ATM de su propia foto y vencimiento.
    # Sumar 38 fotos en niveles mezcla la forma con los cambios de nivel semana a
    # semana; restando el ATM de cada sonrisa queda sólo la forma (el skew).
    def _atm(group: pd.DataFrame) -> float:
        if len(group) < 3:
            return np.nan
        return float(np.polyval(np.polyfit(group["k"], group["iv_crr"], 2), 0.0))

    atm_level = ivs.groupby(["trade_date", "underlying", "expiration"]).apply(_atm, include_groups=False).rename("iv_atm")
    ivs = ivs.join(atm_level, on=["trade_date", "underlying", "expiration"])
    ivs["iv_rel"] = (ivs["iv_crr"] - ivs["iv_atm"]) * 100
    ivs = ivs.loc[ivs["iv_rel"].between(-15, 25)]
    grid_k = np.linspace(-0.2, 0.2, 41)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True, facecolor=SURFACE)
    for ax, und in zip(axes, ("RTX", "BA", "LMT")):
        ax.set_facecolor(SURFACE)
        sub = ivs.loc[ivs["underlying"] == und]
        for label, color in DTE_COLORS.items():
            part = sub.loc[sub["tramo"] == label]
            ax.scatter(part["k"], part["iv_rel"], s=7, color=color, alpha=0.35, linewidths=0)
            if len(part) > 20:
                lo, hi = part["k"].quantile([0.05, 0.95])
                kk = grid_k[(grid_k >= lo) & (grid_k <= hi)]
                coef = np.polyfit(part["k"], part["iv_rel"], 2)
                ax.plot(kk, np.polyval(coef, kk), color=color, lw=2, label=label)
        ax.axvline(0, color="#d4d3cd", lw=1, zorder=0)
        ax.axhline(0, color="#d4d3cd", lw=1, zorder=0)
        ax.grid(axis="y", color="#ecebe6", lw=0.8)
        ax.set_title(und, color=INK, loc="left", fontsize=10)
        ax.set_xlabel(r"log-moneyness $k=\ln(K/F)$")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("IV americana menos IV ATM (puntos)")
    axes[0].legend(frameon=False, title="Plazo (ajuste cuadrático)", fontsize=8, title_fontsize=8, loc="upper right")
    fig.suptitle("Forma de la sonrisa: opciones fuera del dinero, foto semanal, mar-nov 2026",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "iv_vs_moneyness.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 2. Estructura temporal: IV ATM (|k|<2%) contra días al vencimiento.
    atm = ivs.loc[ivs["k"].abs() < 0.02]
    term = atm.groupby(["underlying", pd.cut(atm["dte"], [6, 15, 30, 45, 60, 90, 120])], observed=True)["iv_crr"].median().reset_index()
    term["dte_mid"] = term["dte"].map(lambda iv: iv.mid)
    fig, ax = plt.subplots(figsize=(6.5, 3.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for und, color in zip(("RTX", "BA", "LMT"), ("#2a78d6", "#eb6834", "#1baf7a")):
        part = term.loc[term["underlying"] == und]
        ax.plot(part["dte_mid"], part["iv_crr"] * 100, color=color, lw=2, marker="o", ms=4, label=und)
        last = part.iloc[-1]
        ax.annotate(und, (last["dte_mid"], last["iv_crr"] * 100), xytext=(6, 0), textcoords="offset points",
                    color=INK2, fontsize=8, va="center")
    ax.grid(axis="y", color="#ecebe6", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Días al vencimiento")
    ax.set_ylabel("IV ATM mediana (%)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Estructura temporal de la IV at-the-money", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "iv_vs_vencimiento.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(ROOT / "reportes_dolthub"))
    parser.add_argument("--plots-only", action="store_true", help="Rehace los gráficos con la salida guardada.")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    curated = pd.read_parquet(Path(args.data_root) / "dolthub" / "curated.parquet")
    curated["trade_date"] = pd.to_datetime(curated["trade_date"]).dt.date
    if args.plots_only:
        plot(pd.read_csv(out / "calibracion_iv_observada.csv"), out)
        return
    err, ivs = calibrate(curated)
    err.to_csv(out / "calibracion_errores_por_contrato.csv", index=False)
    ivs.to_csv(out / "calibracion_iv_observada.csv", index=False)
    table = summarize(err)
    table.to_csv(out / "calibracion_rmse.csv", index=False)
    _LOG.info("\n%s", table.to_string(index=False))
    plot(ivs, out)


if __name__ == "__main__":
    main()
