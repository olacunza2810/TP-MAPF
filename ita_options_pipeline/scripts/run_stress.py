r"""Paso 5 de la Parte 2: stress-testing por Monte Carlo de la cartera en producción.

1. **Cartera.** La que la competencia abriría a capacidad máxima en la última
   foto de DoltHub (20/11/2026): hasta tres estructuras por modelo, con sus
   precios límite de entrada.
2. **Calibración.**
   - Correlación y saltos (Merton) con 5 años de retornos diarios (Yahoo).
   - Dinámica de la volatilidad implícita ATM (Heston) con las 181 ruedas de
     DoltHub: :math:`\Delta v = a + b\,v + \varepsilon` da :math:`\kappa=-b/\Delta t`,
     :math:`\theta = a/(\kappa\Delta t)`, :math:`\xi = \operatorname{sd}(\varepsilon)/\sqrt{\bar v\,\Delta t}`
     y :math:`\rho = \operatorname{corr}(\varepsilon, r)`.
   - Volatilidad inicial: la ATM de la sonrisa del binomial en la foto.
3. **Escenarios** a 10 ruedas con 5.000 trayectorias, más escenarios
   determinísticos (shocks y repeticiones históricas).
4. **Coberturas:** sin cobertura, delta y delta-gamma entera (la de paper), con
   las reglas de producción (kill-switch USD 250 por estructura, pérdida diaria
   USD 1.000, take-profit 60%).

Salidas en ``reportes_dolthub/``: ``estres_cartera.csv``, ``estres_parametros.json``,
``estres_resultados.csv``, ``estres_resumen.json`` y ``estres_es99.png``.

Uso::

    python scripts/run_stress.py               # ~10 min
    python scripts/run_stress.py --paths 500   # prueba
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ita_options.arbitrage import ExecutionCosts  # noqa: E402
from ita_options.dolthub import attach_projected_dividends, merge_dividends  # noqa: E402
from ita_options.execution.hedger import pick_gamma_option  # noqa: E402
from ita_options.execution.ledger import Ledger  # noqa: E402
from ita_options.execution.model_rv import CompetingModels, RVSettings, fit_smile  # noqa: E402
from ita_options.escrowed import implied_vol  # noqa: E402
from ita_options.greeks import crr_price  # noqa: E402
from ita_options.stress import _pv_remaining  # noqa: E402
from ita_options.stress import (  # noqa: E402
    Book, HedgeOption, Leg, RiskRules, ScenarioPaths, Structure, risk_metrics, run_account,
    simulate_gbm, simulate_heston, simulate_merton,
)

_LOG = logging.getLogger("estres")
UNDERLYINGS = ("RTX", "BA", "LMT")
HORIZON = 10
HOLIDAYS = [date(2026, 11, 26)]  # Thanksgiving
HEDGES = ("sin_cobertura", "delta", "delta_gamma")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#ecebe6"


# --------------------------------------------------------------------------- #
# Cartera
# --------------------------------------------------------------------------- #


def load_day(data_root: Path, day: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    curated = pd.read_parquet(data_root / "dolthub" / "curated.parquet")
    curated["trade_date"] = pd.to_datetime(curated["trade_date"]).dt.date
    dividends = merge_dividends(pd.read_parquet(data_root / "dolthub" / "dividends.parquet"),
                                data_root / "dolthub" / "dividends_polygon.parquet")
    chain = curated.loc[curated["trade_date"] == day].copy()
    # Spot implícito de la foto (sincrónico con las opciones; ver el paso 3).
    chain["underlying_price"] = chain["implied_spot"]
    chain = attach_projected_dividends(chain, dividends)
    return chain, dividends


def build_books(chain: pd.DataFrame, rate: float) -> tuple[list[Book], pd.DataFrame]:
    """Libros de la competencia a capacidad máxima y su descripción."""
    source = CompetingModels(Ledger(), ExecutionCosts(min_net_edge=10.0),
                             {m: RVSettings(model=m, max_signals=10) for m in ("crr", "bsm")}, cap_per_model=3)
    signals = source(chain)
    by_symbol = chain.set_index("symbol")
    smiles: dict = {}
    fit_base = chain.loc[chain["is_tradable"]]

    def smile(model, und, exp):
        key = (model, und, exp)
        if key not in smiles:
            part = fit_base.loc[(fit_base["underlying"] == und) & (fit_base["expiration"] == exp)]
            smiles[key] = fit_smile(part, model, RVSettings(model=model))
        return smiles[key]

    def make_leg(symbol: str, units: float) -> Leg:
        row = by_symbol.loc[symbol]
        und, exp = row["underlying"], row["expiration"]
        divs = tuple(row["dividend_schedule"])
        sig = {m: smile(m, und, exp).sigma(float(row["strike"])) for m in ("crr", "bsm")}
        model_value = crr_price(float(row["underlying_price"]), float(row["strike"]), float(row["tau"]), rate,
                                sig["crr"], row["option_type"], divs, steps=100)
        mid = 0.5 * (row["bid"] + row["ask"])
        iv = implied_vol("crr", mid, float(row["underlying_price"]), float(row["strike"]), float(row["tau"]), rate,
                         row["option_type"], divs)
        sig["mkt"] = iv if np.isfinite(iv) else sig["crr"]
        return Leg(u=UNDERLYINGS.index(und), strike=float(row["strike"]), flag=row["option_type"],
                   tau0=float(row["tau"]), units=units, sigma=sig,
                   resid=0.5 * (row["bid"] + row["ask"]) - model_value, half=0.5 * (row["ask"] - row["bid"]),
                   dividends=divs)

    books = {m: Book(m, []) for m in ("crr", "bsm")}
    rows = []
    for sig in signals.itertuples():
        model = source.model_of(sig.detector)
        legs = [make_leg(leg["symbol"], float(leg["qty"])) for leg in sig.leg_spec]
        entry_cash = -sum(float(leg["qty"]) * float(leg["price"]) * 100.0 for leg in sig.leg_spec) - sig.commission
        books[model].structures.append(Structure(model, sig.legs, legs, entry_cash, float(sig.net_edge_usd)))
        rows.append({"modelo": model, "estructura": sig.legs, "subyacente": sig.underlying,
                     "vencimiento": sig.expiration, "edge_usd": sig.net_edge_usd, "efectivo_entrada": entry_cash})
    # Opción de gamma por subyacente (la misma regla que el hedger en vivo).
    for model, book in books.items():
        for u, und in enumerate(UNDERLYINGS):
            exps = {by_symbol.loc[l["symbol"], "expiration"] for s in signals.itertuples()
                    if source.model_of(s.detector) == model and s.underlying == und for l in s.leg_spec}
            if not exps:
                continue
            # Sólo vencimientos donde los dos modelos tienen sonrisa (si no, no hay griegas).
            ok = [e for e in chain.loc[chain["underlying"] == und, "expiration"].unique()
                  if smile("crr", und, e) is not None and smile("bsm", und, e) is not None]
            # En vivo la opción se rota cuando le quedan menos de 7 días; acá no se rota
            # dentro del horizonte (14 días corridos), así que se pide al menos 21.
            pick = pick_gamma_option(chain.loc[chain["expiration"].isin(ok)], und, exps, min_dte=21)
            if pick is not None:
                grid = np.diff(np.sort(chain.loc[(chain["underlying"] == und)
                                                 & (chain["expiration"] == pick["expiration"]), "strike"].unique()))
                book.gamma_options[u] = HedgeOption(
                    make_leg(pick["symbol"], 0.0),
                    {m: smile(m, und, pick["expiration"]) for m in ("crr", "bsm")},
                    strike_step=float(np.median(grid)) if len(grid) else 5.0)
                rows.append({"modelo": model, "estructura": f"cobertura gamma: call {pick['strike']:g}",
                             "subyacente": und, "vencimiento": pick["expiration"], "edge_usd": np.nan,
                             "efectivo_entrada": 0.0})
    return [b for b in books.values() if b.structures], pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Calibración
# --------------------------------------------------------------------------- #


def calibrate(data_root: Path, chain: pd.DataFrame) -> dict:
    hist = pd.read_csv(data_root / "underlying_bars" / "historia_10y.csv", parse_dates=["date"])
    prices = hist.pivot(index="date", columns="symbol", values="adjclose")[list(UNDERLYINGS)]
    returns = np.log(prices).diff().dropna()
    five = returns.loc[returns.index >= returns.index.max() - pd.DateOffset(years=5)]
    corr = five.corr().to_numpy()
    mad = 1.4826 * (five - five.median()).abs().median()
    jumps = five.abs() > 4 * mad
    years = len(five) / 252
    lam = (jumps.sum() / years).to_numpy()
    mu_j = five[jumps].mean().fillna(0).to_numpy()
    sd_j = five[jumps].std().fillna(0.05).to_numpy()
    sigma_diff = (five[~jumps].std() * np.sqrt(252)).to_numpy()

    # Volatilidad ATM inicial (sonrisa del binomial, vencimiento de 14 días o más más cercano).
    spot0, vol0 = [], []
    fit_base = chain.loc[chain["is_tradable"]]
    for und in UNDERLYINGS:
        part = fit_base.loc[(fit_base["underlying"] == und) & (fit_base["dte"] >= 14)]
        sm = None
        for exp in sorted(part["expiration"].unique()):
            sm = fit_smile(part.loc[part["expiration"] == exp], "crr", RVSettings(model="crr"))
            if sm is not None:
                break
        spot0.append(float(part["underlying_price"].iloc[0]))
        vol0.append(sm.sigma(sm.forward))

    # Dinámica de la IV ATM con DoltHub.
    cur = pd.read_parquet(data_root / "dolthub" / "curated.parquet")
    atm = cur.loc[cur["dte"].between(20, 60) & (np.log(cur["strike"] / cur["implied_spot"]).abs() < 0.03)
                  & cur["provider_iv"].between(0.05, 1.5)]
    iv = atm.groupby(["trade_date", "underlying"])["provider_iv"].median().unstack()[list(UNDERLYINGS)]
    sp = cur.groupby(["trade_date", "underlying"])["implied_spot"].first().unstack()[list(UNDERLYINGS)]
    r_d = np.log(sp).diff()
    dt = 1 / 252
    heston = {"kappa": [], "theta": [], "xi": [], "rho": [], "beta_vol_retorno": []}
    for und in UNDERLYINGS:
        v = iv[und] ** 2
        x = pd.concat([v.shift(), v.diff(), r_d[und], iv[und].diff()], axis=1).dropna()
        x.columns = ["v", "dv", "r", "dsig"]
        b, a = np.polyfit(x["v"], x["dv"], 1)
        kappa = -b / dt
        resid = x["dv"] - (a + b * x["v"])
        heston["kappa"].append(float(kappa))
        heston["theta"].append(float(a / (kappa * dt)))
        heston["xi"].append(float(resid.std() / math.sqrt(x["v"].mean() * dt)))
        heston["rho"].append(float(np.corrcoef(resid, x["r"])[0, 1]))
        heston["beta_vol_retorno"].append(float(np.polyfit(x["r"], x["dsig"], 1)[0]))
    return {
        "subyacentes": list(UNDERLYINGS), "spot0": spot0, "vol0": vol0, "corr": corr.tolist(),
        "merton": {"lambda": lam.tolist(), "mu_j": mu_j.tolist(), "sigma_j": sd_j.tolist(),
                   "sigma_difusion": sigma_diff.tolist(), "anios": years},
        "heston": heston,
        "ventana_retornos": [str(five.index.min().date()), str(five.index.max().date())],
        "returns": returns,  # para las repeticiones históricas (no va al JSON)
    }


# --------------------------------------------------------------------------- #
# Escenarios
# --------------------------------------------------------------------------- #


def horizon_calendar(day: date) -> tuple[list[date], np.ndarray]:
    dates = [d.date() for d in pd.bdate_range(day, periods=HORIZON + 1 + len(HOLIDAYS), freq="C",
                                               holidays=HOLIDAYS)][: HORIZON + 1]
    return dates, np.array([(d - day).days / 365.0 for d in dates])


def ex_days(dates: list[date], chain: pd.DataFrame) -> list[list[tuple[int, float]]]:
    """Dividendos proyectados con fecha ex dentro del horizonte, como índice de rueda."""
    out = []
    for und in UNDERLYINGS:
        sched = max((tuple(s) for s in chain.loc[chain["underlying"] == und, "dividend_schedule"]), key=len, default=())
        flows = []
        for t_ex, amount in sched:
            ex_date = dates[0] + timedelta(days=round(t_ex * 365))
            idx = next((i for i, d in enumerate(dates) if d >= ex_date), None)
            if idx is not None and idx > 0:
                flows.append((idx, amount))
        out.append(flows)
    return out


def deterministic(spot0: np.ndarray, vol0: np.ndarray, rets: np.ndarray, dvol: np.ndarray, factor: float,
                  paid: np.ndarray, pv: np.ndarray) -> ScenarioPaths:
    """Un camino: ``rets`` ``(D, U)`` retornos log diarios de :math:`S^*`, ``dvol`` cambios de vol ATM.

    Mismo modelo escrowed que los Monte Carlo: :math:`S = S^* + D(t)`.
    """
    star0 = spot0 - pv[0]
    log_s = np.vstack([np.log(star0), np.log(star0) + np.cumsum(rets, axis=0)])
    spot = np.exp(log_s) + pv
    vol = np.vstack([vol0, vol0 + np.cumsum(dvol, axis=0)])
    return ScenarioPaths(spot[None], np.maximum(vol, 0.05)[None], paid, factor)


def replay(returns: pd.DataFrame, end: str) -> np.ndarray:
    """Retornos de las 10 ruedas que terminan en ``end``, por subyacente."""
    window = returns.loc[:end].tail(HORIZON)
    return window[list(UNDERLYINGS)].to_numpy()


def scenarios(params: dict, dates, cal, paid_days, paths: int, seed: int) -> dict[str, tuple[str, ScenarioPaths]]:
    s0 = np.array(params["spot0"])
    v0 = np.array(params["vol0"])
    corr = np.array(params["corr"])
    m, h = params["merton"], params["heston"]
    rate = params["rate"]
    rng = np.random.default_rng(seed)
    out: dict[str, tuple[str, ScenarioPaths]] = {}
    out["gbm"] = ("GBM con la vol implícita ATM", simulate_gbm(s0, v0, corr, rate, HORIZON, paths, rng, paid_days, calendar=cal))
    out["heston"] = ("Heston calibrado a la IV de DoltHub", simulate_heston(
        s0, v0 ** 2, np.clip(h["kappa"], 1, 25), np.clip(h["theta"], 0.01, 1), np.array(h["xi"]),
        np.array(h["rho"]), corr, rate, HORIZON, paths, rng, paid_days, calendar=cal))
    out["heston_estres"] = ("Heston estresado: theta=(1,5 vol)^2, kappa=3, xi=1, rho=-0,7", simulate_heston(
        s0, v0 ** 2, np.full(3, 3.0), (1.5 * v0) ** 2, np.full(3, 1.0), np.full(3, -0.7), corr, rate, HORIZON,
        paths, rng, paid_days, calendar=cal))
    out["merton"] = ("Merton calibrado (5 años, saltos > 4 MAD)", simulate_merton(
        s0, np.array(m["sigma_difusion"]), v0, np.array(m["lambda"]), np.array(m["mu_j"]),
        np.array(m["sigma_j"]), corr, rate, HORIZON, paths, rng, paid_days, calendar=cal))
    out["merton_sectorial"] = ("Merton sectorial: 4 saltos/año comunes de -8% (sd 5%)", simulate_merton(
        s0, np.array(m["sigma_difusion"]), v0, np.full(3, 4.0), np.full(3, -0.08), np.full(3, 0.05), corr, rate,
        HORIZON, paths, rng, paid_days, common=True, calendar=cal))
    liq = simulate_gbm(s0, v0, corr, rate, HORIZON, paths, rng, paid_days, calendar=cal)
    liq.spread_factor = 3.0
    out["liquidez"] = ("GBM con spreads x3 al liquidar", liq)

    paid = np.zeros((HORIZON + 1, 3))
    for u, flows in enumerate(paid_days):
        for day, amount in flows:
            paid[day, u] += amount
    pv = _pv_remaining(s0, paid_days, rate, cal)
    zeros = np.zeros((HORIZON, 3))

    def shock(ret, dv, factor=1.0):
        r = zeros.copy(); r[0] = ret
        d = zeros.copy(); d[0] = dv
        return deterministic(s0, v0, r, d, factor, paid, pv)

    out["caida_15"] = ("Caída de 15% en un día, vol +15 pts, spreads x3",
                       shock(np.full(3, math.log(0.85)), np.full(3, 0.15), 3.0))
    out["suba_10"] = ("Suba de 10% en un día, vol -5 pts", shock(np.full(3, math.log(1.10)), np.full(3, -0.05)))
    out["vol_20"] = ("Vol +20 pts sin movimiento del precio", shock(np.zeros(3), np.full(3, 0.20)))
    # Repeticiones históricas. La vol implícita sube 0,75 pts por cada 1% de caída
    # (regla de estrés: la beta medida en 2026, -0,1, es de un régimen tranquilo).
    for key, label, end in (("covid", "Repetición COVID (10 ruedas al 18/03/2020)", "2020-03-18"),
                            ("aranceles", "Repetición aranceles (10 ruedas al 07/04/2025)", "2025-04-07")):
        rets = replay(params["returns"], end)
        dv = np.where(rets < 0, -0.75 * rets, 0.0)
        out[key] = (label, deterministic(s0, v0, rets, dv, 2.0, paid, pv))
    return out


# --------------------------------------------------------------------------- #
# Salidas
# --------------------------------------------------------------------------- #


def plot(results: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mc = results.loc[results["tipo"] == "monte_carlo"]
    det = results.loc[results["tipo"] == "deterministico"]
    colors = {"sin_cobertura": "#d4d3cd", "delta": "#2a78d6", "delta_gamma": "#104281"}
    names = {"sin_cobertura": "Sin cobertura", "delta": "Delta (acciones)", "delta_gamma": "Delta-gamma entera (paper)"}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [mc["escenario"].nunique(), det["escenario"].nunique()]})
    for ax, part, col, title in ((axes[0], mc, "es99_mid", "Monte Carlo: ES 99% de la cuenta a 10 ruedas, al mid (USD)"),
                                 (axes[1], det, "pnl_mid", "Determinísticos: P&L de la cuenta al mid (USD)")):
        ax.set_facecolor(SURFACE)
        order = list(dict.fromkeys(part["escenario"]))
        x = np.arange(len(order))
        for j, hedge in enumerate(HEDGES):
            vals = part.loc[part["cobertura"] == hedge].set_index("escenario").reindex(order)[col]
            ax.bar(x + (j - 1) * 0.27, vals, width=0.25, color=colors[hedge], label=names[hedge])
        ax.axhline(0, color=INK2, lw=0.8)
        if col == "es99_mid":
            ax.axhline(1000, color="#eb6834", lw=1, ls="--")
            ax.text(-0.45, 1010, "límite de pérdida diaria (referencia)", color="#eb6834", fontsize=8,
                    va="bottom", ha="left")
        ax.set_xticks(x, [o.replace("_", "\n") for o in order])
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center", fontsize=9)
    fig.suptitle("Estrés de la cartera en producción (20/11/2026) con las reglas de riesgo; "
                 "sin incluir el costo de salida (cruzar el spread al cerrar)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "estres_es99.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(ROOT / "reportes_dolthub"))
    parser.add_argument("--day", default="2026-11-20")
    parser.add_argument("--paths", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--plot-only", action="store_true", help="Rehace el gráfico con estres_resultados.csv.")
    args = parser.parse_args(argv)
    if args.plot_only:
        plot(pd.read_csv(Path(args.out) / "estres_resultados.csv"), Path(args.out))
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    day = date.fromisoformat(args.day)
    data_root = Path(args.data_root)

    chain, _ = load_day(data_root, day)
    rate = float(chain["risk_free_rate"].iloc[0])
    books, table = build_books(chain, rate)
    table.to_csv(out / "estres_cartera.csv", index=False)
    _LOG.info("Cartera:\n%s", table.to_string(index=False))
    params = calibrate(data_root, chain)
    params["rate"] = rate
    dates, cal = horizon_calendar(day)
    paid_days = ex_days(dates, chain)
    params["fechas"] = [str(d) for d in dates]
    params["dividendos_en_horizonte"] = {u: f for u, f in zip(UNDERLYINGS, paid_days)}
    (out / "estres_parametros.json").write_text(json.dumps({k: v for k, v in params.items() if k != "returns"},
                                                           indent=2, default=str), encoding="utf-8")
    rules = RiskRules()
    rows = []
    for key, (label, paths) in scenarios(params, dates, cal, paid_days, args.paths, args.seed).items():
        start = time.time()
        cache: dict = {}
        kind = "monte_carlo" if paths.spot.shape[0] > 1 else "deterministico"
        for hedge in HEDGES:
            res = run_account(books, paths, rate, cal, hedge, rules, steps=args.steps, cache=cache)
            row = {"escenario": key, "descripcion": label, "tipo": kind, "cobertura": hedge,
                   "trayectorias": paths.spot.shape[0], "pnl": float(res["pnl"].mean()),
                   "pnl_mid": float(res["pnl_mid"].mean()),
                   "error_estandar_media": float(res["pnl"].std(ddof=1) / math.sqrt(len(res["pnl"])))
                   if len(res["pnl"]) > 1 else 0.0,
                   "p_kill_switch": float(res["kill"].mean()), "p_perdida_diaria": float(res["perdida_diaria"].mean()),
                   "p_take_profit": float(res["take_profit"].mean()),
                   "costo_cobertura": float(res["costo_cobertura"].mean()),
                   "peor_dia_medio": float(res["peor_dia"].mean())}
            # Costo de salida: lo que cuesta cerrar todo cruzando el spread al final.
            row["costo_salida"] = float((res["pnl_mid"] - res["pnl"]).mean())
            if kind == "monte_carlo":
                row.update(risk_metrics(res["pnl"]))
                row.update({f"{k}_mid": v for k, v in risk_metrics(res["pnl_mid"]).items()})
                for model in ("crr", "bsm"):
                    if f"pnl_mid_{model}" in res:
                        mm = risk_metrics(res[f"pnl_mid_{model}"])
                        row[f"media_mid_{model}"], row[f"es99_mid_{model}"] = mm["media"], mm["es99"]
            else:
                for model in ("crr", "bsm"):
                    if f"pnl_mid_{model}" in res:
                        row[f"media_mid_{model}"] = float(res[f"pnl_mid_{model}"][0])
            rows.append(row)
        if key == "gbm":
            # La regla anterior de producción (kill-switch y pérdida diaria a liquidación), para el informe.
            old = RiskRules(mark="liquidacion")
            for hedge in HEDGES:
                res = run_account(books, paths, rate, cal, hedge, old, steps=args.steps, cache=cache)
                rows.append({"escenario": "gbm_regla_anterior", "descripcion": "GBM con kill-switch y pérdida "
                             "diaria medidos a liquidación (regla anterior)", "tipo": "regla_anterior",
                             "cobertura": hedge, "trayectorias": paths.spot.shape[0],
                             "pnl": float(res["pnl"].mean()), "pnl_mid": float(res["pnl_mid"].mean()),
                             "p_kill_switch": float(res["kill"].mean()),
                             "p_perdida_diaria": float(res["perdida_diaria"].mean()),
                             "p_take_profit": float(res["take_profit"].mean()),
                             "costo_cobertura": float(res["costo_cobertura"].mean()),
                             "costo_salida": float((res["pnl_mid"] - res["pnl"]).mean()),
                             **risk_metrics(res["pnl"]),
                             **{f"{k}_mid": v for k, v in risk_metrics(res["pnl_mid"]).items()}})
        _LOG.info("%s listo en %.0fs", key, time.time() - start)
    results = pd.DataFrame(rows)
    results.to_csv(out / "estres_resultados.csv", index=False)
    summary = {"dia": args.day, "trayectorias": args.paths, "semilla": args.seed, "horizonte_ruedas": HORIZON,
               "estructuras": int(sum(len(b.structures) for b in books)),
               "reglas": rules.__dict__,
               "resultados": results.to_dict(orient="records")}
    (out / "estres_resumen.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    plot(results, out)
    cols = ["escenario", "cobertura", "pnl_mid", "es99_mid", "costo_salida", "p_kill_switch", "p_perdida_diaria",
            "p_take_profit", "costo_cobertura"]
    _LOG.info("\n%s", results[[c for c in cols if c in results]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
