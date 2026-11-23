r"""Paso 4 de la Parte 2: efectividad de las coberturas con griegas sobre DoltHub.

Experimento
-----------
- **Posiciones.** Se vende 1 contrato de cada opción operable del universo de
  la estrategia (:math:`|\ln(K/F)| \le 10\%`, 21 a 75 días) el primer día que
  aparece, y se la sigue mientras tenga cotización en fotos consecutivas, hasta
  10 ruedas y mientras le queden al menos 14 días; si vuelve al universo, abre
  otra corrida. Cada día-opción entra una sola vez. DoltHub no trae todos los
  vencimientos todos los días, así que muchas corridas son cortas.
- **Coberturas.** Cada día se rebalancea con las griegas de cada modelo
  (binomial y Black-Scholes, dividendo discreto escrowed), calculadas con la
  sonrisa del propio modelo ajustada en esa foto: sin cobertura, delta con
  acciones, delta-gamma con la call ATM del vencimiento más corto y delta-vega
  con la call ATM del más largo, y delta-gamma entera (la de paper: contratos
  enteros, sólo compras). Ver :mod:`ita_options.hedging`. La opción
  de cobertura se mantiene mientras siga a menos de 5% del dinero en el mismo
  vencimiento: rotarla todos los días a la ATM exacta paga el spread entero.
- **P&L.** A mids observados, con el spot implícito de cada foto (el cierre de
  la acción está desfasado de la foto de opciones; ver el paso 3), el dividendo
  cobrado por la acción y el carry del efectivo. Los costos de rebalanceo se
  informan aparte.

Sesgo a tener en cuenta: DoltHub trae una ventana de strikes alrededor del
spot, y una opción que se aleja mucho deja de cotizar y su posición termina.
Los días con movimientos grandes están subrepresentados; afecta igual a todas
las coberturas y a los dos modelos, así que la comparación entre ellas vale,
pero la varianza sin cobertura está subestimada.

Salidas en ``reportes_dolthub/``: ``cobertura_dias.parquet`` (una fila por
día-posición, modelo y cobertura), ``cobertura_efectividad.csv``,
``cobertura_efectividad_por_grupo.csv``, ``cobertura_dm.csv``,
``cobertura_resumen.json`` y ``cobertura_efectividad.png``.

Uso::

    python scripts/run_hedging.py                 # ~10 min con 2 procesos
    python scripts/run_hedging.py --max-weeks 3   # prueba
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ita_options.dolthub import dividend_schedule, merge_dividends  # noqa: E402
from ita_options.greeks import model_greeks  # noqa: E402
from ita_options.hedging import (  # noqa: E402
    STRATEGIES, HedgeCosts, PositionPath, effectiveness_table, pick_instrument, simulate_position,
)
from ita_options.model_comparison import ComparisonSettings, diebold_mariano, fit_escrowed_smile  # noqa: E402

_LOG = logging.getLogger("cobertura")
MODELS = ("crr", "bsm")
MAX_STEPS = 10
MIN_DTE_HOLD = 14
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#ecebe6"

_G: dict = {}


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #


def load(data_root: Path, spot_column: str) -> dict:
    curated = pd.read_parquet(data_root / "dolthub" / "curated.parquet")
    curated["trade_date"] = pd.to_datetime(curated["trade_date"]).dt.date
    curated["expiration"] = pd.to_datetime(curated["expiration"]).dt.date
    spot = curated.groupby(["underlying", "trade_date"])[spot_column].first().astype(float)
    curated["underlying_price"] = curated.set_index(["underlying", "trade_date"]).index.map(spot)
    dividends = merge_dividends(pd.read_parquet(data_root / "dolthub" / "dividends.parquet"),
                                data_root / "dolthub" / "dividends_polygon.parquet")
    days = sorted(curated["trade_date"].unique())
    quoted = curated.loc[(curated["bid"] > 0) & (curated["ask"] > curated["bid"])]
    quotes = {(d, s): (0.5 * (b + a), 0.5 * (a - b))
              for d, s, b, a in zip(quoted["trade_date"], quoted["symbol"], quoted["bid"], quoted["ask"])}
    # Dividendo realizado entre una foto y la siguiente (lo cobra quien tiene la acción).
    ex = pd.to_datetime(dividends["ex_dividend_date"]).dt.date
    paid: dict[tuple[str, object], float] = {}
    for prev, day in zip(days[:-1], days[1:]):
        for und in curated["underlying"].unique():
            mask = (dividends["ticker"] == und) & (ex > prev) & (ex <= day)
            paid[(und, day)] = float(dividends.loc[mask, "cash_amount"].sum())
    return {"curated": curated, "dividends": dividends, "days": days, "quotes": quotes,
            "spot": spot.to_dict(), "paid": paid}


def _init(data: dict) -> None:
    _G.clear()
    _G.update(data)
    _G["by_slice"] = {k: g for k, g in data["curated"].groupby(["trade_date", "underlying", "expiration"])}
    _G["smiles"] = {}
    _G["greeks"] = {}


def _slice(day, und, exp) -> pd.DataFrame | None:
    return _G["by_slice"].get((day, und, exp))


def _smiles(day, und, exp):
    """Sonrisas de los dos modelos para un vencimiento, con cache. ``None`` si alguna falla."""
    key = (day, und, exp)
    if key not in _G["smiles"]:
        part = _slice(day, und, exp)
        out = None
        if part is not None:
            fit_base = part.loc[part["is_tradable"]]
            if not fit_base.empty and 7 <= int(fit_base["dte"].iloc[0]) <= 120:
                divs = dividend_schedule(_G["dividends"], und, day, exp)
                fits = {m: fit_escrowed_smile(fit_base, m, divs, ComparisonSettings()) for m in MODELS}
                if all(f is not None for f in fits.values()):
                    out = ({m: fits[m][0] for m in MODELS}, divs)
        _G["smiles"][key] = out
    return _G["smiles"][key]


def _greeks(day, row) -> dict | None:
    """Griegas por modelo de una fila de la cadena en ``day``, con la sonrisa de cada modelo."""
    key = (day, row["symbol"])
    if key not in _G["greeks"]:
        sm = _smiles(day, row["underlying"], row["expiration"])
        out = None
        if sm is not None:
            smiles, divs = sm
            try:
                out = {m: model_greeks(m, float(row["underlying_price"]), float(row["strike"]), float(row["tau"]),
                                       float(row["risk_free_rate"]), smiles[m].sigma(float(row["strike"])),
                                       str(row["option_type"]), divs, steps=100) for m in MODELS}
            except ValueError:
                out = None
        _G["greeks"][key] = out
    return _G["greeks"][key]


def _atm_candidates(day, und) -> pd.DataFrame:
    """La call ATM de cada vencimiento con sonrisa en ``day`` (candidatas a cobertura)."""
    cache = _G.setdefault("candidates", {})
    if (day, und) not in cache:
        cache[(day, und)] = _atm_candidates_uncached(day, und)
    return cache[(day, und)]


def _atm_candidates_uncached(day, und) -> pd.DataFrame:
    rows = []
    for (d, u, exp), part in _G["by_slice"].items():
        if d != day or u != und:
            continue
        sm = _smiles(day, und, exp)
        if sm is None:
            continue
        calls = part.loc[(part["option_type"] == "c") & (part["bid"] > 0) & (part["ask"] > part["bid"])]
        if calls.empty:
            continue
        fwd = sm[0]["crr"].forward
        best = calls.iloc[(np.log(calls["strike"] / fwd)).abs().argsort().iloc[0]]
        rows.append(best)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Posiciones
# --------------------------------------------------------------------------- #


def _build_path(entry, row) -> PositionPath | None:
    days, quotes = _G["days"], _G["quotes"]
    i0 = days.index(entry)
    und, exp, sym = row["underlying"], row["expiration"], row["symbol"]
    seq, rows = [entry], [row]
    for d in days[i0 + 1: i0 + 1 + MAX_STEPS]:
        part = _slice(d, und, exp)
        if part is None or (d, sym) not in quotes:
            break
        seq.append(d)
        rows.append(part.loc[part["symbol"] == sym].iloc[0])
        if int(rows[-1]["dte"]) < MIN_DTE_HOLD:
            break
    # Recorta en el primer día en que falten griegas o una cobertura marcable.
    greeks = {m: [] for m in MODELS}
    hedges = {"delta_gamma": [], "delta_vega": []}
    prev_inst: dict = {}
    keep = 1
    for i in range(len(seq) - 1):
        g = _greeks(seq[i], rows[i])
        if g is None:
            break
        step_hedges = {}
        cands = _atm_candidates(seq[i], und)
        ok = not cands.empty
        for kind, strat in (("gamma", "delta_gamma"), ("vega", "delta_vega")):
            if not ok:
                break
            inst = pick_instrument(cands, exp, kind)
            prev = prev_inst.get(strat)
            if inst is not None and prev is not None and prev["expiration"] == inst["expiration"]:
                # Histéresis: se mantiene la opción de ayer mientras siga cerca del dinero
                # (|k| <= 5%) en el mismo vencimiento; rotar todos los días a la ATM exacta
                # paga el spread entero de salida y de entrada.
                today_slice = _slice(seq[i], und, prev["expiration"])
                same = today_slice.loc[today_slice["symbol"] == prev["symbol"]] if today_slice is not None else []
                sm_h = _smiles(seq[i], und, prev["expiration"])
                if len(same) and sm_h is not None and (seq[i], prev["symbol"]) in quotes \
                        and abs(math.log(float(prev["strike"]) / sm_h[0]["crr"].forward)) <= 0.05:
                    inst = same.iloc[0]
            nxt = quotes.get((seq[i + 1], inst["symbol"])) if inst is not None else None
            gh = _greeks(seq[i], inst) if inst is not None else None
            if inst is None or nxt is None or gh is None:
                ok = False
                break
            mid, half = quotes[(seq[i], inst["symbol"])]
            step_hedges[strat] = {"symbol": inst["symbol"], "mid": mid, "half": half, "greeks": gh,
                                  "mid_next": nxt[0], "half_next": nxt[1]}
            prev_inst[strat] = inst
        if not ok:
            break
        for m in MODELS:
            greeks[m].append(g[m])
        for strat in hedges:
            hedges[strat].append(step_hedges[strat])
        keep = i + 2
    if keep < 2:
        return None
    seq = seq[:keep]
    mids = [quotes[(d, sym)][0] for d in seq]
    halves = [quotes[(d, sym)][1] for d in seq]
    spots = [float(_G["spot"][(und, d)]) for d in seq]
    rates = [float(r["risk_free_rate"]) for r in rows[:keep]]
    paid = [0.0] + [_G["paid"].get((und, d), 0.0) for d in seq[1:]]
    return PositionPath(seq, mids, halves, spots, rates, paid, greeks, hedges)


def plan_runs(data: dict) -> list[tuple[object, str]]:
    """Corridas ``(día de entrada, símbolo)`` sin superponerse.

    Cada opción del universo entra el primer día que aparece y se sigue
    mientras cotice en fotos consecutivas (hasta :data:`MAX_STEPS` ruedas y
    mientras le queden :data:`MIN_DTE_HOLD` días). Si vuelve a estar en el
    universo después de terminar, abre una corrida nueva. Así cada día-opción
    se cuenta una sola vez.
    """
    curated, days, quotes = data["curated"], data["days"], data["quotes"]
    uni = curated.loc[curated["is_tradable"] & curated["dte"].between(21, 75)]
    # Banda de moneyness con el forward por paridad aproximado: se refina con la sonrisa en el worker.
    fwd = uni["underlying_price"] * np.exp(uni["risk_free_rate"] * uni["tau"]) - uni["pv_dividends"].fillna(0)
    uni = uni.loc[np.log(uni["strike"] / fwd).abs() <= 0.11]
    index = {d: i for i, d in enumerate(days)}
    dte = dict(zip(zip(curated["trade_date"], curated["symbol"]), curated["dte"]))
    runs = []
    busy_until: dict[str, int] = {}
    for day, sym in sorted(zip(uni["trade_date"], uni["symbol"])):
        i0 = index[day]
        if busy_until.get(sym, -1) >= i0:
            continue
        j = i0
        while j + 1 < len(days) and j - i0 < MAX_STEPS and (days[j + 1], sym) in quotes \
                and dte.get((days[j], sym), 0) >= MIN_DTE_HOLD:
            j += 1
        if j > i0:
            runs.append((day, sym))
            busy_until[sym] = j - 1  # el último día sólo marca: puede abrir la corrida siguiente
    return runs


def _run_batch(runs: list[tuple[object, str]]) -> pd.DataFrame:
    """Simula un lote de corridas con cada modelo y cobertura."""
    frames = []
    costs = HedgeCosts()
    for entry, sym in runs:
        part = _G["curated"].loc[(_G["curated"]["trade_date"] == entry) & (_G["curated"]["symbol"] == sym)]
        if part.empty:
            continue
        row = part.iloc[0]
        sm = _smiles(entry, row["underlying"], row["expiration"])
        if sm is None:
            continue
        fwd = sm[0]["crr"].forward
        k = math.log(float(row["strike"]) / fwd)
        if abs(k) > 0.10:
            continue
        path = _build_path(entry, row)
        if path is None:
            continue
        x = -k if row["option_type"] == "c" else k
        attrs = {
            "posicion": f"{entry}|{sym}", "entrada": entry, "underlying": row["underlying"],
            "option_type": row["option_type"], "dte_entrada": int(row["dte"]),
            "dinero": "OTM" if x < -0.03 else ("ATM" if x <= 0.03 else "ITM"),
            "plazo": "21-45 días" if int(row["dte"]) <= 45 else "46-75 días",
            "con_dividendo": bool(sum(path.dividend) > 0),
        }
        dd = [abs(a.delta - b.delta) for a, b in zip(path.greeks["crr"], path.greeks["bsm"])]
        for model in MODELS:
            for strat in STRATEGIES:
                sim = simulate_position(path, model, strat, costs=costs)
                frames.append(sim.assign(modelo=model, cobertura=strat, dif_delta=dd, **attrs))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Tablas
# --------------------------------------------------------------------------- #


def dm_by_strategy(days: pd.DataFrame) -> pd.DataFrame:
    """Diebold-Mariano del error de cobertura al cuadrado, BS contra binomial, por cobertura."""
    rows = []
    for strat, part in days.loc[days["cobertura"] != "sin_cobertura"].groupby("cobertura"):
        wide = part.pivot_table(index=["posicion", "paso", "dia"], columns="modelo", values="pnl").reset_index()
        for label, frame in (("bruto", wide),):
            d = (frame["bsm"] ** 2 - frame["crr"] ** 2).groupby(frame["dia"]).mean().sort_index()
            res = diebold_mariano(d)
            rows.append({"cobertura": strat, "perdida": "pnl^2 " + label, "dias_posicion": len(frame), **res})
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["delta", "delta_gamma", "delta_gamma_entera", "delta_vega"]
    labels = {"delta": "Delta\n(acción)", "delta_gamma": "Delta-gamma\n(ATM corta + acción)",
              "delta_gamma_entera": "Delta-gamma entera\n(sólo compras, paper)",
              "delta_vega": "Delta-vega\n(ATM larga + acción)"}
    colors = {"crr": "#2a78d6", "bsm": "#eb6834"}
    names = {"crr": "Binomial (CRR)", "bsm": "Black-Scholes"}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.9), sharey=True, facecolor=SURFACE)
    x = np.arange(len(order))
    for ax, col, title in zip(axes, ("efectividad", "efectividad_neta"),
                              ("Bruta (a mids)", "Neta de costos de rebalanceo")):
        ax.set_facecolor(SURFACE)
        for j, model in enumerate(MODELS):
            vals = table.set_index(["modelo", "cobertura"]).loc[model].reindex(order)[col] * 100
            bars = ax.bar(x + (j - 0.5) * 0.36, vals, width=0.34, color=colors[model], label=names[model])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 1.5, f"{v:.0f}%", ha="center",
                        va="bottom", fontsize=8, color=INK2)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x, [labels[o] for o in order])
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Efectividad: 1 − Var(cubierta)/Var(sin cubrir), %")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=9, loc="lower center", ncol=2)
    fig.suptitle("Coberturas con griegas sobre opciones vendidas de RTX, BA y LMT — rebalanceo diario, mar-nov 2026",
                 x=0.01, ha="left", color=INK, fontsize=11)
    for ax in axes:
        ax.set_ylim(0, 108)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "cobertura_efectividad.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(ROOT / "reportes_dolthub"))
    parser.add_argument("--spot", choices=["implied_spot", "underlying_price"], default="implied_spot")
    parser.add_argument("--max-weeks", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.spot == "implied_spot" else "_spot_cierre"
    days_path = out / f"cobertura_dias{suffix}.parquet"
    if args.tables_only:
        days = pd.read_parquet(days_path)
    else:
        data = load(Path(args.data_root), args.spot)
        runs = plan_runs(data)
        weeks = pd.Series([d for d, _ in runs]).map(lambda d: d.isocalendar()[:2])
        batches = [list(g) for _, g in pd.Series(runs).groupby(weeks.to_numpy())]
        if args.max_weeks:
            batches = batches[: args.max_weeks]
        _LOG.info("%d corridas en %d semanas", sum(len(b) for b in batches), len(batches))
        start = time.time()
        frames = []
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(data,)) as pool:
            for i, frame in enumerate(pool.map(_run_batch, batches), 1):
                frames.append(frame)
                _LOG.info("Semana %d/%d (%.0fs)", i, len(batches), time.time() - start)
        days = pd.concat(frames, ignore_index=True)
        days.to_parquet(days_path, index=False)

    table = effectiveness_table(days)
    table.to_csv(out / f"cobertura_efectividad{suffix}.csv", index=False)
    groups = pd.concat([effectiveness_table(days, by=[g]).rename(columns={g: "valor"}).assign(grupo=g)
                        for g in ("dinero", "option_type", "plazo", "underlying", "con_dividendo")],
                       ignore_index=True)
    groups.to_csv(out / f"cobertura_efectividad_por_grupo{suffix}.csv", index=False)
    dm = dm_by_strategy(days)
    dm.to_csv(out / f"cobertura_dm{suffix}.csv", index=False)
    base = days.loc[(days["cobertura"] == "sin_cobertura") & (days["modelo"] == "crr")]
    hedged = days.loc[days["cobertura"] != "sin_cobertura"]
    last = hedged.groupby(["modelo", "cobertura", "posicion"])["paso"].transform("max")
    middle = hedged.loc[(hedged["paso"] > 0) & (hedged["paso"] < last)]
    summary = {
        "spot": args.spot,
        "posiciones": int(base["posicion"].nunique()),
        "dias_posicion": int(len(base)),
        "dias_de_entrada": int(base["entrada"].nunique()),
        "pasos_medios_por_posicion": float(base.groupby("posicion").size().mean()),
        "dif_delta_media": float(base["dif_delta"].mean()),
        "dif_delta_por_grupo": {f"{t}_{m}": float(v) for (t, m), v in
                                base.groupby(["option_type", "dinero"])["dif_delta"].mean().items()},
        # El costo se separa en armar/desarmar la cobertura (primer y último día de cada
        # corrida) y rebalancear (días intermedios): con corridas cortas domina lo primero.
        "costo_por_posicion": {f"{m}_{c}": float(v) for (m, c), v in
                               hedged.groupby(["modelo", "cobertura"]).apply(
                                   lambda g: g.groupby("posicion")["costo"].sum().mean(), include_groups=False).items()},
        "costo_rebalanceo_dia_intermedio": {f"{m}_{c}": float(v) for (m, c), v in
                                            middle.groupby(["modelo", "cobertura"])["costo"].mean().items()},
        "dias_intermedios": int(len(middle.loc[(middle["modelo"] == "crr") & (middle["cobertura"] == "delta")])),
        "efectividad": table.to_dict(orient="records"),
        "dm": dm.to_dict(orient="records"),
    }
    (out / f"cobertura_resumen{suffix}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if not suffix:
        plot(table, out)
    _LOG.info("\n%s", table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
