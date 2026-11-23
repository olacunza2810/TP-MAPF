r"""Scorecard de la decisión final (Parte 2, paso 6), con los criterios fijados de antemano.

La consigna pide decidir, con evidencia, qué modelo (binomial o Black-Scholes)
y qué estrategia resultaron superiores. Los criterios, las métricas y las
reglas de decisión quedan escritos acá **antes** de tener el P&L del paper
trading (el commit que agrega este archivo lo fecha), para que el resultado de
producción no pueda elegir su propio criterio. Los criterios M1-M3 y E1-E3 se
fijaron después de los backtests de los pasos 3 a 5, cuyos resultados ya se
conocían; el que todavía no tiene datos es M4. El script sólo los evalúa.

Modelo: binomial (CRR) contra Black-Scholes (BS)
------------------------------------------------
Cada criterio da un ganador o "empate". Gana el modelo con más criterios
ganados; con igualdad, se declara empate y se elige el **más simple**
(Black-Scholes: fórmula cerrada, sin discretización).

M1. **Repricing** (paso 3). Diebold-Mariano del error en medios spreads sobre
    toda la cadena. Gana el modelo con menor error si :math:`p < 0{,}05`.
M2. **Relevancia económica del repricing** (paso 3). Fracción de contratos en
    los que la prima de ejercicio anticipado supera medio spread. Si es menor
    a 5%, la diferencia de M1 no cambia decisiones de trading y M2 da empate;
    si no, M2 la da al binomial.
M3. **Cobertura** (paso 4). Diebold-Mariano del error de cobertura delta al
    cuadrado. Gana el de menor error si :math:`p < 0{,}05`.
M4. **Producción** (paper). P&L realizado por modelo, por estructura cerrada.
    Sólo decide con al menos 10 estructuras cerradas por modelo y una
    diferencia de medias mayor a dos errores estándar; si no, "sin evidencia
    suficiente" (no suma para nadie).

Estrategia de cobertura: sin cobertura, delta, delta-gamma entera
-----------------------------------------------------------------
E1. **Efectividad neta de costos** (paso 4): mayor :math:`E` neta.
E2. **Riesgo de cola** (paso 5): menor ES 99% al mid promediado en los
    escenarios Monte Carlo.
E3. **Costo** (paso 5): menor costo medio de cobertura en el escenario GBM.
Gana la estrategia con más criterios; con igualdad, la de menor costo.

Uso::

    python scripts/scorecard.py                     # sin paper: M4 queda pendiente
    python scripts/scorecard.py --ledger data/paper/ledger.sqlite
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reportes_dolthub"
ALPHA = 0.05
NAMES = {"crr": "binomial", "bsm": "Black-Scholes"}


def _dm_winner(row: pd.Series, a: str, b: str) -> str:
    """``d = L_a - L_b``: positivo favorece a ``b``."""
    if not math.isfinite(row["p_valor"]) or row["p_valor"] >= ALPHA:
        return "empate"
    return b if row["d_medio"] > 0 else a


def model_criteria(ledger: Path | None) -> list[dict]:
    rows = []
    dm = pd.read_csv(R / "comparacion_dm.csv")
    total = dm.loc[(dm["grupo"] == "total") & (dm["comparacion"] == "bsm vs crr") & (dm["perdida"] == "L")].iloc[0]
    rows.append({"criterio": "M1 repricing (DM, medios spreads)", "evidencia":
                 f"d = {total['d_medio']:.4f}, DM = {total['dm']:.2f}, p = {total['p_valor']:.1e}",
                 "ganador": _dm_winner(total, "bsm", "crr")})
    summary = json.loads((R / "comparacion_resumen.json").read_text(encoding="utf-8"))
    share = summary["pct_eep_mayor_medio_spread"]
    rows.append({"criterio": "M2 relevancia económica (EEP > medio spread)", "evidencia": f"{100 * share:.2f}% de los contratos",
                 "ganador": "empate" if share < 0.05 else "crr"})
    hedge = pd.read_csv(R / "cobertura_dm.csv").set_index("cobertura").loc["delta"]
    rows.append({"criterio": "M3 cobertura delta (DM, error al cuadrado)", "evidencia":
                 f"d = {hedge['d_medio']:.2f}, DM = {hedge['dm']:.2f}, p = {hedge['p_valor']:.2f}",
                 "ganador": _dm_winner(hedge, "bsm", "crr")})
    rows.append(paper_criterion(ledger))
    return rows


def paper_criterion(ledger: Path | None) -> dict:
    name = "M4 producción (P&L realizado por estructura, paper)"
    if ledger is None or not Path(ledger).exists():
        return {"criterio": name, "evidencia": "sin ledger de paper todavía", "ganador": "pendiente"}
    with sqlite3.connect(str(ledger)) as conn:
        st = pd.read_sql("SELECT detector, state, realized_pnl FROM strategies", conn)
    st["modelo"] = st["detector"].str.extract(r"^rv_(crr|bsm)_")[0]
    closed = st.loc[(st["state"] == "closed") & st["modelo"].notna()]
    stats = closed.groupby("modelo")["realized_pnl"].agg(["count", "mean", "std"])
    text = "; ".join(f"{NAMES[m]}: {int(r['count'])} cerradas, media USD {r['mean']:.1f}" for m, r in stats.iterrows())
    if len(stats) < 2 or (stats["count"] < 10).any():
        return {"criterio": name, "evidencia": text or "sin estructuras cerradas", "ganador": "sin evidencia suficiente"}
    se = math.sqrt(sum((r["std"] ** 2) / r["count"] for _, r in stats.iterrows()))
    diff = stats.loc["crr", "mean"] - stats.loc["bsm", "mean"]
    winner = "sin evidencia suficiente" if abs(diff) <= 2 * se else ("crr" if diff > 0 else "bsm")
    return {"criterio": name, "evidencia": f"{text}; diferencia {diff:.1f} ± {se:.1f}", "ganador": winner}


def _unique_min(series: pd.Series, label: dict, tol: float = 0.005) -> str:
    """El de menor valor, o ``empate`` si otro queda a menos de ``tol`` (relativo)."""
    best = series.min()
    tied = series.loc[(series - best).abs() <= tol * max(abs(best), 1e-9)]
    return label[tied.index[0]] if len(tied) == 1 else "empate"


def strategy_criteria() -> list[dict]:
    eff = pd.read_csv(R / "cobertura_efectividad.csv")
    eff = eff.loc[eff["cobertura"].isin(["delta", "delta_gamma_entera"])].groupby("cobertura")["efectividad_neta"].mean()
    stress = pd.read_csv(R / "estres_resultados.csv")
    mc = stress.loc[stress["tipo"] == "monte_carlo"]
    es = mc.groupby("cobertura")["es99_mid"].mean()
    cost = mc.loc[mc["escenario"] == "gbm"].set_index("cobertura")["costo_cobertura"]
    label = {"sin_cobertura": "sin cobertura", "delta": "delta", "delta_gamma": "delta-gamma entera",
             "delta_gamma_entera": "delta-gamma entera"}
    rows = [
        {"criterio": "E1 efectividad neta (paso 4)",
         "evidencia": ", ".join(f"{label[k]} {100 * v:.0f}%" for k, v in eff.items()),
         "ganador": label[eff.idxmax()]},
        {"criterio": "E2 ES 99% al mid, promedio MC (paso 5)",
         "evidencia": ", ".join(f"{label[k]} USD {v:,.0f}".replace(",", ".") for k, v in es.items()),
         "ganador": _unique_min(es, label)},
        {"criterio": "E3 costo de cobertura, GBM (paso 5)",
         "evidencia": ", ".join(f"{label[k]} USD {v:.2f}" for k, v in cost.items()),
         "ganador": _unique_min(cost.drop("sin_cobertura"), label)},
    ]
    return rows


def decide(rows: list[dict], candidates: list[str], tiebreak: str) -> str:
    score = {c: sum(r["ganador"] == c for r in rows) for c in candidates}
    best = max(score.values())
    leaders = [c for c, s in score.items() if s == best]
    return leaders[0] if len(leaders) == 1 and best > 0 else tiebreak


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ledger", default=str(ROOT / "data" / "paper" / "ledger.sqlite"))
    parser.add_argument("--out", default=str(R))
    args = parser.parse_args(argv)
    models = model_criteria(Path(args.ledger))
    strategies = strategy_criteria()
    model = decide(models, ["crr", "bsm"], "bsm")
    strategy = decide(strategies, ["delta", "delta-gamma entera", "sin cobertura"], "delta")
    out = {
        "modelo": {"criterios": models, "decision": model, "nombre": NAMES[model],
                   "regla": "más criterios ganados; empate -> Black-Scholes (más simple)"},
        "estrategia": {"criterios": strategies, "decision": strategy,
                       "regla": "más criterios ganados; empate -> menor costo"},
        "pendiente": [r["criterio"] for r in models if r["ganador"] in ("pendiente", "sin evidencia suficiente")],
    }
    Path(args.out, "scorecard.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for title, rows, decision in (("Modelo", models, NAMES[model]), ("Cobertura", strategies, strategy)):
        print(f"\n{title}: {decision}")
        for r in rows:
            print(f"  {r['criterio']:<52} {r['ganador']:<26} {r['evidencia']}")


if __name__ == "__main__":
    main()
