r"""Verifica que los números del informe salgan de ``reportes_dolthub/``.

La devolución de la entrega parcial marcó que el informe y la presentación se
contradecían (dos subyacentes contra tres, 99,3% contra 76,5% de filas
conservadas). Este script recalcula cada número clave desde los CSV, lo
formatea como aparece en el informe (coma decimal, punto de miles) y falla si
no lo encuentra en ``informe_proyecto.md``.

Uso (desde ``ita_options_pipeline``)::

    python scripts/check_informe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reportes_dolthub"


def miles(x: float) -> str:
    """Entero con punto de miles: 95290 -> '95.290'; negativos con signo '−'."""
    s = f"{abs(round(x)):,}".replace(",", ".")
    return ("−" if round(x) < 0 else "") + s


def dec(x: float, n: int = 2) -> str:
    """Decimal con coma: 0.438 -> '0,44'."""
    s = f"{abs(x):.{n}f}".replace(".", ",")
    return ("−" if round(x, n) < 0 else "") + s


def pct(x: float) -> str:
    return f"{round(100 * x)}%"


def expected() -> dict[str, str]:
    out: dict[str, str] = {}
    data = json.loads((R / "resumen_datos.json").read_text(encoding="utf-8"))
    out["filas"] = miles(data["filas"])
    out["explotables"] = miles(data["explotables"])
    out["ruedas"] = str(data["ruedas"])
    for und, n in data["filas_por_subyacente"].items():
        out[f"filas_{und}"] = f"{und} {miles(n)}"
    out["spread_mediano"] = dec(100 * data["spread_rel_mediano"], 1) + "%"
    out["desfase_mediano"] = dec(data["desfase_spot_abs_mediano_pct"]) + "%"
    out["desfase_p90"] = dec(data["desfase_spot_abs_p90_pct"]) + "%"

    variants = pd.read_csv(R / "arbitraje_violaciones_por_variante.csv").set_index("variante")["put_call_parity"]
    for name, n in variants.items():
        out[f"paridad_{name}"] = miles(n)

    resumen = pd.read_csv(R / "arbitraje_resumen.csv")
    out["arbitraje_ejecutadas"] = "| 0 |" if (resumen["filled"] == 0).all() else "ejecutadas>0"

    calib = pd.read_csv(R / "calibracion_rmse.csv")
    crr = calib.loc[calib["modelo"] == "crr"]
    for r in crr.itertuples():
        key = f"calib_{r.muestra}_{r.ajuste}_{r.underlying}"
        out[key + "_rmse"] = dec(r.rmse_usd)
        out[key + "_dentro"] = pct(r.pct_dentro_bid_ask)

    wf = pd.read_csv(R / "rv_walkforward.csv")
    for model, grp in wf.groupby("modelo"):
        out[f"wf_{model}_oos_pnl"] = miles(grp["oos_pnl"].sum())
        out[f"wf_{model}_oos_trades"] = f"{int(grp['oos_trades'].sum())} operaciones"

    full = pd.read_csv(R / "rv_grilla_muestra_completa.csv")
    for model, grp in full.groupby("modelo"):
        out[f"grilla_{model}_dsr_max"] = dec(grp["dsr_n18"].max())
        out[f"grilla_{model}_pnl_min"] = miles(grp["pnl"].min())
        out[f"grilla_{model}_pnl_max"] = miles(grp["pnl"].max())

    sens = pd.read_csv(R / "rv_sensibilidad_10_ordenes.csv")
    out["sensibilidad_todas_pierden"] = "36 configuraciones pierden" if (sens["pnl"] < 0).all() else "alguna gana"
    for model, grp in sens.groupby("modelo"):
        out[f"sens_{model}_pnl"] = f"USD {miles(grp['pnl'].min())} a {miles(grp['pnl'].max())}"
    return out


def tex(x: float, n: int = 1) -> str:
    """Decimal en LaTeX con coma: 13.09 -> '13{,}1'."""
    return dec(x, n).replace(",", "{,}")


def pct1(x: float) -> str:
    return dec(100 * x, 1) + "%"


def expected_part2() -> dict[str, str]:
    """Números de la Parte 2 (secciones 7 a 11)."""
    out: dict[str, str] = {}
    cmp_ = json.loads((R / "comparacion_resumen.json").read_text(encoding="utf-8"))
    out["p2_contratos"] = miles(cmp_["contratos"]) + " contratos"
    out["p2_eep_medio_spread"] = pct1(cmp_["pct_eep_mayor_medio_spread"])
    out["p2_dentro_crr"] = pct1(cmp_["dentro_crr"])
    out["p2_dentro_bsm"] = pct1(cmp_["dentro_bsm"])
    uni = cmp_["universo_estrategia"]
    out["p2_universo"] = miles(uni["contratos"]) + " contratos"
    out["p2_brecha_10usd"] = pct(uni["pct_brecha_crr_bsm_mayor_10usd"])
    out["p2_puts_itm_bs"] = f"USD {round(-uni['sesgo_usd_contrato']['p_ITM']['bsm'])}"
    out["p2_calls_itm_q"] = f"USD {round(-uni['sesgo_usd_contrato']['c_ITM']['crr_q'])}"
    dm = pd.read_csv(R / "comparacion_dm.csv")
    L = dm.loc[dm["perdida"] == "L"]
    tot = L.loc[(L["grupo"] == "total") & (L["comparacion"] == "bsm vs crr")].iloc[0]
    out["p2_dm"] = f"DM^* = {tex(tot['dm'])}"
    out["p2_dbar"] = f"\\bar d = {tex(tot['d_medio'], 3)}"
    q = L.loc[(L["grupo"] == "total") & (L["comparacion"] == "crr_q vs crr")].iloc[0]
    out["p2_dm_q"] = f"DM^* = {tex(q['dm'])}"
    deep = L.loc[(L["grupo"] == "option_type+dinero") & (L["option_type"] == "p") & (L["dinero"] == "ITM profundo")
                 & (L["comparacion"] == "bsm vs crr")].iloc[0]
    out["p2_dm_puts"] = f"\\bar d = {tex(deep['d_medio'], 3)}$, $DM^* = {tex(deep['dm'])}"
    sin_ex = L.loc[(L["grupo"] == "option_type+fecha_ex") & (L["option_type"] == "c") & (L["fecha_ex"] == "sin fecha ex")
                   & (L["comparacion"] == "bsm vs crr")].iloc[0]
    fit = L.loc[(L["grupo"] == "en_ajuste") & (L["en_ajuste"] == True) & (L["comparacion"] == "bsm vs crr")].iloc[0]  # noqa: E712
    out["p2_p_sin_ex"] = f"p = {tex(sin_ex['p_valor'], 2)}"
    out["p2_p_ajuste"] = f"p = {tex(fit['p_valor'], 2)}"
    cl = json.loads((R / "comparacion_resumen_spot_cierre.json").read_text(encoding="utf-8"))
    cl_dm = next(d for d in cl["dm_total"] if d["comparacion"] == "bsm vs crr" and d["perdida"] == "L")
    out["p2_dm_cierre"] = f"DM^* = {tex(cl_dm['dm'])}"
    out["p2_dentro_cierre"] = f"sólo el {pct(cl['dentro_crr'])} de los precios"

    hed = json.loads((R / "cobertura_resumen.json").read_text(encoding="utf-8"))
    out["p2_posiciones"] = f"{miles(hed['posiciones'])} posiciones"
    out["p2_dias_posicion"] = f"{miles(hed['dias_posicion'])}\ndías-posición".replace("\\n", "\n")
    eff = pd.read_csv(R / "cobertura_efectividad.csv").groupby("cobertura")[["efectividad", "efectividad_neta"]].mean()
    names = {"delta": "Delta", "delta_gamma": "Delta-gamma", "delta_gamma_entera": "Delta-gamma entera",
             "delta_vega": "Delta-vega"}
    cost = {k.split("_", 1)[1]: v for k, v in hed["costo_por_posicion"].items() if k.startswith("crr_")}
    for key, name in names.items():
        row = f"| {name} | {pct(eff.loc[key, 'efectividad'])} | {pct(eff.loc[key, 'efectividad_neta'])} |"
        out[f"p2_eff_{key}"] = row
        if key in cost:
            out[f"p2_cost_{key}"] = f"{row} USD {round(cost[key])} |"
    out["p2_dif_delta"] = f"difiere en {dec(hed['dif_delta_media'], 3)} en promedio"
    dmh = pd.read_csv(R / "cobertura_dm.csv").set_index("cobertura")["p_valor"]
    out["p2_dm_cobertura"] = (f"$p = {tex(dmh['delta'], 2)}$ en\ndelta, ${tex(dmh['delta_gamma'], 2)}$ en delta-gamma, "
                              f"${tex(dmh['delta_vega'], 2)}$ en delta-vega").replace("\\n", "\n")
    hc = pd.read_csv(R / "cobertura_efectividad_spot_cierre.csv").groupby("cobertura")["efectividad"].mean()
    out["p2_cierre_delta"] = f"de {pct(eff.loc['delta', 'efectividad'])} a {pct(hc['delta'])}"

    st = pd.read_csv(R / "estres_resultados.csv")
    mc = st.loc[st["tipo"] == "monte_carlo"].pivot(index="escenario", columns="cobertura", values="es99_mid")
    labels = {"gbm": "GBM", "heston": "Heston calibrado", "heston_estres": "Heston estresado",
              "merton": "Merton calibrado", "merton_sectorial": "Merton sectorial", "liquidez": "Spreads x3"}
    for key, label in labels.items():
        r = mc.loc[key]
        out[f"p2_es_{key}"] = (f"| {label} | {miles(r['sin_cobertura'])} | {miles(r['delta'])} | "
                               f"{miles(r['delta_gamma'])} |")
    gbm = st.loc[(st["escenario"] == "gbm")].set_index("cobertura")
    liq = st.loc[(st["escenario"] == "liquidez")].set_index("cobertura")
    out["p2_salida"] = f"USD {miles(gbm.loc['sin_cobertura', 'costo_salida'])} en condiciones normales y USD {miles(liq.loc['sin_cobertura', 'costo_salida'])}"
    mc_all = st.loc[st["tipo"] == "monte_carlo"]
    out["p2_kill"] = f"en el {pct(mc_all['p_kill_switch'].min())} a {pct(mc_all['p_kill_switch'].max())} de las trayectorias"
    old = st.loc[st["escenario"] == "gbm_regla_anterior"].set_index("cobertura")
    out["p2_regla_anterior"] = f"USD {miles(old.loc['sin_cobertura', 'pnl_mid'])} contra {miles(gbm.loc['sin_cobertura', 'pnl_mid'])}"
    out["p2_por_modelo"] = f"USD {round(gbm.loc['delta', 'es99_mid_bsm'])} contra {round(gbm.loc['delta', 'es99_mid_crr'])} en GBM"
    es_mean = mc_all.groupby("cobertura")["es99_mid"].mean()
    out["p2_e2"] = (f"delta USD {miles(es_mean['delta'])}, delta-gamma USD {miles(es_mean['delta_gamma'])}, "
                    f"sin cubrir USD {miles(es_mean['sin_cobertura'])}")
    return out


def expected_replay() -> dict[str, str]:
    """Números de la simulación histórica 2020-2026 (sección 12), desde ``reportes_replay/``."""
    rp = ROOT / "reportes_replay"
    datos = json.loads((rp / "resumen_datos.json").read_text(encoding="utf-8"))
    res = pd.read_csv(rp / "replay_resumen.csv")
    m4 = pd.read_csv(rp / "replay_m4.csv").set_index("modo")
    trades = pd.read_csv(rp / "replay_estructuras_delta.csv")
    closed = trades.dropna(subset=["cerrada"])
    years = pd.read_csv(rp / "replay_por_anio.csv")
    years = years[years.modo == "delta"].pivot(index="anio", columns="modelo", values="pnl")
    d = res[res.modo == "delta"].set_index("modelo")
    s = res[res.modo == "sin_cobertura"].set_index("modelo")
    means = closed.groupby("modelo")[["edge", "pnl_al_mid", "pnl"]].mean()
    reasons = closed.groupby("motivo").pnl.agg(["count", "mean"])
    ba = closed[closed.subyacente == "BA"]
    events = pd.read_csv(rp / "replay_eventos_delta.csv")
    out = {
        "replay ruedas": f"{miles(datos['ruedas'])} ruedas",
        "replay cotizaciones": f"{miles(datos['filas'])}\ncotizaciones",
        "replay explotables": f"el {pct(datos['supervivencia'])} pasa los filtros",
        "replay spread": f"mediano {dec(100 * datos['spread_rel_mediano'], 1)}%",
        "replay órdenes": f"| Órdenes publicadas | {miles(d.loc['crr', 'ordenes'])} | {miles(d.loc['bsm', 'ordenes'])} |",
        "replay cerradas": f"| Llenadas y cerradas | {d.loc['crr', 'cerradas']} | {d.loc['bsm', 'cerradas']} |",
        "replay acierto": f"| Con ganancia | {pct(d.loc['crr', 'acierto'])} | {pct(d.loc['bsm', 'acierto'])} |",
        "replay pnl": f"| P&L realizado | USD {miles(d.loc['crr', 'pnl_total'])} | USD {miles(d.loc['bsm', 'pnl_total'])} |",
        "replay media": (f"| P&L medio por estructura (EE) | USD {miles(d.loc['crr', 'pnl_medio'])} "
                         f"({round(d.loc['crr', 'error_estandar'])}) | USD {miles(d.loc['bsm', 'pnl_medio'])} "
                         f"({round(d.loc['bsm', 'error_estandar'])}) |"),
        "replay valor final": f"| Valor final al mid, con la cobertura | USD {miles(d.loc['crr', 'valor_final_mid'])} | USD {miles(d.loc['bsm', 'valor_final_mid'])} |",
        "replay costo cobertura": f"| Costo de la cobertura en siete años | USD {round(d.loc['crr', 'costo_cobertura'])} | USD {round(d.loc['bsm', 'costo_cobertura'])} |",
        "replay sin cobertura": f"USD {miles(s.loc['crr', 'valor_final_mid'])} para el binomial y USD {miles(s.loc['bsm', 'valor_final_mid'])} para Black-Scholes",
        "replay edge": f"| Edge prometido al publicar | USD {dec(means.loc['crr', 'edge'], 1)} | USD {dec(means.loc['bsm', 'edge'], 1)} |",
        "replay al mid": f"| Resultado si se saliera al mid | USD {dec(means.loc['crr', 'pnl_al_mid'], 1)} | USD {dec(means.loc['bsm', 'pnl_al_mid'], 1)} |",
        "replay real": f"| Resultado real, pagando el spread de salida | USD {dec(means.loc['crr', 'pnl'], 1)} | USD {dec(means.loc['bsm', 'pnl'], 1)} |",
        "replay llenado": f"({len(closed)} de {miles(len(trades))})",
        "replay BA": f"{len(ba)} de las {len(closed)} estructuras\n(USD {miles(ba.pnl.sum())})",
        "replay take-profit": f"{reasons.loc['convergencia', 'count']} estructuras cierran por take-profit con USD {round(reasons.loc['convergencia', 'mean'])}",
        "replay pre-vencimiento": f"{reasons.loc['pre_vencimiento', 'count']} llegan a 2 días del vencimiento sin converger y pierden USD {-round(reasons.loc['pre_vencimiento', 'mean'])}",
        "replay kill": f"{reasons.loc['kill_switch', 'count']} tocan el kill-switch y pierden USD {-round(reasons.loc['kill_switch', 'mean'])}",
        "replay cortes": f"se activó {(events.evento == 'perdida_diaria').sum()} veces",
        "replay M4": f"= {tex(m4.loc['delta', 'diferencia'])}$ con".replace("−", "-"),
        "replay M4 EE": f"= {tex(m4.loc['delta', 'error_estandar'])}$",
        "replay M4 z": f"$z = {tex(m4.loc['delta', 'z'], 2)}$".replace("−", "-"),
        "replay M4 sin cobertura": f"Sin cobertura tampoco ($z = {tex(m4.loc['sin_cobertura', 'z'], 2)}$)".replace("−", "-"),
        "replay años crr": "USD " + ", ".join(miles(-years.loc[y, 'crr']) + f" ({y})" for y in range(2021, 2025)) + f", {miles(-years.loc[2025, 'crr'])} (2025) y {miles(-years.loc[2026, 'crr'])}\n(2026)",
        "replay años bsm": (f"USD {miles(-years.loc[2020, 'bsm'])} (2020), " + ", ".join(miles(-years.loc[y, 'bsm']) for y in range(2021, 2025))
                            + f", {miles(-years.loc[2025, 'bsm'])} y\n{miles(-years.loc[2026, 'bsm'])}"),
    }
    return {k: v.replace("\\n", " ").replace("\n", " ") for k, v in out.items()}


def main() -> int:
    text = (ROOT / "informe_proyecto.md").read_text(encoding="utf-8")
    numbers = {**expected(), **expected_part2(), **expected_replay()}
    # Las líneas del informe se cortan en cualquier espacio: se compara sin saltos de línea.
    flat = " ".join(text.split())
    missing = {k: v for k, v in numbers.items() if " ".join(v.split()) not in flat}
    total = len(numbers)
    if missing:
        print(f"{len(missing)} de {total} números del informe no coinciden con los reportes:")
        for key, value in missing.items():
            print(f"  {key}: se esperaba «{value}»")
        return 1
    print(f"OK: los {total} números clave del informe coinciden con reportes_dolthub/ y reportes_replay/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
