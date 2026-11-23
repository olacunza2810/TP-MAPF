"""Gráficos del resumen para la presentación (todos salen de los archivos del repo).

    python resumen_presentacion/hacer_graficos.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ita_options.greeks import bs_greeks, crr_price  # noqa: E402

OUT = ROOT / "resumen_presentacion" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
REP = ROOT / "reportes_dolthub"
RPL = ROOT / "reportes_replay"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5df"
CRR, BSM, AQUA, YEL, RED, VIO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#6250d6"
NEUTRAL = "#b9b8b1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "savefig.bbox": "tight", "savefig.dpi": 200,
})


def fmt_usd(x: float) -> str:
    s = f"{abs(x):,.0f}".replace(",", ".")
    return ("−" if x < 0 else "") + s


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("ok", name)


def bar_labels(ax, bars, fmt, inside=False, color=INK, dy=0.0):
    for b in bars:
        h = b.get_height()
        y = h + dy if h >= 0 else h - dy
        ax.annotate(fmt(h), (b.get_x() + b.get_width() / 2, y), ha="center",
                    va="bottom" if h >= 0 else "top", fontsize=8.5, color=color)


# 1. Convergencia del árbol (ejemplo de Hull con dividendo discreto)
def convergencia() -> None:
    S, K, r, sig, T = 52.0, 50.0, 0.10, 0.40, 5 / 12
    divs = ((3.5 / 12, 2.06),)
    ns = np.arange(5, 301, 5)
    am = [crr_price(S, K, T, r, sig, "p", divs, steps=int(n)) for n in ns]
    eu = [crr_price(S, K, T, r, sig, "p", divs, steps=int(n), american=False) for n in ns]
    bs = bs_greeks(S, K, T, r, sig, "p", divs).price
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.plot(ns, am, color=CRR, lw=2, label="Árbol CRR americano")
    ax.plot(ns, eu, color=BSM, lw=2, label="Árbol CRR europeo")
    ax.axhline(bs, color=INK, lw=1, ls="--", label=f"Black-Scholes (fórmula) = {bs:.3f}")
    ax.fill_between(ns, eu, am, color=CRR, alpha=0.10)
    ax.annotate("Prima de ejercicio anticipado\n(lo que vale poder ejercer antes)", (120, (am[23] + eu[23]) / 2),
                xytext=(130, 4.33), fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    ax.set_xlabel("Pasos del árbol (N)")
    ax.set_ylabel("Precio de la put (USD)")
    ax.set_title("El árbol converge a Black-Scholes y la versión americana vale más")
    ax.set_ylim(3.9, 4.48)
    ax.legend(loc="lower right", fontsize=8.5, ncol=2)
    save(fig, "f01_convergencia")


# 2. Qué hizo cada corrección con la paridad
def paridad() -> None:
    labels = ["Parte 1\n(yield continuo,\nspot del cierre)", "+ dividendo\ndiscreto", "+ banda de spot\n(desfase horario)",
              "Ejecutables en\nla rueda siguiente"]
    vals = [1744, 1737, 3, 0]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    bars = ax.bar(labels, vals, color=[NEUTRAL, NEUTRAL, CRR, AQUA], width=0.62)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:,}".replace(",", "."), (b.get_x() + b.get_width() / 2, v + 25), ha="center", fontsize=10,
                    fontweight="bold", color=INK)
    ax.set_ylabel("Violaciones de paridad detectadas")
    ax.set_title("Las 1.744 \"oportunidades\" de paridad eran un problema de datos")
    ax.set_ylim(0, 2000)
    save(fig, "f02_paridad")


# 3. Calibración: % de precios dentro del bid-ask
def calibracion() -> None:
    d = pd.read_csv(REP / "calibracion_rmse.csv")
    d = d[d.modelo == "crr"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
    for ax, (muestra, title) in zip(axes, [("in_sample", "Misma foto"), ("out_of_sample", "Rueda siguiente")]):
        g = d[d.muestra == muestra]
        x = np.arange(3)
        unds = ["BA", "LMT", "RTX"]
        sup = [g[(g.ajuste == "superficie") & (g.underlying == u)].pct_dentro_bid_ask.iloc[0] * 100 for u in unds]
        con = [g[(g.ajuste == "sigma_constante") & (g.underlying == u)].pct_dentro_bid_ask.iloc[0] * 100 for u in unds]
        b1 = ax.bar(x - 0.19, sup, 0.36, color=CRR, label="Sonrisa calibrada")
        b2 = ax.bar(x + 0.19, con, 0.36, color=NEUTRAL, label="Una sola volatilidad")
        bar_labels(ax, b1, lambda v: f"{v:.0f}%", dy=1)
        bar_labels(ax, b2, lambda v: f"{v:.0f}%", dy=1)
        ax.set_xticks(x, unds)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 108)
    axes[0].set_ylabel("% de precios dentro del bid-ask")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("La sonrisa reproduce la cadena del día, pero hay que recalibrarla cada día",
                 x=0.02, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.04)
    save(fig, "f03_calibracion")


# 4. Diebold-Mariano por grupo
def dm_grupos() -> None:
    d = pd.read_csv(REP / "comparacion_dm.csv")
    d = d[(d.comparacion == "bsm vs crr") & (d.perdida == "L")]
    rows = []
    tot = d[d.grupo == "total"].iloc[0]
    rows.append(("Total (36.515 contratos)", tot.dm))
    m = d[d.grupo == "option_type+dinero"]
    names = {"c": "Calls", "p": "Puts"}
    for _, r in m.iterrows():
        rows.append((f"{names[r.option_type]} {r.dinero}", r.dm))
    f = d[d.grupo == "option_type+fecha_ex"]
    for _, r in f.iterrows():
        rows.append((f"{names[r.option_type]}, {r.fecha_ex}", r.dm))
    rows = rows[:1] + sorted(rows[1:], key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    y = np.arange(len(rows))[::-1]
    vals = [v for _, v in rows]
    cols = [INK if i == 0 else (CRR if v > 1.96 else (BSM if v < -1.96 else NEUTRAL)) for i, (_, v) in enumerate(rows)]
    ax.barh(y, vals, color=cols, height=0.65)
    ax.set_yticks(y, [n for n, _ in rows], fontsize=8.3)
    ax.axvline(1.96, color=INK2, ls="--", lw=0.8)
    ax.axvline(-1.96, color=INK2, ls="--", lw=0.8)
    ax.text(2.1, y[-1] - 0.6, "significativo al 5%", fontsize=7.5, color=INK2)
    for yy, v in zip(y, vals):
        ax.annotate(f"{v:.1f}", (v + (0.25 if v >= 0 else -0.25), yy), va="center",
                    ha="left" if v >= 0 else "right", fontsize=7.8, color=INK)
    ax.set_xlabel("Estadístico DM* (positivo = el binomial erra menos)")
    ax.set_title("El binomial gana donde la teoría dice: puts ITM y fechas ex")
    ax.grid(axis="y", visible=False)
    save(fig, "f04_dm_grupos")


# 5. Efectividad de las coberturas
def coberturas() -> None:
    d = pd.read_csv(REP / "cobertura_efectividad.csv")
    d = d[d.modelo == "crr"].set_index("cobertura")
    order = ["delta", "delta_gamma", "delta_gamma_entera", "delta_vega"]
    labels = ["Delta\n(acciones)", "Delta-gamma", "Delta-gamma\nentera", "Delta-vega"]
    costo = ["USD 3", "USD 133", "USD 154", "USD 188"]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    b1 = ax.bar(x - 0.19, d.loc[order, "efectividad"] * 100, 0.36, color=NEUTRAL, label="Antes de costos")
    b2 = ax.bar(x + 0.19, d.loc[order, "efectividad_neta"] * 100, 0.36, color=AQUA, label="Después de costos")
    bar_labels(ax, b1, lambda v: f"{v:.0f}%", dy=1)
    bar_labels(ax, b2, lambda v: f"{v:.0f}%", dy=1)
    for xi, c in zip(x, costo):
        ax.text(xi, 6, f"costo por\nposición\n{c}", ha="center", fontsize=7.6, color=INK, va="bottom")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 125)
    ax.set_ylabel("Reducción de la varianza diaria (%)")
    ax.set_title("Cubrir la delta con acciones: 95% del riesgo por USD 3")
    ax.legend(loc="upper center", fontsize=8.5, ncol=2)
    save(fig, "f05_coberturas")


# 6. Estrés
def estres() -> None:
    d = pd.read_csv(REP / "estres_resultados.csv")
    d = d[d.tipo == "monte_carlo"]
    esc = ["gbm", "heston", "heston_estres", "merton", "merton_sectorial", "liquidez"]
    names = ["GBM", "Heston", "Heston\nestresado", "Merton", "Merton\nsectorial", "Spreads x3"]
    x = np.arange(len(esc))
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for i, (cob, col, lab) in enumerate([("sin_cobertura", NEUTRAL, "Sin cobertura"), ("delta", AQUA, "Delta"),
                                          ("delta_gamma", VIO, "Delta-gamma")]):
        v = [d[(d.escenario == e) & (d.cobertura == cob)].es99_mid.iloc[0] for e in esc]
        ax.bar(x + (i - 1) * 0.26, v, 0.25, color=col, label=lab)
    ax.set_xticks(x, names, fontsize=8.5)
    ax.set_ylabel("ES 99% a 10 ruedas (USD)")
    ax.set_title("Pérdida esperada en el peor 1% de los escenarios")
    ax.legend(fontsize=8.5, ncol=3, loc="upper left")
    ax.set_ylim(0, 2100)
    save(fig, "f06_estres")


# --- Simulación histórica ---------------------------------------------------
def replay_figs() -> None:
    daily = pd.read_csv(RPL / "replay_diario_delta.csv", parse_dates=["fecha"])
    trades = pd.read_csv(RPL / "replay_estructuras_delta.csv")
    closed = trades.dropna(subset=["cerrada"]).copy()
    closed["anio"] = pd.to_datetime(closed["cerrada"]).dt.year

    # 7. Curva de P&L
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot(daily.fecha, daily.valor_crr, color=CRR, lw=1.8, label="Binomial (CRR)")
    ax.plot(daily.fecha, daily.valor_bsm, color=BSM, lw=1.8, label="Black-Scholes")
    ax.plot(daily.fecha, daily.valor, color=INK, lw=1.1, ls="--", label="Cuenta total")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.annotate(fmt_usd(daily.valor_crr.iloc[-1]), (daily.fecha.iloc[-1], daily.valor_crr.iloc[-1]), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=CRR, va="center")
    ax.annotate(fmt_usd(daily.valor_bsm.iloc[-1]), (daily.fecha.iloc[-1], daily.valor_bsm.iloc[-1]), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=BSM, va="center")
    ax.annotate(fmt_usd(daily.valor.iloc[-1]), (daily.fecha.iloc[-1], daily.valor.iloc[-1]), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=INK, va="center")
    covid = pd.Timestamp("2020-03-16")
    ax.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30"), color=RED, alpha=0.08)
    ax.text(covid, 400, "COVID", fontsize=7.5, color=RED, ha="center")
    ax.set_ylabel("P&L acumulado al mid (USD)")
    ax.set_title("Siete años del sistema completo: pierde de forma sostenida")
    ax.legend(loc="lower left", fontsize=8.5)
    save(fig, "f07_replay_equity")

    # 8. Por año
    piv = closed.groupby(["anio", "modelo"]).pnl.sum().unstack().fillna(0)
    x = np.arange(len(piv))
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    b1 = ax.bar(x - 0.19, piv["crr"], 0.36, color=CRR, label="Binomial (CRR)")
    b2 = ax.bar(x + 0.19, piv["bsm"], 0.36, color=BSM, label="Black-Scholes")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if h:
                ax.annotate(f"{h / 1000:.1f}k".replace(".", ",").replace("-", "−"),
                            (b.get_x() + b.get_width() / 2, h - 60), ha="center", va="top", fontsize=7.4, color=INK)
    ax.set_xticks(x, piv.index.astype(str))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylim(piv.min().min() * 1.25, 300)
    ax.set_ylabel("P&L realizado (USD)")
    ax.set_title("Ningún año positivo, con ninguno de los dos modelos")
    ax.legend(fontsize=8.5, loc="lower left")
    save(fig, "f08_replay_anios")

    # 9. Embudo
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    stages = ["Órdenes publicadas", "Llenadas", "Cerradas con ganancia"]
    for i, (m, col, lab) in enumerate([("crr", CRR, "Binomial"), ("bsm", BSM, "Black-Scholes")]):
        t = trades[trades.modelo == m]
        vals = [len(t), int(t.abierta.notna().sum()), int((t.pnl > 0).sum())]
        y = np.arange(3)[::-1] + (0.2 if i == 0 else -0.2)
        bars = ax.barh(y, vals, 0.38, color=col, label=lab)
        for yy, v in zip(y, vals):
            ax.annotate(f"{v:,}".replace(",", ".") + (f"  ({v / vals[0]:.1%})".replace(".", ",") if v != vals[0] else ""),
                        (v + 15, yy), va="center", fontsize=8, color=INK)
    ax.set_yticks(np.arange(3)[::-1], stages)
    ax.set_xlim(0, 1900)
    ax.set_title("De 100 órdenes, se llenan 6: y se llenan cuando no conviene")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "f09_replay_embudo")

    # 10. Dónde se va la plata
    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    cats = ["Edge prometido\npor el modelo", "Resultado si se\nsaliera al mid", "Resultado real\n(cruzando el spread)"]
    x = np.arange(3)
    for i, (m, col, lab) in enumerate([("crr", CRR, "Binomial"), ("bsm", BSM, "Black-Scholes")]):
        g = closed[closed.modelo == m]
        vals = [g.edge.mean(), g.pnl_al_mid.mean(), g.pnl.mean()]
        bars = ax.bar(x + (i - 0.5) * 0.36, vals, 0.34, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:+.0f}".replace("-", "−"), (b.get_x() + b.get_width() / 2, v + (3 if v >= 0 else -3)),
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, cats)
    ax.set_ylim(-125, 35)
    ax.set_ylabel("USD por estructura (promedio)")
    ax.set_title("Prometía +11 y termina en −100: selección adversa más spread")
    ax.legend(fontsize=8.5, loc="lower left")
    save(fig, "f10_replay_descomposicion")

    # 11. Motivos de cierre
    mot = closed.groupby("motivo").pnl.agg(["count", "mean"]).sort_values("count", ascending=False)
    names = {"pre_vencimiento": "Antes del\nvencimiento", "convergencia": "Take-profit\n(convergencia)",
             "kill_switch": "Kill-switch", "perdida_diaria": "Corte por\npérdida diaria", "vencimiento": "Vencimiento"}
    names = {k: v.replace("\n", " ") for k, v in names.items()}
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6), sharey=True)
    cols = [NEUTRAL, AQUA, RED, YEL, VIO][: len(mot)]
    y = np.arange(len(mot))[::-1]
    axes[0].barh(y, mot["count"], color=cols, height=0.62)
    for yy, v in zip(y, mot["count"]):
        axes[0].annotate(str(v), (v + 1.5, yy), va="center", fontsize=8.5)
    axes[0].set_yticks(y, [names[i] for i in mot.index])
    axes[0].set_xlim(0, 95)
    axes[0].set_title("Cuántas cierran por cada motivo", fontsize=10)
    axes[1].barh(y, mot["mean"], color=cols, height=0.62)
    for yy, v in zip(y, mot["mean"]):
        axes[1].annotate(f"{v:+.0f}".replace("-", "−"), (v + (8 if v >= 0 else -8), yy), va="center",
                         ha="left" if v >= 0 else "right", fontsize=8.5)
    axes[1].axvline(0, color=INK2, lw=0.8)
    axes[1].set_xlim(-470, 110)
    axes[1].set_title("P&L medio por motivo (USD)", fontsize=10)
    for ax in axes:
        ax.grid(axis="y", visible=False)
    save(fig, "f11_replay_motivos")

    # 12. Distribución
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    bins = np.arange(-700, 401, 50)
    ax.hist(closed[closed.modelo == "crr"].pnl.clip(-700, 400), bins=bins, color=CRR, alpha=0.75, label="Binomial")
    ax.hist(closed[closed.modelo == "bsm"].pnl.clip(-700, 400), bins=bins, color=BSM, alpha=0.6, label="Black-Scholes")
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_xlabel("P&L por estructura (USD)")
    ax.set_ylabel("Estructuras")
    ax.set_title("Ganancias chicas y frecuentes, pérdidas grandes")
    ax.legend(fontsize=8.5)
    save(fig, "f12_replay_distribucion")

    # 13. M4: medias con intervalo de 2 errores estándar
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    for i, (m, col, lab) in enumerate([("crr", CRR, "Binomial"), ("bsm", BSM, "Black-Scholes")]):
        g = closed[closed.modelo == m].pnl
        mu, se = g.mean(), g.std(ddof=1) / math.sqrt(len(g))
        ax.errorbar(mu, i, xerr=2 * se, fmt="o", color=col, ms=8, capsize=5, lw=2)
        ax.annotate(f"{lab}: {mu:.0f} ± {2 * se:.0f}  (n = {len(g)})".replace(": -", ": −"), (mu, i + 0.28), ha="center",
                    fontsize=8.5, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks([])
    ax.set_ylim(-0.6, 1.8)
    ax.set_xlim(-170, 20)
    ax.set_xlabel("P&L medio por estructura cerrada (USD), ± 2 errores estándar")
    ax.set_title("Los intervalos se superponen: no hay diferencia entre modelos")
    save(fig, "f13_replay_m4")


if __name__ == "__main__":
    convergencia()
    paridad()
    calibracion()
    dm_grupos()
    coberturas()
    estres()
    replay_figs()
