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


def main() -> int:
    text = (ROOT / "informe_proyecto.md").read_text(encoding="utf-8")
    missing = {k: v for k, v in expected().items() if v not in text}
    total = len(expected())
    if missing:
        print(f"{len(missing)} de {total} números del informe no coinciden con reportes_dolthub/:")
        for key, value in missing.items():
            print(f"  {key}: se esperaba «{value}»")
        return 1
    print(f"OK: los {total} números clave del informe coinciden con reportes_dolthub/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
