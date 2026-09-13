"""Validación de la muestra sintética contra la distribución de cada activo real.

Genera la evidencia que responde la objeción del profesor:

1. **Coherencia interna** — la vol realizada del camino iguala a la IV con la que
   se valúan las opciones (antes no lo hacía).
2. **Momentos** — vol anualizada, skew y exceso de curtosis simulados vs. el
   objetivo calibrado de cada activo.
3. **Figura** — histograma de retornos diarios vs. Normal y Q-Q plot por activo,
   que muestra las colas gordas y la asimetría negativa de un vistazo.

Uso:

    python validate_sample.py            # imprime tablas y guarda la figura
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ita_options.calibration import (
    ASSET_PROFILES,
    distribution_report,
    simulate_daily_path,
)

OUT_DIR = Path("outputs")


def _print_moment_table() -> pd.DataFrame:
    df = distribution_report()  # defaults: 24 réplicas de 2.520 días, seed 7
    show = df.copy()
    for c in show.columns:
        if show[c].dtype == float:
            show[c] = show[c].round(3)
    print("\n=== Distribución diaria: objetivo (activo real) vs simulado ===")
    print(show.to_string(index=False))
    return df


def _make_figure() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    tickers = list(ASSET_PROFILES)
    moments = distribution_report().set_index("activo")
    fig, axes = plt.subplots(len(tickers), 2, figsize=(11, 3.1 * len(tickers)))
    fig.suptitle(
        "Muestra sintética vs. Normal — retornos diarios calibrados por activo",
        fontsize=13, fontweight="bold",
    )

    for row, tk in enumerate(tickers):
        prof = ASSET_PROFILES[tk]
        rng = np.random.default_rng(7 + row)
        r = np.diff(np.log(simulate_daily_path(prof, 5040, rng)))
        mu, sd = r.mean(), r.std(ddof=1)
        # Momentos promediados (tabla), no los de esta única realización.
        exk = float(moments.loc[tk, "curtosis_exceso_simulada"])
        skw = float(moments.loc[tk, "skew_simulado"])
        vol = float(moments.loc[tk, "vol_simulada"])

        # Histograma vs Normal ajustada
        ax = axes[row, 0]
        ax.hist(r, bins=80, density=True, alpha=0.65, color="#3b6ea5")
        xs = np.linspace(r.min(), r.max(), 400)
        ax.plot(xs, stats.norm.pdf(xs, mu, sd), "r--", lw=1.4, label="Normal")
        ax.set_yscale("log")  # log en Y hace visibles las colas
        ax.set_title(
            f"{tk}: vol {vol:.0%}  skew {skw:+.2f}  curt.exc {exk:+.1f}  (promedios)",
            fontsize=10,
        )
        ax.set_xlabel("retorno diario (log)")
        ax.set_ylabel("densidad (log)")
        ax.legend(fontsize=8)

        # Q-Q plot contra la Normal
        ax = axes[row, 1]
        stats.probplot(r, dist="norm", plot=ax)
        ax.get_lines()[0].set_markersize(2.5)
        ax.get_lines()[0].set_color("#3b6ea5")
        ax.get_lines()[1].set_color("red")
        ax.set_title(f"{tk}: Q-Q vs Normal (colas fuera de la recta = colas gordas)",
                     fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "validacion_distribucion.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    print("=" * 72)
    print("  VALIDACIÓN DE LA MUESTRA SINTÉTICA — RTX / BA / LMT (ITA)")
    print("=" * 72)

    print("\n=== Coherencia: vol difusiva vs. IV atm con la que se valúa ===")
    print(f"  {'activo':>7} {'sigma_difusiva':>15} {'vol_total(=IV atm)':>20} {'div':>6}")
    for tk, prof in ASSET_PROFILES.items():
        print(f"  {tk:>7} {prof.diffusion_sigma():>14.1%} "
              f"{prof.annual_vol:>19.0%} {prof.dividend_yield:>6.1%}")

    _print_moment_table()
    path = _make_figure()
    print(f"\nFigura guardada en: {path.resolve()}")


if __name__ == "__main__":
    main()
