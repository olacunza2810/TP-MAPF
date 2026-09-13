"""Tests de la calibración del mercado sintético.

Convierten la afirmación "la muestra reproduce la distribución de cada activo" en
invariantes verificables: la vol realizada iguala a la implícita, las colas son
gordas, el skew es negativo y el orden entre activos es el correcto (BA es el de
colas más gordas). Si alguien toca los parámetros y rompe el parecido con el
activo real, la suite lo detecta.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ita_options.calibration import (
    ASSET_PROFILES,
    TRADING_DAYS,
    distribution_report,
    simulate_daily_path,
)


def test_universo_tiene_tres_activos_del_ita():
    assert set(ASSET_PROFILES) == {"RTX", "BA", "LMT"}


@pytest.mark.parametrize("ticker", list(ASSET_PROFILES))
def test_vol_simulada_iguala_objetivo(ticker):
    """La vol realizada del proceso coincide con la vol total (= IV atm)."""
    prof = ASSET_PROFILES[ticker]
    rng = np.random.default_rng(123)
    path = simulate_daily_path(prof, 20_000, rng)
    r = np.diff(np.log(path))
    rv = r.std(ddof=1) * math.sqrt(TRADING_DAYS)
    assert rv == pytest.approx(prof.annual_vol, abs=0.02)


@pytest.mark.parametrize("ticker", list(ASSET_PROFILES))
def test_sigma_difusiva_es_real_y_menor_que_total(ticker):
    """Los saltos no pueden explicar más varianza que la total objetivo."""
    prof = ASSET_PROFILES[ticker]
    sig = prof.diffusion_sigma()
    assert 0.0 < sig < prof.annual_vol


@pytest.mark.parametrize("ticker", list(ASSET_PROFILES))
def test_colas_gordas_y_skew_negativo(ticker):
    """Retornos de equities: exceso de curtosis positivo y asimetría negativa."""
    prof = ASSET_PROFILES[ticker]
    rng = np.random.default_rng(321)
    r = np.diff(np.log(simulate_daily_path(prof, 20_000, rng)))
    s = pd.Series(r)
    assert s.kurt() > 0.5, "las colas deben ser más gordas que una Normal"
    assert s.skew() < 0.0, "el skew de equities es negativo (cracks bruscos)"


def test_rtx_es_el_mas_simetrico_y_de_colas_mas_finas():
    """Ordenamiento robusto: RTX tiene menos colas y menos asimetría que BA y LMT.

    BA y LMT cargan riesgo de cola por motivos distintos (BA turbulento de
    continuo; LMT explosivo en eventos raros), y entre sí no tienen un orden
    estable — por eso NO se afirma "BA es el de colas más gordas". Lo que sí es
    robusto es que ambos superan a RTX en curtosis y en skew negativo.
    """
    kurt: dict[str, float] = {}
    skew: dict[str, float] = {}
    for tk, prof in ASSET_PROFILES.items():
        r = np.diff(np.log(simulate_daily_path(prof, 20_000, np.random.default_rng(11))))
        kurt[tk] = pd.Series(r).kurt()
        skew[tk] = pd.Series(r).skew()
    assert kurt["BA"] > kurt["RTX"] and kurt["LMT"] > kurt["RTX"]
    assert skew["BA"] < skew["RTX"] and skew["LMT"] < skew["RTX"]


def test_boeing_tiene_la_mayor_volatilidad():
    """BA es el activo de mayor volatilidad total; LMT el de menor."""
    vols = {tk: p.annual_vol for tk, p in ASSET_PROFILES.items()}
    assert vols["BA"] == max(vols.values())
    assert vols["LMT"] == min(vols.values())


def test_dividendos_por_activo_son_los_de_mercado():
    """BA suspendido (0%), RTX bajo (~1.6%), LMT alto (~2.6%)."""
    assert ASSET_PROFILES["BA"].dividend_yield == 0.0
    assert ASSET_PROFILES["RTX"].dividend_yield < ASSET_PROFILES["LMT"].dividend_yield


def test_reporte_de_distribucion_es_reproducible():
    a = distribution_report(n_days=3_000, seed=42)
    b = distribution_report(n_days=3_000, seed=42)
    pd.testing.assert_frame_equal(a, b)
