"""Tests de los detectores de arbitraje.

El test central es de ausencia: sobre una cadena generada por el propio modelo,
ningún detector debe encontrar nada. Es la prueba que fallan tres de los cuatro
detectores del notebook de cátedra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ita_options.arbitrage import (
    ExecutionCosts,
    calibrate_chain_sigma,
    detect_butterfly_convexity,
    run_all_detectors,
)

from conftest import SIGMA

COSTS = ExecutionCosts(min_net_edge=1.0)


def test_sin_falsos_positivos_en_cadena_limpia(multi_expiry_chain) -> None:
    """Cero oportunidades sobre precios libres de arbitraje por construcción."""
    assert run_all_detectors(multi_expiry_chain, COSTS).empty


def test_mariposa_del_notebook_da_falsos_positivos(clean_chain) -> None:
    """El test de convexidad original dispara sobre una cadena convexa.

    Reproduce la celda 41 del notebook para dejar documentada la diferencia:
    comprar el cuerpo y vender las alas da un portafolio de payoff no positivo,
    así que cobrar por armarlo nunca es arbitraje.
    """
    calls = (
        clean_chain.loc[clean_chain["option_type"] == "c"]
        .sort_values("strike")
        .reset_index(drop=True)
    )
    false_positives = 0
    for i in range(len(calls) - 2):
        k1, k2, k3 = calls.loc[i:i + 2, "strike"]
        weight = (k3 - k2) / (k3 - k1)
        cost = calls.loc[i + 1, "ask"] - (
            weight * calls.loc[i, "bid"] + (1 - weight) * calls.loc[i + 2, "bid"]
        )
        false_positives += cost < -1e-8

    assert false_positives == len(calls) - 2
    assert detect_butterfly_convexity(clean_chain, COSTS).empty


def test_detecta_violacion_de_convexidad_inyectada(clean_chain) -> None:
    """Encarecer el strike central rompe la convexidad y debe detectarse."""
    chain = clean_chain.copy()
    mask = (chain["option_type"] == "c") & (chain["strike"] == 150.0)
    chain.loc[mask, ["bid", "ask"]] += 3.0

    results = run_all_detectors(chain, COSTS)
    assert "butterfly" in set(results["detector"])
    assert (results["net_edge_usd"] > 0).all()


def test_cotizacion_en_cero_no_genera_oportunidad(clean_chain) -> None:
    """Un ask de cero es una cotización ausente, no un contrato gratis."""
    chain = clean_chain.copy()
    chain.loc[chain["strike"] == 155.0, "ask"] = 0.0
    assert run_all_detectors(chain, COSTS).empty


def test_filtro_de_liquidez_ausente_falla_fuerte(clean_chain) -> None:
    """Sin ``is_tradable``, pedir filtrado debe romper, no saltear en silencio."""
    with pytest.raises(KeyError, match="is_tradable"):
        run_all_detectors(clean_chain.drop(columns=["is_tradable"]), COSTS)


def test_calibracion_global_recupera_la_sigma(clean_chain) -> None:
    """La sigma que minimiza el error sobre la cadena es la que la generó."""
    sigma = calibrate_chain_sigma(clean_chain, steps=64, sigma_grid=(0.05, 0.60, 56))
    assert abs(sigma - SIGMA) < 0.02


def test_no_compara_contratos_de_subyacentes_distintos(clean_chain) -> None:
    """Dos cadenas limpias de acciones distintas, juntas, no generan señales.

    El mismo strike sobre acciones que cotizan a precios distintos representa
    moneyness distintos. Si los detectores agrupan sólo por vencimiento,
    comparan un call de BA contra uno de RTX y producen violaciones masivas que
    no existen. Este test cubre el caso que las fixtures de un solo subyacente
    dejaban pasar.

    La segunda cadena se construye escalando spot, strikes y primas por el mismo
    factor: el precio de una opción es homogéneo de grado uno en
    :math:`(S, K)`, así que la cadena escalada sigue libre de arbitraje.
    """
    factor = 1.45
    otra = clean_chain.copy()
    otra["underlying"] = "BA"
    for column in ("underlying_price", "strike", "bid", "ask", "mid_price"):
        otra[column] = (clean_chain[column] * factor).round(2)
    otra["symbol"] = otra["symbol"].str.replace("RTX", "BA", regex=False)

    assert run_all_detectors(otra, COSTS).empty, "la cadena escalada debe ser limpia"

    mezcla = pd.concat([clean_chain, otra], ignore_index=True)
    assert run_all_detectors(mezcla, COSTS).empty
