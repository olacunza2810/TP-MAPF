"""Tests del motor de valuación e inversión de volatilidad implícita."""

from __future__ import annotations

import math

import pytest

from ita_options.volatility import (
    bsm_price,
    crr_american_price,
    implied_volatility,
    no_arbitrage_bounds,
)

SPOT, STRIKE, TAU, RATE = 150.0, 155.0, 0.5, 0.0425


def test_crr_converge_al_europeo_sin_dividendos() -> None:
    """Sin dividendos, el call americano vale lo mismo que el europeo."""
    european = bsm_price(SPOT, STRIKE, TAU, RATE, 0.0, 0.28, "c")
    american = crr_american_price(SPOT, STRIKE, TAU, RATE, 0.0, 0.28, "c", 1024)
    assert abs(american - european) < 5e-3


def test_paridad_put_call_europea() -> None:
    r"""Verifica :math:`C - P = S e^{-q\tau} - K e^{-r\tau}`."""
    call = bsm_price(SPOT, STRIKE, TAU, RATE, 0.0, 0.28, "c")
    put = bsm_price(SPOT, STRIKE, TAU, RATE, 0.0, 0.28, "p")
    expected = SPOT - STRIKE * math.exp(-RATE * TAU)
    assert abs((call - put) - expected) < 1e-8


def test_put_americana_no_vale_menos_que_la_europea() -> None:
    """La prima de ejercicio anticipado no puede ser negativa."""
    european = bsm_price(SPOT, STRIKE, TAU, RATE, 0.0, 0.28, "p")
    american = crr_american_price(SPOT, STRIKE, TAU, RATE, 0.0, 0.28, "p", 1024)
    assert american >= european - 1e-8


@pytest.mark.parametrize("flag", ["c", "p"])
@pytest.mark.parametrize("sigma_true", [0.12, 0.28, 0.65])
def test_round_trip_de_volatilidad_implicita(flag: str, sigma_true: float) -> None:
    """Invertir el precio generado por una sigma devuelve esa misma sigma.

    El caso de sigma baja es el que rompe el clamp ``max(sigma, r)`` del
    notebook de cátedra: con la cota adaptativa del árbol CRR, converge.
    """
    price = crr_american_price(SPOT, STRIKE, TAU, RATE, 0.02, sigma_true, flag, 256)
    recovered = implied_volatility(
        price, SPOT, STRIKE, TAU, RATE, 0.02, flag, american=True, steps=256
    )
    assert abs(recovered - sigma_true) < 1e-4


def test_precio_fuera_de_cotas_devuelve_nan() -> None:
    """Un precio que viola no arbitraje no tiene IV: debe dar ``NaN``."""
    lower, _ = no_arbitrage_bounds(SPOT, STRIKE, TAU, RATE, 0.0, "c", True)
    assert math.isnan(
        implied_volatility(lower - 0.5, SPOT, STRIKE, TAU, RATE, 0.0, "c")
    )
