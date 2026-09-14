r"""Pruebas de la fase 0 contra la cuenta **paper** de Alpaca.

Responde las preguntas que definen la capa de órdenes antes de programarla:

1. **Convención de signo** del ``limit_price`` en una orden ``mleg``: se envía
   un vertical de débito (compra del call ATM, venta del strike siguiente) con
   ``limit_price`` igual al ancho del spread. Si llena, el precio positivo es
   débito; si no, se prueba con el ancho negativo.
2. **Conversión**: con 100 acciones compradas, ¿acepta una ``mleg`` que vende
   el call y compra la put del mismo strike? El call corto debería contar como
   cubierto.
3. **Conversión reversa**: con 100 acciones en corto, ¿acepta una ``mleg`` que
   compra el call y vende la put?

En 2 y 3 la ``mleg`` se envía con un precio que no puede llenar (pide un
crédito imposible): sólo interesa si Alpaca la **acepta o la rechaza**.

Seguridad:

- sólo cuenta paper: se aborta si ``ALPACA_PAPER=false``;
- sin ``--submit`` sólo muestra lo que enviaría;
- exige mercado abierto para enviar;
- al terminar cancela las órdenes que creó y cierra las posiciones que abrió.

**No probado todavía contra la API real**: correrlo primero sin ``--submit``.

Uso::

    $env:ALPACA_API_KEY="..."; $env:ALPACA_SECRET_KEY="..."
    python scripts/paper_probe.py --underlying RTX            # plan, sin enviar
    python scripts/paper_probe.py --underlying RTX --submit   # ejecuta en paper
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpaca.data.historical.stock import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockLatestQuoteRequest  # noqa: E402
from alpaca.trading.client import TradingClient  # noqa: E402
from alpaca.trading.enums import ContractType, OrderSide, TimeInForce  # noqa: E402
from alpaca.trading.requests import (  # noqa: E402
    GetOptionContractsRequest,
    MarketOrderRequest,
)

from ita_options.doctor import evaluate_account  # noqa: E402
from ita_options.execution.orders import build_mleg_order, strategy_order_id  # noqa: E402

FILLED = {"filled", "partially_filled"}
FINAL = {"filled", "canceled", "expired", "rejected", "done_for_day"}


class Probe:
    """Ejecuta y registra las pruebas, limpiando todo lo que crea."""

    def __init__(self, client: TradingClient, submit: bool) -> None:
        self.client = client
        self.submit = submit
        self.log: list[dict[str, Any]] = []
        self.created_orders: list[str] = []

    def record(self, step: str, **fields: Any) -> None:
        """Agrega un paso al registro y lo imprime."""
        entry = {"step": step, "at": datetime.now(timezone.utc).isoformat(), **fields}
        self.log.append(entry)
        print(f"[{step}] " + ", ".join(f"{k}={v}" for k, v in fields.items()))

    def send(self, step: str, request: Any) -> Any | None:
        """Envía un request (o lo muestra en modo plan) y registra la respuesta."""
        summary = request.model_dump(exclude_none=True, mode="json")
        if not self.submit:
            self.record(step, modo="plan", request=summary)
            return None
        try:
            order = self.client.submit_order(request)
        except Exception as exc:  # noqa: BLE001 - la respuesta del broker es el dato
            self.record(step, resultado="rechazada", error=str(exc)[:500], request=summary)
            return None
        self.created_orders.append(str(order.id))
        self.record(step, resultado="aceptada", id=str(order.id),
                    estado=str(order.status.value))
        return order

    def wait(self, order: Any, seconds: float) -> str:
        """Espera hasta ``seconds`` a que la orden llene por completo o termine.

        ``partially_filled`` no corta la espera: la primera corrida en paper
        envió la ``mleg`` con la compra de acciones todavía parcial y Alpaca la
        rechazó por descubierta.
        """
        deadline = time.monotonic() + seconds
        status = str(order.status.value)
        while time.monotonic() < deadline and status not in FINAL:
            time.sleep(1.0)
            status = str(self.client.get_order_by_id(order.id).status.value)
        return status

    def cancel(self, order: Any | None) -> None:
        """Cancela una orden creada por la prueba, si sigue abierta."""
        if order is None or not self.submit:
            return
        try:
            self.client.cancel_order_by_id(order.id)
        except Exception as exc:  # noqa: BLE001 - puede estar llena o cancelada
            self.record("cancelar", id=str(order.id), aviso=str(exc)[:200])

    def close(self, symbols: list[str]) -> None:
        """Cierra las posiciones abiertas por la prueba, las cortas primero.

        Cerrar primero la pata larga de un spread dejaría la corta descubierta,
        que una cuenta de nivel 3 no puede sostener.
        """
        if not self.submit:
            return
        positions = {p.symbol: p for p in self.client.get_all_positions()}
        ordered = sorted(
            (s for s in symbols if s in positions),
            key=lambda s: float(positions[s].qty) >= 0,
        )
        for symbol in ordered:
            try:
                self.client.close_position(symbol)
                self.record("cerrar_posicion", simbolo=symbol)
            except Exception as exc:  # noqa: BLE001
                self.record("cerrar_posicion", simbolo=symbol, error=str(exc)[:300])
            time.sleep(1.0)


def _leg(symbol: str, qty: float, strike: float, option_type: str,
         expiration: date) -> dict[str, Any]:
    return {"kind": "option", "symbol": symbol, "qty": qty, "price": 0.0,
            "strike": strike, "option_type": option_type, "expiration": expiration}


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada."""
    parser = argparse.ArgumentParser(
        description="Pruebas de fase 0 en la cuenta paper de Alpaca.")
    parser.add_argument("--underlying", default="RTX")
    parser.add_argument("--submit", action="store_true",
                        help="Enviar órdenes a la cuenta paper.")
    parser.add_argument("--out", type=Path, default=ROOT / "reportes")
    parser.add_argument("--fill-timeout", type=float, default=20.0)
    args = parser.parse_args(argv)

    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Faltan ALPACA_API_KEY / ALPACA_SECRET_KEY.")
    if os.environ.get("ALPACA_PAPER", "true").lower() == "false":
        raise SystemExit("Este script sólo corre contra la cuenta paper.")

    client = TradingClient(key, secret, paper=True)
    probe = Probe(client, args.submit)
    ticker = args.underlying.upper()
    symbols_touched: list[str] = [ticker]
    answers: dict[str, Any] = {}

    try:
        account = client.get_account()
        check = evaluate_account(account, paper=True)
        probe.record("cuenta", ok=check.passed, detalle=check.detail)
        clock = client.get_clock()
        probe.record("reloj", abierto=clock.is_open)
        if args.submit and not clock.is_open:
            raise SystemExit("Mercado cerrado: las pruebas con --submit necesitan fills.")

        quote = StockHistoricalDataClient(key, secret).get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=[ticker])
        )[ticker]
        spot = (float(quote.bid_price) + float(quote.ask_price)) / 2
        contracts = client.get_option_contracts(GetOptionContractsRequest(
            underlying_symbols=[ticker],
            expiration_date_gte=date.today() + timedelta(days=14),
            expiration_date_lte=date.today() + timedelta(days=60),
            limit=1000,
        )).option_contracts
        if not contracts:
            raise SystemExit(f"Sin contratos de {ticker} entre 14 y 60 días.")
        expiration = min(c.expiration_date for c in contracts)
        chain = [c for c in contracts if c.expiration_date == expiration]
        calls = {float(c.strike_price): c.symbol for c in chain if c.type == ContractType.CALL}
        puts = {float(c.strike_price): c.symbol for c in chain if c.type == ContractType.PUT}
        strikes = sorted(set(calls) & set(puts))
        atm = min(strikes, key=lambda k: abs(k - spot))
        upper = min((k for k in calls if k > atm), default=None)
        if upper is None:
            raise SystemExit("No hay strike por encima del ATM para el vertical.")
        width = upper - atm
        symbols_touched += [calls[atm], calls[upper], puts[atm]]
        probe.record("contratos", spot=round(spot, 2), vencimiento=str(expiration),
                     atm=atm, superior=upper, ancho=width)

        # 1. Convención de signo con un vertical de débito.
        vertical = [_leg(calls[atm], +1, atm, "c", expiration),
                    _leg(calls[upper], -1, upper, "c", expiration)]
        credit_sign: int | None = None
        for price in (+width, -width):
            order = probe.send(
                f"1_vertical_debito_limit_{price:+g}",
                build_mleg_order(vertical, 1, price, client_order_id=strategy_order_id(
                    "probe_vertical", [calls[atm], calls[upper]],
                    f"{time.time()}{price}", True)),
            )
            if order is None:
                continue
            status = probe.wait(order, args.fill_timeout)
            probe.record(f"1_estado_{price:+g}", estado=status)
            probe.cancel(order)
            probe.close([calls[atm], calls[upper]])
            if status in FILLED:
                # Llenó pagando: ese signo es débito y el opuesto es crédito.
                credit_sign = -1 if price > 0 else +1
                break
        answers["credit_limit_sign"] = credit_sign
        probe.record(
            "respuesta_1", CREDIT_LIMIT_SIGN=credit_sign,
            nota=("confirmar o actualizar ita_options/execution/orders.py"
                  if credit_sign else "sin determinar"),
        )

        # Precio que pide un crédito imposible: no debería llenar.
        impossible = (credit_sign or -1) * round(spot * 2, 2)

        # 2. Conversión: acción comprada, luego -C +P.
        stock = probe.send("2_comprar_100_acciones", MarketOrderRequest(
            symbol=ticker, qty=100, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
        stock_status = probe.wait(stock, args.fill_timeout) if stock is not None else None
        probe.record("2_estado_accion", estado=stock_status)
        conversion = [_leg(calls[atm], -1, atm, "c", expiration),
                      _leg(puts[atm], +1, atm, "p", expiration)]
        order = None
        if args.submit and stock_status != "filled":
            probe.record("2_mleg_omitida", motivo="las acciones no llenaron por completo")
            answers["conversion_mleg_aceptada"] = None
        else:
            order = probe.send("2_mleg_conversion", build_mleg_order(
                conversion, 1, impossible, client_order_id=strategy_order_id(
                    "probe_conversion", [calls[atm], puts[atm]], time.time(), True)))
            answers["conversion_mleg_aceptada"] = (order is not None) if args.submit else None
        probe.cancel(order)
        probe.close([calls[atm], puts[atm], ticker])

        # 3. Conversión reversa: acción en corto, luego +C -P.
        if not getattr(account, "shorting_enabled", False):
            probe.record("3_omitida", motivo="la cuenta no tiene short habilitado")
        else:
            short = probe.send("3_vender_corto_100_acciones", MarketOrderRequest(
                symbol=ticker, qty=100, side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY))
            short_status = probe.wait(short, args.fill_timeout) if short is not None else None
            probe.record("3_estado_accion", estado=short_status)
            reverse = [_leg(calls[atm], +1, atm, "c", expiration),
                       _leg(puts[atm], -1, atm, "p", expiration)]
            order = None
            if args.submit and short_status != "filled":
                probe.record("3_mleg_omitida", motivo="el corto no llenó por completo")
                answers["reversa_mleg_aceptada"] = None
            else:
                order = probe.send("3_mleg_reversa", build_mleg_order(
                    reverse, 1, impossible, client_order_id=strategy_order_id(
                        "probe_reverse", [calls[atm], puts[atm]], time.time(), True)))
                answers["reversa_mleg_aceptada"] = (order is not None) if args.submit else None
            probe.cancel(order)
            probe.close([calls[atm], puts[atm], ticker])
    finally:
        if args.submit:
            for order_id in probe.created_orders:
                try:
                    client.cancel_order_by_id(order_id)
                except Exception:  # noqa: BLE001 - ya estaba en estado final
                    pass
            probe.close(symbols_touched)
        args.out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = args.out / f"paper_probe_{stamp}.json"
        target.write_text(
            json.dumps({"submit": args.submit, "answers": answers, "log": probe.log},
                       indent=2, default=str),
            "utf-8",
        )
        print(f"\nRespuestas: {answers}\nRegistro: {target}")


if __name__ == "__main__":
    main()
