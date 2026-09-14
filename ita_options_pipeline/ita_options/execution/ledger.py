"""Registro persistente de estrategias, órdenes y eventos, y reconciliación.

El ledger es la memoria del piloto. Una estrategia agrupa todas sus patas
(opciones y, en la paridad, acciones) y pasa por estos estados:

``pending`` → ``open`` → ``closing`` → ``closed``

o termina antes en ``rejected``, ``rejected_pretrade``, ``aborted``,
``edge_gone``, ``hedge_failed``, ``unfilled`` o ``dry_run``.

Cada pata guarda sus **unidades por estructura** con signo: el ratio entero de
la ``mleg`` para las opciones y ±100 acciones para la pata de acción. Con eso,
la posición que el ledger espera ver en la cuenta es la suma de
``unidades × estructuras abiertas`` por símbolo, y :func:`reconcile` la compara
contra las posiciones reales del broker.
"""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .orders import integer_ratios

__all__ = [
    "OPEN_STATES",
    "Ledger",
    "ReconciliationReport",
    "reconcile",
    "structure_key",
    "legs_with_units",
]

OPEN_STATES = ("open", "closing")

_STRATEGY_COLUMNS = {
    "state", "contracts_open", "residual_edge", "entry_cash", "exit_cash",
    "realized_pnl", "exit_reason", "message", "closed_at",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id TEXT PRIMARY KEY,
    structure_key TEXT NOT NULL,
    detector TEXT,
    underlying TEXT,
    route TEXT,
    state TEXT NOT NULL,
    contracts_requested INTEGER,
    contracts_open INTEGER DEFAULT 0,
    predicted_edge REAL,
    residual_edge REAL,
    entry_cash REAL DEFAULT 0,
    exit_cash REAL DEFAULT 0,
    realized_pnl REAL,
    exit_reason TEXT,
    message TEXT,
    legs_json TEXT NOT NULL,
    signal_at TEXT,
    session_date TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    closed_at TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    client_order_id TEXT PRIMARY KEY,
    broker_id TEXT,
    strategy_id TEXT,
    role TEXT,
    status TEXT,
    qty REAL,
    filled_qty REAL,
    filled_avg_price REAL,
    request_json TEXT,
    error TEXT,
    submitted_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    strategy_id TEXT,
    kind TEXT NOT NULL,
    payload_json TEXT
);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    def default(obj: Any) -> Any:
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return str(obj)

    return json.dumps(value, default=default)


def _clean(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def structure_key(signal: Mapping[str, Any]) -> str:
    """Identidad de una estructura: detector y símbolos de todas las patas."""
    symbols = sorted(str(leg["symbol"]) for leg in signal["leg_spec"])
    return f"{signal['detector']}|{'|'.join(symbols)}"


def legs_with_units(leg_spec: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Patas serializables con sus unidades por estructura.

    Las opciones llevan el ratio entero con signo que se envía en la ``mleg``;
    la acción, ±100 acciones por unidad.
    """
    legs = [dict(leg) for leg in leg_spec]
    option_legs = [leg for leg in legs if leg.get("kind") == "option"]
    ratios = integer_ratios([float(leg["qty"]) for leg in option_legs]) if option_legs else []
    ratio_iter = iter(ratios)
    out = []
    for leg in legs:
        units = (
            next(ratio_iter) if leg.get("kind") == "option"
            else int(round(math.copysign(100, float(leg["qty"]))))
        )
        out.append({
            "kind": leg.get("kind"),
            "symbol": str(leg["symbol"]),
            "qty": float(leg["qty"]),
            "price": float(leg["price"]),
            "strike": _clean(float(leg["strike"])) if leg.get("strike") is not None else None,
            "option_type": leg.get("option_type", ""),
            "expiration": (
                None if leg.get("expiration") is None or str(leg.get("expiration")) == "NaT"
                else str(getattr(leg["expiration"], "date", lambda: leg["expiration"])())
            ),
            "units": int(units),
        })
    return out


class Ledger:
    """Ledger SQLite. ``path=':memory:'`` sirve para tests y dry-runs."""

    def __init__(self, path: Path | str = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- estrategias --------------------------------------------------------

    def create_strategy(
        self, signal: Mapping[str, Any], route: str, contracts: int, session_date: date
    ) -> str:
        """Registra una estrategia en estado ``pending`` y devuelve su id."""
        strategy_id = uuid.uuid4().hex
        now = _utcnow()
        self._conn.execute(
            "INSERT INTO strategies (id, structure_key, detector, underlying, route, state,"
            " contracts_requested, predicted_edge, legs_json, signal_at, session_date,"
            " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                strategy_id, structure_key(signal), str(signal.get("detector")),
                str(signal.get("underlying")), route, "pending", int(contracts),
                _clean(float(signal.get("net_edge_usd", float("nan")))),
                _json(legs_with_units(signal["leg_spec"])),
                str(signal.get("timestamp")), session_date.isoformat(), now, now,
            ),
        )
        self._conn.commit()
        return strategy_id

    def update_strategy(self, strategy_id: str, **fields: Any) -> None:
        """Actualiza columnas permitidas de una estrategia."""
        unknown = set(fields) - _STRATEGY_COLUMNS
        if unknown:
            raise ValueError(f"Columnas no actualizables: {sorted(unknown)}")
        if not fields:
            return
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values = [_clean(v) for v in fields.values()]
        self._conn.execute(
            f"UPDATE strategies SET {assignments}, updated_at = ? WHERE id = ?",
            (*values, _utcnow(), strategy_id),
        )
        self._conn.commit()

    def _row_to_strategy(self, row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["legs"] = json.loads(record.pop("legs_json"))
        return record

    def strategy(self, strategy_id: str) -> dict[str, Any]:
        row = self._conn.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
        if row is None:
            raise KeyError(strategy_id)
        return self._row_to_strategy(row)

    def strategies(
        self, states: Iterable[str] | None = None, session_date: date | None = None
    ) -> list[dict[str, Any]]:
        query, params = "SELECT * FROM strategies WHERE 1=1", []
        if states is not None:
            states = list(states)
            query += f" AND state IN ({','.join('?' * len(states))})"
            params.extend(states)
        if session_date is not None:
            query += " AND session_date = ?"
            params.append(session_date.isoformat())
        rows = self._conn.execute(query + " ORDER BY created_at", params).fetchall()
        return [self._row_to_strategy(row) for row in rows]

    def open_strategies(self) -> list[dict[str, Any]]:
        return self.strategies(OPEN_STATES)

    def seen_structure(self, key: str, session_date: date) -> bool:
        """Si la estructura ya se intentó en la rueda o sigue abierta de antes."""
        row = self._conn.execute(
            "SELECT 1 FROM strategies WHERE structure_key = ? AND (session_date = ?"
            " OR state IN ('open', 'closing')) LIMIT 1",
            (key, session_date.isoformat()),
        ).fetchone()
        return row is not None

    def realized_pnl(self, session_date: date) -> float:
        """P&L realizado de las estrategias cerradas en la rueda."""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) FROM strategies"
            " WHERE state = 'closed' AND substr(closed_at, 1, 10) = ?",
            (session_date.isoformat(),),
        ).fetchone()
        return float(row[0])

    # -- órdenes y eventos --------------------------------------------------

    def record_order(
        self,
        strategy_id: str,
        role: str,
        request: Any,
        snapshot: Any | None = None,
        error: str | None = None,
    ) -> None:
        """Registra una orden enviada, con su respuesta o su rechazo."""
        payload = (
            request.model_dump(mode="json", exclude_none=True)
            if hasattr(request, "model_dump") else dict(request)
        )
        client_order_id = payload.get("client_order_id") or f"sin-id-{uuid.uuid4().hex[:12]}"
        now = _utcnow()
        self._conn.execute(
            "INSERT OR REPLACE INTO orders (client_order_id, broker_id, strategy_id, role,"
            " status, qty, filled_qty, filled_avg_price, request_json, error, submitted_at,"
            " updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                client_order_id,
                getattr(snapshot, "id", None),
                strategy_id,
                role,
                getattr(snapshot, "status", "rejected" if error else None),
                _clean(getattr(snapshot, "qty", payload.get("qty"))),
                _clean(getattr(snapshot, "filled_qty", 0.0)),
                _clean(getattr(snapshot, "filled_avg_price", None)),
                _json(payload),
                error,
                now,
                now,
            ),
        )
        self._conn.commit()

    def update_order(self, snapshot: Any) -> None:
        """Actualiza estado y fills de una orden conocida por su id de broker."""
        self._conn.execute(
            "UPDATE orders SET status = ?, filled_qty = ?, filled_avg_price = ?, updated_at = ?"
            " WHERE broker_id = ?",
            (snapshot.status, _clean(snapshot.filled_qty), _clean(snapshot.filled_avg_price),
             _utcnow(), snapshot.id),
        )
        self._conn.commit()

    def orders(self, strategy_id: str | None = None) -> list[dict[str, Any]]:
        if strategy_id is None:
            rows = self._conn.execute("SELECT * FROM orders ORDER BY submitted_at").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM orders WHERE strategy_id = ? ORDER BY submitted_at", (strategy_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def event(self, kind: str, payload: Any = None, strategy_id: str | None = None) -> None:
        self._conn.execute(
            "INSERT INTO events (at, strategy_id, kind, payload_json) VALUES (?,?,?,?)",
            (_utcnow(), strategy_id, kind, _json(payload)),
        )
        self._conn.commit()

    def events(self, kind: str | None = None) -> list[dict[str, Any]]:
        if kind is None:
            rows = self._conn.execute("SELECT * FROM events ORDER BY id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE kind = ? ORDER BY id", (kind,)
            ).fetchall()
        return [dict(row) for row in rows]

    # -- posiciones ---------------------------------------------------------

    def expected_positions(self) -> dict[str, float]:
        """Posición por símbolo que deberían tener las estrategias abiertas."""
        expected: dict[str, float] = {}
        for strategy in self.open_strategies():
            for leg in strategy["legs"]:
                amount = float(leg["units"]) * int(strategy["contracts_open"] or 0)
                expected[leg["symbol"]] = expected.get(leg["symbol"], 0.0) + amount
        return {symbol: qty for symbol, qty in expected.items() if qty != 0}

    def close(self) -> None:
        self._conn.close()


@dataclass
class ReconciliationReport:
    """Diferencias entre la posición esperada por el ledger y la del broker."""

    expected: dict[str, float]
    actual: dict[str, float]
    diffs: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.diffs


def reconcile(
    ledger: Ledger, broker: Any, roots: Iterable[str] | None = None, tolerance: float = 1e-6
) -> ReconciliationReport:
    """Compara posiciones esperadas y reales.

    Args:
        ledger: Ledger con las estrategias abiertas.
        broker: Broker con ``positions()``.
        roots: Si se pasan, las posiciones del broker cuyo símbolo no empieza
            por alguno de estos tickers se ignoran (otras operaciones de la
            cuenta). Las esperadas por el ledger siempre se comparan.
        tolerance: Diferencia admitida.
    """
    expected = ledger.expected_positions()
    actual = {s: q for s, q in broker.positions().items() if abs(q) > tolerance}
    roots = tuple(roots) if roots is not None else None
    symbols = set(expected) | {
        s for s in actual if roots is None or any(s.startswith(r) for r in roots)
    }
    diffs = {
        s: (expected.get(s, 0.0), actual.get(s, 0.0))
        for s in sorted(symbols)
        if abs(expected.get(s, 0.0) - actual.get(s, 0.0)) > tolerance
    }
    return ReconciliationReport(expected, actual, diffs)
