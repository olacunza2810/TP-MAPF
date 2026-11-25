"""El reporte de sesión cuenta sólo las órdenes y eventos de esa rueda."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ita_options.execution.ledger import Ledger
from ita_options.execution.live_runner import _on_session, write_session_report


def test_on_session_usa_hora_de_nueva_york() -> None:
    # 02:00 UTC del 26/11 son las 21:00 del 25/11 en Nueva York.
    assert _on_session("2026-11-26T02:00:00+00:00", date(2026, 11, 25))
    assert not _on_session("2026-11-26T02:00:00+00:00", date(2026, 11, 26))
    assert not _on_session(None, date(2026, 11, 25))


def test_reporte_no_acumula_ordenes_de_ruedas_previas(tmp_path: Path) -> None:
    ledger = Ledger()
    for i, stamp in enumerate(["2026-11-24T15:00:00+00:00", "2026-11-25T15:00:00+00:00",
                               "2026-11-25T16:00:00+00:00"]):
        ledger.record_order("s", "mleg_open", {"client_order_id": f"o{i}", "limit_price": 1.0})
        ledger._conn.execute("UPDATE orders SET submitted_at = ? WHERE client_order_id = ?",
                             (stamp, f"o{i}"))
    ledger.event("vieja")
    ledger._conn.execute("UPDATE events SET at = '2026-11-24T15:00:00+00:00'")
    ledger.event("nueva")
    ledger._conn.execute("UPDATE events SET at = '2026-11-25T15:00:00+00:00' WHERE kind = 'nueva'")

    target = write_session_report(ledger, date(2026, 11, 25), tmp_path)
    summary = json.loads((target / "summary.json").read_text("utf-8"))
    assert summary["orders"] == 2
    assert (target / "events.csv").read_text("utf-8").count("nueva") == 1
    assert "vieja" not in (target / "events.csv").read_text("utf-8")
