"""SQLite alert store (CLAUDE.md §7.8)."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path

from src.common.schemas import AlertEvent, Evidence

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id   TEXT PRIMARY KEY,
    type       TEXT NOT NULL,
    level      INTEGER NOT NULL,
    global_id  INTEGER,
    zone       TEXT,
    t_start_ms INTEGER NOT NULL,
    t_raise_ms INTEGER NOT NULL,
    cams       TEXT NOT NULL,
    evidence   TEXT NOT NULL,
    acked      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_alerts_raise ON alerts (t_raise_ms);
"""


class AlertStore:
    def __init__(self, path: str | Path) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.executescript(_SCHEMA)
        self.lock = threading.Lock()  # live: snapshot worker writes evidence concurrently

    def add(self, a: AlertEvent) -> None:
        with self.lock:
            self._add(a)

    def set_evidence(self, alert_id: str, evidence: Evidence) -> bool:
        with self.lock:
            cur = self.conn.execute(
                "UPDATE alerts SET evidence = ? WHERE alert_id = ?",
                (evidence.model_dump_json(), alert_id),
            )
            self.conn.commit()
        return cur.rowcount == 1

    def _add(self, a: AlertEvent) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO alerts VALUES (?,?,?,?,?,?,?,?,?,0)",
            (
                a.alert_id,
                a.type.value,
                a.level,
                a.global_id,
                a.zone,
                a.t_start_ms,
                a.t_raise_ms,
                json.dumps(a.cams),
                a.evidence.model_dump_json(),
            ),
        )
        self.conn.commit()

    def add_many(self, alerts: Iterable[AlertEvent]) -> int:
        n = 0
        for a in alerts:
            self.add(a)
            n += 1
        return n

    def ack(self, alert_id: str) -> bool:
        with self.lock:
            cur = self.conn.execute("UPDATE alerts SET acked = 1 WHERE alert_id = ?", (alert_id,))
            self.conn.commit()
        return cur.rowcount == 1

    def recent(
        self, limit: int = 100, min_level: int = 1, unacked_only: bool = False
    ) -> list[dict]:
        q = "SELECT * FROM alerts WHERE level >= ?"
        if unacked_only:
            q += " AND acked = 0"
        q += " ORDER BY t_raise_ms DESC LIMIT ?"
        cur = self.conn.execute(q, (min_level, limit))
        cols = [c[0] for c in cur.description]
        rows = []
        for r in cur.fetchall():
            d = dict(zip(cols, r, strict=True))
            d["cams"] = json.loads(d["cams"])
            d["evidence"] = json.loads(d["evidence"])
            d["acked"] = bool(d["acked"])
            rows.append(d)
        return rows

    def close(self) -> None:
        self.conn.close()
