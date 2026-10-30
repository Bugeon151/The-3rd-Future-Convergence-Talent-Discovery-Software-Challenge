"""SQLite event store for safety events and outcomes.

Stores only event metadata (never raw video unless explicitly enabled) which
supports the privacy mode.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any


class EventStore:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
        self.db_path = db_path
        self.run_id = str(uuid.uuid4())[:8]
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    ts REAL,
                    t_sec REAL,
                    status TEXT,
                    kind TEXT,
                    subject_id INTEGER,
                    target_id INTEGER,
                    risk_initial REAL,
                    risk_final REAL,
                    recommended TEXT,
                    details TEXT,
                    origin TEXT
                )
                """
            )
            self._conn.commit()

    def add_event(
        self,
        status: str,
        kind: str,
        subject_id: int,
        target_id: int,
        risk_initial: float,
        t_sec: float,
        recommended: str = "",
        origin: str = "engine",
        details: dict[str, Any] | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (run_id, ts, t_sec, status, kind, subject_id, target_id, "
                "risk_initial, risk_final, recommended, details, origin) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    self.run_id,
                    time.time(),
                    float(t_sec),
                    status,
                    kind,
                    int(subject_id),
                    int(target_id),
                    float(risk_initial),
                    None,
                    recommended,
                    json.dumps(details or {}, ensure_ascii=False),
                    origin,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def update_event(self, event_id: int, status: str | None = None, risk_final: float | None = None,
                     recommended: str | None = None, details: dict[str, Any] | None = None) -> None:
        sets = []
        vals: list[Any] = []
        if status is not None:
            sets.append("status = ?")
            vals.append(status)
        if risk_final is not None:
            sets.append("risk_final = ?")
            vals.append(float(risk_final))
        if recommended is not None:
            sets.append("recommended = ?")
            vals.append(recommended)
        if details is not None:
            sets.append("details = ?")
            vals.append(json.dumps(details, ensure_ascii=False))
        if not sets:
            return
        vals.append(event_id)
        with self._lock:
            self._conn.execute(f"UPDATE events SET {', '.join(sets)} WHERE id = ?", vals)
            self._conn.commit()

    def query(self, status: str | None = None, limit: int = 200) -> list[dict]:
        sql = "SELECT * FROM events"
        params: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
            cols = [d[0] for d in self._conn.execute("SELECT * FROM events LIMIT 1").description]
        return [dict(zip(cols, r)) for r in rows]

    def summary(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute("SELECT status, COUNT(*) FROM events GROUP BY status").fetchall()
        return {s: int(c) for s, c in rows}

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM events")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()