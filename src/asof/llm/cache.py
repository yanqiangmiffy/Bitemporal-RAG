"""SQLite cache for LLM replies and embeddings, keyed by the exact request.

With a populated cache the whole pipeline replays a paper run with no model
endpoint at all (mirrors the campus upstream's cache/llm_cache.sqlite).
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..jsonl import _dumps, _loads


def request_key(kind: str, payload: dict[str, Any]) -> str:
    blob = kind.encode() + b"\x00" + _dumps(payload)
    return hashlib.sha256(blob).hexdigest()


class Cache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, kind TEXT, v BLOB)"
        )
        self._conn.commit()

    def get(self, key: str) -> Any | None:
        blob = self.get_bytes(key)
        return _loads(blob) if blob is not None else None

    def get_bytes(self, key: str) -> bytes | None:
        with self._lock:
            row = self._conn.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, kind: str, value: Any) -> None:
        self.put_bytes_many(kind, [(key, _dumps(value))])

    def put_bytes_many(self, kind: str, entries: Iterable[tuple[str, bytes]]) -> None:
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO kv (k, kind, v) VALUES (?,?,?)",
                ((key, kind, blob) for key, blob in entries),
            )

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM kv").fetchone()[0]
