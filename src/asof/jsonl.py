"""Tiny JSONL/JSON helpers (orjson if available, else stdlib)."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

try:
    import orjson

    def _dumps(o: Any) -> bytes:
        return orjson.dumps(o, option=orjson.OPT_NON_STR_KEYS)

    def _loads(b: str | bytes) -> Any:
        return orjson.loads(b)
except Exception:  # pragma: no cover - fallback
    import json

    def _dumps(o: Any) -> bytes:
        return json.dumps(o, ensure_ascii=False).encode("utf-8")

    def _loads(b: str | bytes) -> Any:
        return json.loads(b)


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                yield _loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "wb") as f:
        for r in rows:
            f.write(_dumps(r))
            f.write(b"\n")
            n += 1
    return n


def read_json(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return _loads(f.read())


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(_dumps(obj))
