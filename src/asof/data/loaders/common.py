"""Shared helpers for dataset loaders."""
from __future__ import annotations

import datetime as _dt
import html
import re
from pathlib import Path

_DATE_RE = re.compile(r"(20\d{2})[年\-/.\s]{0,2}(\d{1,2})[月\-/.\s]{0,2}(\d{1,2})")


def extract_first_date(text: str | None) -> _dt.date | None:
    """First yyyy-mm-dd-like date in free text (handles 中文 forms)."""
    m = _DATE_RE.search(text or "")
    if not m:
        return None
    try:
        return _dt.date(*(int(x) for x in m.groups()))
    except ValueError:
        return None


def date_from_filename(name: str) -> _dt.date | None:
    """Parse YYYYMMDD (then YYYY) embedded in a filename."""
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", name)
    if m:
        try:
            return _dt.date(*(int(x) for x in m.groups()))
        except ValueError:
            pass
    m = re.search(r"(20\d{2})", name)
    return _dt.date(int(m.group(1)), 1, 1) if m else None


def read_text(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="ignore")


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t　]+")
_NL = re.compile(r"\n{3,}")


def html_to_text(raw: str) -> str:
    """Very small HTML-to-text: drop script/style, strip tags, unescape."""
    raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?i)</p>", "\n", raw)
    txt = html.unescape(_TAG.sub("", raw))
    txt = _WS.sub(" ", txt)
    txt = _NL.sub("\n\n", txt)
    return txt.strip()


def load_text_any(path: str | Path) -> str:
    """Read .txt directly; strip tags for .html/.htm. PDFs/docs are skipped
    (their extracted .txt siblings carry the text in this dataset)."""
    p = Path(path)
    if p.suffix.lower() in (".html", ".htm"):
        return html_to_text(read_text(p))
    if p.suffix.lower() in (".txt", ".rst", ".md", ""):
        return read_text(p)
    return ""   # pdf/doc/docx: rely on sibling .txt in this corpus
