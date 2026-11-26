"""SAFE foreign-exchange circulars -> notice stream.

Builds notices from the FinTrace effectiveness notices, the located prior-document
texts, and the four main circular packages (safe_2020_14, safe_2023_08/28,
safe_2024_11). Dates are parsed from the text; the revision_sources.jsonl event
index is attached to notice metadata. Gold questions for this domain are pending
the paper's annotation protocol (clause-level before/after values), so the
question set is empty by default and reported as such in stats.json.
"""
from __future__ import annotations

from pathlib import Path

from ...jsonl import read_jsonl
from ...schema import Notice, Question
from ...temporal import POS_INF
from ...temporal import parse_date as D
from .common import extract_first_date, load_text_any

DOMAIN = "safe_fx"


def _notice(doc_id: str, path: Path, family: str, cited: list[str] | None = None) -> Notice | None:
    text = load_text_any(path)
    if not text.strip():
        return None
    pub = extract_first_date(text)
    return Notice(
        doc_id=doc_id, published=D(pub) if pub else POS_INF,
        title=doc_id, text=text, domain=DOMAIN, family=family,
        cited_ids=cited or [],
    )


def load(datasets_root: str | Path) -> tuple[list[Notice], list[Question]]:
    root = Path(datasets_root) / DOMAIN
    notices: list[Notice] = []

    # prior documents (old regulations that later get amended)
    for p in sorted((root / "prior_text").glob("*.txt")):
        n = _notice(f"prior-{p.stem}", p, "safe_prior")
        if n:
            notices.append(n)

    # FinTrace effectiveness notices
    for p in sorted((root / "fintrace").glob("*.txt")):
        n = _notice(f"fintrace-{p.stem}", p, "safe_fintrace")
        if n:
            notices.append(n)

    # main circular packages (notice.html carries the publication text)
    for pkg in sorted(root.glob("safe_*")):
        html = pkg / "notice.html"
        if html.exists():
            n = _notice(pkg.name, html, pkg.name)
            if n:
                notices.append(n)

    # Link the clause-level revision events (for later annotation) onto the
    # circular that introduces them, via the shared family id.
    rev = Path(datasets_root) / "revision_sources.jsonl"
    if rev.exists():
        events_by_family: dict[str, list[str]] = {}
        for r in read_jsonl(rev):
            if r.get("domain") == DOMAIN and r.get("family"):
                events_by_family.setdefault(r["family"], []).append(r["id"])
        for n in notices:
            fam = (n.family or "").replace("safe_", "SAFE-").replace("_", "-")
            if fam in events_by_family:
                n.clause_ids = events_by_family[fam]

    # Questions pending annotation -> empty for now (honest; see stats.json note).
    questions: list[Question] = []
    return notices, questions
