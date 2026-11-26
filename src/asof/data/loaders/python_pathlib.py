"""CPython pathlib docs across official releases -> version-chain notice stream,
plus a small English seed question set (verify before reporting).

Each release's pathlib.rst is a 'notice' published on the release date; the
field is a method signature. Unchanged controls (open, is_mount) give Unch
items; changed methods (is_file/is_dir gained follow_symlinks in 3.13; glob
gained recurse_symlinks in 3.13) give Post items.
"""
from __future__ import annotations

from pathlib import Path

from ...jsonl import read_jsonl
from ...schema import Condition, Notice, QType, Question
from ...temporal import parse_date as D
from .common import read_text

DOMAIN = "python_pathlib"


def _load_notices(root: Path, ext_root: Path) -> list[Notice]:
    rel = ext_root / "python_pathlib" / "releases.jsonl"
    notices: list[Notice] = []
    if not rel.exists():
        return notices
    for r in read_jsonl(rel):
        files = r.get("files", {})
        rst = files.get("pathlib.rst")
        # files are relative to the ext root; first-batch 3.12/3.13 live at datasets root
        text = ""
        for base in (ext_root, root.parent):
            if rst and (base / rst).exists():
                text = read_text(base / rst)
                break
        notices.append(Notice(
            doc_id=f"pathlib-{r['version']}", published=D(r["release_date"]),
            title=f"pathlib {r['version']}", text=text, domain=DOMAIN, family="pathlib",
        ))
    return notices


_SEED = [
    ("pl-open-unch", QType.UNCH, "2025-01-01", "2024-12-01", "No change",
     "pathlib-3.13.0", "Path.open signature is unchanged between 3.12 and 3.13",
     "Between pathlib 3.12 and 3.13, did Path.open change its signature?"),
    ("pl-ismount-unch", QType.UNCH, "2025-01-01", "2024-12-01", "No change",
     "pathlib-3.13.0", "Path.is_mount signature is unchanged between 3.12 and 3.13",
     "Between pathlib 3.12 and 3.13, did Path.is_mount change its signature?"),
    ("pl-isfile-post", QType.POST, "2025-01-01", "2024-12-01", "Yes (3.13)",
     "pathlib-3.13.0", "In 3.13 Path.is_file accepts a follow_symlinks parameter",
     "As of pathlib 3.13, does Path.is_file accept a follow_symlinks parameter?"),
    ("pl-isfile-pre", QType.PRE, "2025-01-01", "2023-01-01", "No (not in 3.12)",
     "pathlib-3.12.0", "In 3.12 Path.is_file has no follow_symlinks parameter",
     "In pathlib 3.12, did Path.is_file accept a follow_symlinks parameter?"),
]


def _seed_questions() -> list[Question]:
    return [
        Question(
            qid=qid, domain=DOMAIN, qtype=qt, question=q,
            ask_date=D(ask), val_date=D(val), gold_answer=gold,
            gold_span=span, gold_doc=doc, target_condition=Condition.of(lang="en"),
            family="pathlib", split="dev",
            meta={"status": "auto-seed; verify against whatsnew.rst"},
        )
        for qid, qt, ask, val, gold, doc, span, q in _SEED
    ]


def load(datasets_root: str | Path) -> tuple[list[Notice], list[Question]]:
    root = Path(datasets_root) / DOMAIN
    ext_root = Path(datasets_root) / "extension_20261121"
    return _load_notices(root, ext_root), _seed_questions()
