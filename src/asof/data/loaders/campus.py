"""Campus notices (CANDOR/CondQA-Campus). Already a built benchmark; we reuse
the 1,009 notices as the publication stream and map the 247 items to Questions.

Native kinds (T1..T6) do not map 1:1 to the six bitemporal diagnostics, so we
preserve the original kind in ``meta['campus_kind']`` and use a documented,
approximate mapping: T6 (campus mismatch / refuse) -> Inap, else Unch (a field
stated in a notice). Campus mainly contributes version chains + conditional QA.
"""
from __future__ import annotations

from pathlib import Path

from ...jsonl import read_jsonl
from ...schema import Condition, Notice, QType, Question
from ...temporal import parse_date as D

DOMAIN = "campus"


def _qtype(item: dict) -> QType:
    if item.get("kind") == "T6" or item.get("gold_action") not in ("answer", None):
        return QType.INAP
    return QType.UNCH


def load(datasets_root: str | Path) -> tuple[list[Notice], list[Question]]:
    root = Path(datasets_root) / "campus"
    notices: list[Notice] = []
    npath = root / "notices.jsonl"
    if npath.exists():
        for r in read_jsonl(npath):
            notices.append(Notice(
                doc_id=r["doc_id"], published=D(r.get("published")),
                title=r.get("title", ""), text=r.get("text", ""),
                domain=DOMAIN, family=r.get("family") or r.get("cycle"),
            ))

    questions: list[Question] = []
    bpath = root / "benchmark.jsonl"
    if bpath.exists():
        for r in read_jsonl(bpath):
            src = r.get("source", {}) or {}
            today = r.get("today") or (r.get("hidden", {}) or {}).get("today")
            campus = (r.get("hidden", {}) or {}).get("campus") or src.get("campus")
            action = r.get("gold_action", "answer")
            if action == "answer":
                gold = "/".join(r.get("gold_tokens", [])) or src.get("value", "")
            else:
                gold = "无法确定"
            cond = Condition.of(campus=campus) if campus else Condition.top()
            questions.append(Question(
                qid=r.get("id", f"campus-{len(questions):04d}"), domain=DOMAIN,
                qtype=_qtype(r), question=r.get("question", ""),
                ask_date=D(today), val_date=D(today),
                gold_answer=gold, gold_span=src.get("quote", ""),
                gold_doc=src.get("gid"), target_condition=cond,
                family=src.get("title"), split="test",
                meta={"campus_kind": r.get("kind"), "gold_action": action,
                      "neg_tokens": r.get("neg_tokens", [])},
            ))
    return notices, questions
