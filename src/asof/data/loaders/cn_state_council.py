"""State Council orders + MoJ law versions -> notice stream;
road_752 worked examples -> real 6-type diagnostic questions.

Question-type inference for the road_752 items:
  * asked before the notice was public            -> Unseen
  * amendment evidence and target >= effective     -> Post
  * prior-text evidence and target < effective      -> Win
  * answer identical across the effective boundary  -> Unch (field unchanged)
"""
from __future__ import annotations

from pathlib import Path

from ...jsonl import read_jsonl
from ...schema import Condition, Notice, QType, Question
from ...temporal import parse_date as D
from .common import html_to_text, read_text

DOMAIN = "cn_state_council"


def _notice_from_order(o: dict) -> Notice:
    pub = o.get("first_public_date") or o.get("signed_date")
    return Notice(
        doc_id=o["order_id"],
        published=D(pub),
        effective=D(o["explicit_effective_date"]) if o.get("explicit_effective_date") else None,
        title=o["order_id"],
        text="\n".join(o.get("paragraphs", [])),
        domain=DOMAIN,
        family=o.get("order_id"),
    )


def _law_version_notice(root: Path, src: str, hist_date: str, family: str) -> Notice | None:
    p = root / src
    if not p.exists():
        return None
    return Notice(
        doc_id=f"law:{Path(src).stem}",
        published=D(hist_date),
        title=family,
        text=html_to_text(read_text(p)),
        domain=DOMAIN,
        family=family,
    )


def _infer_qtype(item: dict) -> QType:
    ask, target = D(item["ask_date"]), D(item["target_date"])
    pub = D(item.get("known_public_by")) if item.get("known_public_by") else None
    eff = D(item["explicit_effective_date"]) if item.get("explicit_effective_date") else None
    roles = {e.get("role") for e in item.get("citable_evidence", [])}
    if pub is not None and ask < pub:
        return QType.UNSEEN
    if "amendment" in roles and eff is not None and target >= eff:
        return QType.POST
    if "prior_text" in roles and eff is not None and target < eff:
        return QType.WIN
    return QType.UNCH


def _question(item: dict) -> Question:
    span = " / ".join(e.get("evidence", "") for e in item.get("citable_evidence", []))
    docs = [e.get("path") for e in item.get("citable_evidence", []) if e.get("path")]
    return Question(
        qid=item["id"], domain=DOMAIN, qtype=_infer_qtype(item),
        question=item["question"], ask_date=D(item["ask_date"]),
        val_date=D(item["target_date"]), gold_answer=item["reference_answer"],
        gold_span=span, gold_doc=item.get("order_id"),
        target_condition=Condition.top(), family=item.get("order_id"),
        split="dev",   # worked examples: excluded from blind test
        meta={"scope": item.get("scope"), "evidence_docs": docs,
              "known_public_by": item.get("known_public_by"),
              "effective": item.get("explicit_effective_date")},
    )


def _relabel_unchanged(questions: list[Question]) -> None:
    """If the same question has equal answers across the effective boundary, it
    is testing an unchanged field -> relabel those items Unch."""
    by_q: dict[str, list[Question]] = {}
    for q in questions:
        by_q.setdefault(q.question, []).append(q)
    for group in by_q.values():
        answers = {q.gold_answer for q in group}
        has_pre = any(q.qtype in (QType.WIN,) for q in group)
        has_post = any(q.qtype == QType.POST for q in group)
        if len(answers) == 1 and has_pre and has_post:
            for q in group:
                if q.qtype != QType.UNSEEN:
                    q.qtype = QType.UNCH


def load(datasets_root: str | Path) -> tuple[list[Notice], list[Question]]:
    root = Path(datasets_root)
    ext = root / "extension_20261121"
    notices: dict[str, Notice] = {}

    orders = ext / "cn_state_council" / "orders.jsonl"
    if orders.exists():
        for o in read_jsonl(orders):
            n = _notice_from_order(o)
            notices[n.doc_id] = n

    linked = ext / "cn_state_council" / "linked_changes.jsonl"
    if linked.exists():
        for lk in read_jsonl(linked):
            for src, date in ((lk.get("prior_source"), lk.get("prior_history_date")),
                              (lk.get("next_source"), lk.get("next_history_date"))):
                if src and date:
                    nv = _law_version_notice(ext, src, date, lk.get("family", ""))
                    if nv:
                        notices[nv.doc_id] = nv

    questions: list[Question] = []
    road = ext / "cn_state_council" / "road_752_examples.jsonl"
    if road.exists():
        questions = [_question(it) for it in read_jsonl(road)]
        _relabel_unchanged(questions)

    return list(notices.values()), questions
