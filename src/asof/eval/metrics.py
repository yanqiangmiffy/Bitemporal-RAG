"""Scoring. The main metric is Joint accuracy: an item is correct iff the answer
AND the cited source span are both correct (paper §Experiments/Setup)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..schema import Question

_KEEP = re.compile(r"[A-Za-z0-9.\-/%一-鿿]")
_ABSTAIN = ("无法确定", "不适用", "无法回答", "notdetermin", "notapplicable", "cannotdetermine")


def norm(s: str) -> str:
    return "".join(_KEEP.findall((s or "").lower()))


def _num_variants(s: str) -> set[str]:
    v = {s}
    x = s.replace("%", "")
    x = re.sub(r"(\d+\.\d*?)0+\b", r"\1", x)      # 2.60 -> 2.6
    x = re.sub(r"(\d+)\.(?=\D|$)", r"\1", x)         # 2. -> 2
    v.add(x)
    return {y for y in v if y}


def is_abstain(s: str) -> bool:
    n = norm(s)
    return any(a in n for a in _ABSTAIN)


def answer_match(pred: str, gold: str) -> bool:
    if is_abstain(gold):
        return is_abstain(pred)
    P = _num_variants(norm(pred))
    G = _num_variants(norm(gold))
    return any(g and (g in p or p in g) for g in G for p in P)


def span_match(pred_span: str, gold_span: str, gold_answer: str = "") -> bool:
    if not gold_span or is_abstain(gold_answer):
        return True                      # nothing to ground (e.g. abstain items)
    p, g = norm(pred_span), norm(gold_span)
    if not p:
        return False
    if len(g) >= 8 and g[:8] in p:
        return True
    sp, sg = set(p), set(g)
    jac = len(sp & sg) / max(1, len(sp | sg))
    return jac >= 0.4


@dataclass
class Scored:
    qtype: str
    answer_ok: bool
    span_ok: bool
    joint_ok: bool


def score_item(q: Question, a: Any) -> Scored:
    """``a`` is any object with ``.text`` and ``.cited_span`` (e.g. methods.Answer)."""
    ao = answer_match(a.text, q.gold_answer)
    so = span_match(a.cited_span, q.gold_span, q.gold_answer)
    return Scored(q.qtype.value, ao, so, ao and so)


def evaluate(questions: list[Question], answers: list[Any], joint: bool = True) -> dict:
    types = ["Pre", "Post", "Unch", "Win", "Inap", "Unseen"]
    correct = {t: 0 for t in types}
    total = {t: 0 for t in types}
    n_ok = 0
    per_item = []
    for q, a in zip(questions, answers):
        s = score_item(q, a)
        ok = s.joint_ok if joint else s.answer_ok
        total[s.qtype] = total.get(s.qtype, 0) + 1
        correct[s.qtype] = correct.get(s.qtype, 0) + int(ok)
        n_ok += int(ok)
        per_item.append({"qid": q.qid, "qtype": s.qtype, "answer_ok": s.answer_ok,
                         "span_ok": s.span_ok, "joint_ok": s.joint_ok,
                         "pred": a.text, "gold": q.gold_answer})
    by_type = {t: round(100 * correct[t] / total[t], 1) for t in types if total[t]}
    n = len(questions)
    return {
        "overall": round(100 * n_ok / n, 1) if n else 0.0,
        "by_type": by_type,
        "n": n,
        "n_by_type": {t: total[t] for t in types if total[t]},
        "per_item": per_item,
    }
