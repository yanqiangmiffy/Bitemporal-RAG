"""Answer / span matching and the joint metric."""
from dataclasses import dataclass

from asof.eval.metrics import answer_match, evaluate, span_match
from asof.schema import QType, Question
from asof.temporal import parse_date as D


@dataclass
class A:
    text: str
    cited_span: str = ""


def test_answer_match_numeric_and_abstain():
    assert answer_match("利率为2.60%", "2.6%")          # trailing-zero / % tolerant
    assert answer_match("2.6%", "2.60%")
    assert not answer_match("2.85%", "2.6%")
    assert answer_match("无法确定该问题", "无法确定")      # abstain items
    assert not answer_match("是的，适用", "无法确定")


def test_span_match():
    assert span_match("……5年以上首套2.6%……", "5年以上首套2.6%")
    assert span_match("anything", "", gold_answer="无法确定")   # abstain -> no span needed
    assert not span_match("", "必须引用的原文片段")


def test_evaluate_joint_by_type():
    qs = [
        Question("q1", "d", QType.POST, "?", D("2025-06-01"), D("2025-06-01"),
                 gold_answer="2.6%", gold_span="2.6%"),
        Question("q2", "d", QType.UNSEEN, "?", D("2025-05-01"), D("2025-06-01"),
                 gold_answer="2.85%", gold_span="2.85%"),
    ]
    good = [A("2.6%", "rate 2.6%"), A("2.85%", "was 2.85%")]
    m = evaluate(qs, good, joint=True)
    assert m["overall"] == 100.0 and m["by_type"]["Unseen"] == 100.0
    bad = [A("2.6%", "rate 2.6%"), A("2.6%", "2.6%")]   # Unseen wrong
    m2 = evaluate(qs, bad, joint=True)
    assert m2["by_type"]["Unseen"] == 0.0 and m2["overall"] == 50.0
