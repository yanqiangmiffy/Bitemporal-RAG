"""Schema serialization round-trips and condition-lattice behavior."""
from asof.schema import Condition, Fact, Notice, QType, Question
from asof.temporal import Interval
from asof.temporal import parse_date as D


def test_fact_roundtrip():
    f = Fact(obj="loan", field_name="rate", value="2.6%",
             valid=Interval.of("2025-05-09", "INF"), visible=Interval.of("2025-05-08", "INF"),
             condition=Condition.of(term=">5y"), source_span="rate is 2.6%", doc_id="a2")
    g = Fact.from_dict(f.to_dict())
    assert (g.obj, g.field_name, g.value) == ("loan", "rate", "2.6%")
    assert g.valid.start == D("2025-05-09") and g.visible.start == D("2025-05-08")
    assert g.condition.eq == {"term": ">5y"}


def test_notice_and_question_roundtrip():
    n = Notice(doc_id="a2", published=D("2025-05-08"), effective=D("2025-05-09"),
               text="...", domain="beijing_housing", cited_ids=["x"])
    assert Notice.from_dict(n.to_dict()).effective == D("2025-05-09")
    q = Question(qid="q1", domain="d", qtype=QType.UNSEEN, question="?",
                 ask_date=D("2025-05-01"), val_date=D("2025-06-01"), gold_answer="2.85%")
    assert Question.from_dict(q.to_dict()).qtype == QType.UNSEEN


def test_condition_lattice():
    top = Condition.top()
    a = Condition.of(disb="after")
    b = Condition.of(disb="before")
    assert top.compatible(a) and a.compatible(top)
    assert not a.compatible(b)                 # contradict on same dimension
    retained = top.minus(a)                    # "not after"
    assert retained.compatible(b) and not retained.compatible(a)
