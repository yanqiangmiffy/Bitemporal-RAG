"""The paper's housing-fund counterexample, in code.

A rate changes from 2.85% to 2.60%, published 2025-05-08, effective 2025-05-09.
The corrected bitemporal store must answer a query asked *before* publication,
about a future business date, with the pre-revision value (the Unseen case).
A single-timeline store (ablation: -visible axis) gets Unseen wrong but still
gets Win right -- exactly the paper's claim.
"""
from asof.schema import Condition, Fact, Notice, RevisionType
from asof.store import BitemporalStore, LocateImpact, flim_update
from asof.temporal import Interval
from asof.temporal import parse_date as D

OBJ, FIELD = "首套个人住房公积金贷款", "利率"


def _build(use_visible: bool) -> BitemporalStore:
    store = BitemporalStore(use_visible_time=use_visible)
    store.append(Fact(
        obj=OBJ, field_name=FIELD, value="2.85%",
        valid=Interval.of("2024-01-01", "INF"), visible=Interval.of("2024-01-01", "INF"),
        condition=Condition.top(), doc_id="base",
    ))
    a2 = Notice(doc_id="a2", published=D("2025-05-08"), effective=D("2025-05-09"),
                text="利率调整为2.60%", domain="beijing_housing")
    new = Fact(obj=OBJ, field_name=FIELD, value="2.60%",
               valid=Interval.of("INF", "INF"), visible=Interval.of("INF", "INF"))
    flim_update(store, a2, [new],
                classify=lambda f, C: (RevisionType.REPLACE, None),
                locate=LocateImpact(use_exact_id=False))
    return store


def _ans(store, ask, val):
    facts = store.query(D(ask), D(val))
    return [f.value for f in facts]


def test_bitemporal_answers_all_four_cases():
    s = _build(use_visible=True)
    # Unseen: asked 05-01 (before publication) about a June disbursement -> old value
    assert _ans(s, "2025-05-01", "2025-06-01") == ["2.85%"]
    # Win: asked 05-08 (visible) about 05-08 (not yet effective) -> old value
    assert _ans(s, "2025-05-08", "2025-05-08") == ["2.85%"]
    # Post: asked/val in June after effective -> new value
    assert _ans(s, "2025-06-01", "2025-06-01") == ["2.60%"]
    # Pre: asked later, business date in the past -> old value
    assert _ans(s, "2025-06-01", "2025-01-01") == ["2.85%"]


def test_single_timeline_fails_unseen_but_keeps_win():
    s = _build(use_visible=False)
    # Unseen is answered with the not-yet-knowable new value -> WRONG (demonstrates the bug)
    assert _ans(s, "2025-05-01", "2025-06-01") == ["2.60%"]
    # Win is still correct: valid-time alone resolves "published but not effective"
    assert _ans(s, "2025-05-08", "2025-05-08") == ["2.85%"]


def test_closing_preserves_history_on_visible_axis():
    s = _build(use_visible=True)
    # old belief is closed at p on the visible axis but its valid interval is intact
    closed = [f for f in s.facts if not f.is_current_belief]
    assert closed and all(f.visible.end == D("2025-05-08") for f in closed)
    # nothing is physically deleted
    assert len(s.facts) >= 3  # original (closed) + bounded history + new value
