from asof.schema import Condition, Fact, Notice, RevisionType
from asof.store import BitemporalStore, LocateImpact, flim_update
from asof.temporal import Interval, POS_INF, parse_date


def test_partial_revision_does_not_create_history_for_a_disjoint_population():
    start, published = parse_date("2024-01-01"), parse_date("2025-01-01")
    store = BitemporalStore()
    for children, value in [("2", "800"), ("3", "900")]:
        store.append(Fact(obj="family", field_name="benefit", value=value,
            condition=Condition(eq={"children": children}),
            valid=Interval(start, POS_INF), visible=Interval(start, POS_INF)))
    notice = Notice(doc_id="three-child-revision", published=published,
                    effective=published, text="Three-child families receive 1500.")
    new = Fact(obj="family", field_name="benefit", value="1500",
               condition=Condition(eq={"children": "3"}),
               valid=Interval(start, POS_INF), visible=Interval(start, POS_INF))
    log = flim_update(store, notice, [new],
        lambda fact, candidates: (RevisionType.PARTIAL, fact.condition), LocateImpact())
    history = store.query(published, parse_date("2024-06-01"),
                          Condition(eq={"children": "3"}))
    assert {fact.value for fact in history} == {"900"}
    assert log.affected == 1
    unaffected = store.query(published, published, Condition(eq={"children": "2"}))
    assert {fact.value for fact in unaffected} == {"800"}
