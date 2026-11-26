"""Independent truth tables for condition partitioning and successive revisions."""
import itertools

from asof.schema import Condition, Fact, Notice, RevisionType
from asof.store import BitemporalStore, LocateImpact, flim_update
from asof.temporal import Interval, parse_date as D


def test_difference_partitions_two_dimensional_cover_without_losing_mixed_cases():
    base = Condition.top()
    cover = Condition.of(region="CN", loan="first")
    retained = base.difference(cover)
    for region, loan in itertools.product(["CN", "US", "UK"], ["first", "second", "third"]):
        point = Condition.of(region=region, loan=loan)
        hits = sum(c.compatible(point) for c in retained)
        assert hits == int(not (region == "CN" and loan == "first"))
    assert cover.difference(cover) == []
    assert base.difference(base) == []


def test_difference_respects_excluded_values_and_existing_constraints():
    base = Condition(ne={"region": frozenset({"UK"})})
    cover = Condition(eq={"loan": "first"}, ne={"region": frozenset({"US"})})
    for region, loan in itertools.product(["CN", "US", "UK"], ["first", "second"]):
        point = Condition.of(region=region, loan=loan)
        expected = region != "UK" and not (loan == "first" and region != "US")
        assert sum(c.compatible(point) for c in base.difference(cover)) == int(expected)


def test_third_revision_does_not_extend_previously_bounded_history():
    store = BitemporalStore()
    store.append(Fact(obj="loan", field_name="rate", value="1",
                      valid=Interval.of("2020-01-01", "INF"),
                      visible=Interval.of("2020-01-01", "INF")))
    classified_candidates = []
    def classify(fact, candidates):
        classified_candidates.append([g.value for g in candidates])
        return RevisionType.REPLACE, None
    for year, value in [(2021, "2"), (2022, "3"), (2023, "4")]:
        notice = Notice(str(year), D(f"{year}-01-01"), "", effective=D(f"{year}-02-01"))
        fact = Fact(obj="loan", field_name="rate", value=value,
                    valid=Interval.of("INF", "INF"), visible=Interval.of("INF", "INF"))
        flim_update(store, notice, [fact], classify, LocateImpact())
    assert classified_candidates == [["1"], ["2"], ["3"]]
    # Expected values come from the stated update schedule, not replaying Flim.
    for asking_year in [2020, 2021, 2022, 2023, 2024]:
        for business_year in [2020, 2021, 2022, 2023, 2024]:
            last_known_effective = min(asking_year, business_year, 2023)
            expected = str(max(0, last_known_effective - 2020) + 1)
            actual = store.query(D(f"{asking_year}-06-01"), D(f"{business_year}-06-01"))
            assert [f.value for f in actual] == [expected]


def test_partial_on_already_specific_population_does_not_keep_old_value_forever():
    cond = Condition.of(region="CN", loan="first")
    store = BitemporalStore()
    store.append(Fact(obj="loan", field_name="rate", value="old", condition=cond,
                      valid=Interval.of("2020-01-01", "INF"),
                      visible=Interval.of("2020-01-01", "INF")))
    notice = Notice("new", D("2021-01-01"), "", effective=D("2021-02-01"))
    fact = Fact(obj="loan", field_name="rate", value="new", condition=cond,
                valid=Interval.of("INF", "INF"), visible=Interval.of("INF", "INF"))
    flim_update(store, notice, [fact], lambda f, c: (RevisionType.PARTIAL, cond), LocateImpact())
    assert [f.value for f in store.query(D("2022-01-01"), D("2022-01-01"), cond)] == ["new"]
