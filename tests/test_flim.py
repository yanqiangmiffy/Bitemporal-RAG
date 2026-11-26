"""Flim: partial-coverage split and incremental == full-rebuild equivalence."""
from asof.schema import Condition, Fact, Notice, RevisionType
from asof.store import BitemporalStore, FlimOptions, LocateImpact, flim_update
from asof.temporal import Interval
from asof.temporal import parse_date as D

OBJ, FIELD = "二套个人住房公积金贷款", "最低首付比例"
COV = Condition.of(disb="after")        # covered sub-condition (disbursed on/after teff)


def _partial_store(use_condition_split: bool) -> BitemporalStore:
    store = BitemporalStore(use_visible_time=True)
    store.append(Fact(
        obj=OBJ, field_name=FIELD, value="30%",
        valid=Interval.of("2024-01-01", "INF"), visible=Interval.of("2024-01-01", "INF"),
        condition=Condition.top(), doc_id="base",
    ))
    a = Notice(doc_id="a_partial", published=D("2025-08-08"), effective=D("2025-08-08"),
               text="二套最低首付比例调整为25%（针对此后放款）", domain="beijing_housing")
    new = Fact(obj=OBJ, field_name=FIELD, value="25%",
               valid=Interval.of("INF", "INF"), visible=Interval.of("INF", "INF"), condition=COV)
    flim_update(store, a, [new],
                classify=lambda f, C: (RevisionType.PARTIAL, COV),
                locate=LocateImpact(use_exact_id=False),
                opts=FlimOptions(use_condition_split=use_condition_split))
    return store


def _ans(store, ask, val, cond):
    return sorted(f.value for f in store.query(D(ask), D(val), cond))


def test_partial_split_keeps_unchanged_condition():
    s = _partial_store(use_condition_split=True)
    before, after = Condition.of(disb="before"), Condition.of(disb="after")
    # covered condition in the future -> new value
    assert _ans(s, "2026-01-01", "2026-01-01", after) == ["25%"]
    # untouched (retained) condition in the future -> old value preserved (Unch)
    assert _ans(s, "2026-01-01", "2026-01-01", before) == ["30%"]


def test_no_condition_split_clobbers_unchanged():
    s = _partial_store(use_condition_split=False)
    before = Condition.of(disb="before")
    # ablation treats partial as replace -> the retained 'before' case is lost
    assert _ans(s, "2026-01-01", "2026-01-01", before) == []


def _stream():
    """A small multi-notice stream over two objects/fields."""
    notices, newfacts, cls = [], [], []
    base = BitemporalStore(use_visible_time=True)
    base.append(Fact(obj="A", field_name="x", value="v0",
                     valid=Interval.of("2020-01-01", "INF"),
                     visible=Interval.of("2020-01-01", "INF"), doc_id="base"))
    steps = [
        ("n1", "2021-01-01", "2021-02-01", "A", "x", "v1", RevisionType.REPLACE),
        ("n2", "2022-01-01", "2022-01-01", "B", "y", "w0", RevisionType.INSERT),
        ("n3", "2023-01-01", "2023-03-01", "A", "x", "v2", RevisionType.REPLACE),
    ]
    for doc, pub, eff, o, fld, val, r in steps:
        notices.append(Notice(doc_id=doc, published=D(pub), effective=D(eff), text=""))
        newfacts.append(Fact(obj=o, field_name=fld, value=val,
                             valid=Interval.of("INF", "INF"), visible=Interval.of("INF", "INF")))
        cls.append(r)
    return base, notices, newfacts, cls


def test_incremental_equals_full_rebuild():
    # Incremental: keep one store, apply notices one at a time.
    inc, notices, newfacts, cls = _stream()
    locate = LocateImpact(use_exact_id=False)
    for n, f, r in zip(notices, newfacts, cls):
        flim_update(inc, n, [f.copy_with()], lambda f, C, r=r: (r, None), locate)

    # Full rebuild: fresh store, replay the whole prefix from empty.
    full, notices2, newfacts2, cls2 = _stream()
    for n, f, r in zip(notices2, newfacts2, cls2):
        flim_update(full, n, [f.copy_with()], lambda f, C, r=r: (r, None), locate)

    # Query-equivalence on a grid of (t_ask, t_val) -> identical value multisets.
    grid = ["2020-06-01", "2021-06-01", "2022-06-01", "2023-06-01", "2024-06-01"]
    for ask in grid:
        for val in grid:
            qi = sorted(f.value for f in inc.query(D(ask), D(val)))
            qf = sorted(f.value for f in full.query(D(ask), D(val)))
            assert qi == qf, f"mismatch at ask={ask} val={val}: {qi} != {qf}"
