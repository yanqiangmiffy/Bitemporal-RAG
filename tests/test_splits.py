from collections import Counter

import pytest

from asof.eval.splits import make_domain_split, make_grouped_split


def test_forced_exposure_is_development_and_partition_is_reproducible():
    rows = [{"qid": str(i), "category": "a" if i < 70 else "b"} for i in range(100)]
    exposed = {"2", "72", "73"}
    first = make_domain_split(rows, exposed, dev_size=20, validation_size=10)
    assert first == make_domain_split(list(reversed(rows)), exposed, dev_size=20, validation_size=10)
    assert all(first[q] == "development" for q in exposed)
    assert Counter(first.values()) == {"development": 20, "validation": 10, "test": 70}
    assert Counter(r["category"] for r in rows if first[r["qid"]] == "development") == {"a": 14, "b": 6}


def test_split_refuses_to_hide_prior_exposure_when_development_is_too_small():
    rows = [{"qid": str(i)} for i in range(10)]
    with pytest.raises(ValueError):
        make_domain_split(rows, {"1", "2", "3"}, dev_size=2, validation_size=1)


def test_duplicate_question_group_never_crosses_holdout_boundary():
    rows = [{"qid": str(i), "category": "a" if i < 30 else "b",
             "group": "duplicate" if i in {0, 10, 20, 35} else str(i)} for i in range(60)]
    split = make_grouped_split(rows, {"0"}, dev_size=12, validation_size=8)
    assert {split[str(i)] for i in [0, 10, 20, 35]} == {"development"}
    assert Counter(split.values()) == {"development": 12, "validation": 8, "test": 40}
    assert split == make_grouped_split(list(reversed(rows)), {"0"}, dev_size=12, validation_size=8)
