#!/usr/bin/env python
"""Controlled demonstration that the Flim condition-split is *necessary* for
partial revisions -- isolated from LLM extraction/classification noise.

A broad rule ("every family: 800/month") is partially revised ("three-child
families: 1500"). We hold the upstream fixed with a deterministic classifier that
(correctly) labels the narrowing PARTIAL, and toggle only the store's
condition-split. Query an *unchanged* sub-population (one-child) after the
revision:

  * split ON  -> the retained branch (child-order != three-child, still 800) is
                 kept, so one-child correctly resolves to 800;
  * split OFF -> the partial is handled as a full replace, clobbering the broad
                 rule, so one-child has no applicable fact (wrong).

This is the end-to-end `no_condition_split` ablation made rigorous: whether the
*classifier* fires PARTIAL is a separate extraction-quality question, but given a
correct classification the store mechanism is what retains unchanged conditions.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
from datetime import date

from asof.schema import Condition, Fact, Notice, RevisionType
from asof.store import BitemporalStore, FlimOptions, LocateImpact, flim_update
from asof.temporal import POS_INF, Interval


def D(s): return date(*map(int, s.split("-")))


def classifier(f, cands):
    """Deterministic: a same-(obj,field) current belief exists -> the new fact
    narrows it (PARTIAL, covering the new fact's own condition); else INSERT."""
    if any(g.okey == f.okey for g in cands):
        return RevisionType.PARTIAL, f.condition
    return RevisionType.INSERT, None


def fresh(value, cond):
    return Fact(obj="育儿补贴", field_name="每月标准", value=value,
                valid=Interval(POS_INF, POS_INF), visible=Interval(POS_INF, POS_INF),
                condition=cond, source_span=value)


def run_variant(split: bool):
    store = BitemporalStore(use_visible_time=True)
    locate = LocateImpact(use_exact_id=False)      # (obj,field) recall
    opts = FlimOptions(use_condition_split=split)
    # n1: broad rule, every family 800 (condition = top)
    flim_update(store, Notice("n1", D("2024-01-01"), "all=800", effective=D("2024-01-01")),
                [fresh("800元", Condition.top())], classifier, locate, opts)
    # n2: partial revision, three-child -> 1500
    flim_update(store, Notice("n2", D("2024-07-01"), "3child=1500", effective=D("2024-07-01")),
                [fresh("1500元", Condition.of(孩次="三孩"))], classifier, locate, opts)
    # query the UNCHANGED one-child population after the revision
    res = store.query(D("2025-01-01"), D("2025-01-01"), Condition.of(孩次="一孩"))
    vals = sorted({f.value for f in res})
    return vals


def main():
    on = run_variant(split=True)
    off = run_variant(split=False)
    print("Partial revision: broad 800 -> three-child 1500; query ONE-CHILD afterwards")
    print(f"  condition-split ON : one-child applicable value(s) = {on}   "
          f"{'OK (retained 800)' if '800元' in on else 'WRONG'}")
    print(f"  condition-split OFF: one-child applicable value(s) = {off}   "
          f"{'WRONG (clobbered)' if '800元' not in off else 'ok'}")
    assert "800元" in on and "800元" not in off, "demonstration did not hold"
    print("=> condition-split is necessary: without it a partial revision clobbers "
          "unchanged sub-populations.")


if __name__ == "__main__":
    main()
