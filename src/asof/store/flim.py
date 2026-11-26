"""Flim: field-level incremental maintenance (paper Algorithm 1).

On each notice, Flim locates the affected current beliefs, classifies the
revision, and writes append-only on the visible axis:
  * insert         -> append new version, valid [teff,∞), visible [p,∞)
  * replace/delete -> close each old belief at p, re-append its bounded history
                      [start,teff); replace also appends the new value
  * partial        -> close each old belief at p, re-append a retained branch
                      (untouched condition, still open) and a covered-history
                      branch (bounded at teff), then append the new value

The classifier is injected: a deterministic rule in tests, an LLM in real runs.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..schema import Condition, Fact, Notice, RevisionType
from ..temporal import POS_INF, Date, Interval
from .bitemporal import BitemporalStore
from .locate import LocateImpact

# classify(new_fact, candidates) -> (revision_type, covered_condition_for_partial)
Classifier = Callable[[Fact, list[Fact]], tuple[RevisionType, Condition | None]]


@dataclass
class FlimOptions:
    use_condition_split: bool = True   # ablation: off -> partial handled as replace


@dataclass
class UpdateLog:
    doc_id: str
    affected: int = 0        # k: number of prior versions touched
    writes: int = 0          # versions appended / closed
    classified: int = 0      # (new fact) classifications
    deferred: int = 0        # unrelated / to-verify
    seconds: float = 0.0
    llm_calls: int = 0
    llm_api_calls: int = 0
    llm_cache_hits: int = 0

    def add(self, other: UpdateLog) -> None:
        self.affected += other.affected
        self.writes += other.writes
        self.classified += other.classified
        self.deferred += other.deferred
        self.seconds += other.seconds
        self.llm_calls += other.llm_calls
        self.llm_api_calls += other.llm_api_calls
        self.llm_cache_hits += other.llm_cache_hits


def flim_update(
    store: BitemporalStore,
    notice: Notice,
    new_facts: list[Fact],
    classify: Classifier,
    locate: LocateImpact,
    opts: FlimOptions | None = None,
) -> UpdateLog:
    opts = opts or FlimOptions()
    p: Date = notice.published
    teff: Date = notice.effective or notice.published   # fallback: effective on publish
    log = UpdateLog(doc_id=notice.doc_id)

    for f in new_facts:
        f.doc_id = f.doc_id or notice.doc_id
        f.domain = f.domain or notice.domain
        # Already ended valid-time history cannot be changed by a later-effective
        # revision. Exclude it before classification as well as before writing.
        C = [g for g in locate(f, store) if g.valid.end > teff]
        r, covered = classify(f, C)
        log.classified += 1

        if r == RevisionType.INSERT:
            store.append(f.copy_with(valid=Interval(teff, POS_INF),
                                     visible=Interval(p, POS_INF), rho=notice.doc_id))
            log.writes += 1

        elif r in (RevisionType.REPLACE, RevisionType.DELETE):
            _apply_replace(store, f, C, r, p, teff, notice.doc_id, log)

        elif r == RevisionType.PARTIAL and opts.use_condition_split:
            _apply_partial(store, f, C, covered, p, teff, notice.doc_id, log)

        elif r == RevisionType.PARTIAL and not opts.use_condition_split:
            # Ablation: no split -> clobber the whole old condition (breaks Unch).
            _apply_replace(store, f, C, RevisionType.REPLACE, p, teff, notice.doc_id, log)

        else:  # UNRELATED / to-verify
            log.deferred += 1

    return log


def _apply_replace(store, f, C, r, p, teff, doc_id, log) -> None:
    affected = [g for g in C if g.condition.compatible(f.condition) and g.valid.end > teff]
    for g in affected:
        store.close_visible(g, p)
        if store.use_visible_time:
            if g.valid.start < teff:
                store.append(g.copy_with(valid=Interval(g.valid.start, min(g.valid.end, teff)),
                                         visible=Interval(p, POS_INF)))
                log.writes += 1
        else:
            g.valid = Interval(g.valid.start, min(g.valid.end, teff))
    log.affected += len(affected)
    if r == RevisionType.REPLACE:
        store.append(f.copy_with(valid=Interval(teff, POS_INF),
                                 visible=Interval(p, POS_INF), rho=doc_id))
        log.writes += 1


def _apply_partial(store, f, C, covered, p, teff, doc_id, log) -> None:
    cov = covered or f.condition
    affected = 0
    for g in C:
        overlap = g.condition.meet(cov)
        if overlap is None or g.valid.end <= teff:
            continue
        store.close_visible(g, p)
        affected += 1
        if store.use_visible_time:
            # A conjunction's complement may need several disjoint branches.
            for retained in g.condition.difference(cov):
                store.append(g.copy_with(condition=retained, valid=g.valid,
                                         visible=Interval(p, POS_INF)))
                log.writes += 1
            # covered-history branch: old value on the covered condition, bounded
            if g.valid.start < teff:
                store.append(g.copy_with(condition=overlap,
                                         valid=Interval(g.valid.start, min(g.valid.end, teff)),
                                         visible=Interval(p, POS_INF)))
                log.writes += 1
        else:
            g.valid = Interval(g.valid.start, min(g.valid.end, teff))
    log.affected += affected
    store.append(f.copy_with(condition=cov, valid=Interval(teff, POS_INF),
                             visible=Interval(p, POS_INF), rho=doc_id))
    log.writes += 1
