"""LocateImpact: find the prior versions a new fact may affect.

Two tiers (paper §Update path, stage 2):
  1. exact: match on cited document/clause id and object (precise, cheap);
  2. vector recall fallback over the (object, field) key (pluggable embedder).

Only *currently-believed* versions (t_tau_end == ∞) are candidates. Returning
exactly the compatible (object, field, condition) current versions makes the
step "sound and complete" in the sense Theorem 1 assumes.
"""
from __future__ import annotations

from collections.abc import Callable

from ..schema import Fact
from .bitemporal import BitemporalStore

# A vector-recall function: (query_fact, candidate_facts, topk) -> ranked subset.
VectorRecall = Callable[[Fact, list[Fact], int], list[Fact]]


class LocateImpact:
    def __init__(
        self,
        use_exact_id: bool = True,
        vector_recall: VectorRecall | None = None,
        vector_topk: int = 20,
    ):
        self.use_exact_id = use_exact_id
        self.vector_recall = vector_recall
        self.vector_topk = vector_topk

    def __call__(self, f: Fact, store: BitemporalStore) -> list[Fact]:
        # Candidates are the current beliefs of THIS object only (indexed): a
        # revision can only touch versions of the same object, so this keeps the
        # step proportional to one object's facts, not the whole store (Prop. 1).
        current = store.current_for_obj(f.obj)

        # Tier 1: exact id/clause/object match.
        if self.use_exact_id:
            exact = [
                g for g in current
                if g.obj == f.obj
                and (
                    (f.clause_id and g.clause_id == f.clause_id)
                    or (g.doc_id is not None and g.doc_id in (f.doc_id, *(f.rho or "",)))
                )
            ]
            if exact:
                return _dedup(exact)

        # Tier 2: (object, field) match, optionally re-ranked by vector recall.
        okey_matches = [g for g in current if g.okey == f.okey]
        if self.vector_recall is not None and okey_matches:
            return _dedup(self.vector_recall(f, okey_matches, self.vector_topk))
        return _dedup(okey_matches)


def _dedup(facts: list[Fact]) -> list[Fact]:
    seen, out = set(), []
    for g in facts:
        if g.fact_id not in seen:
            seen.add(g.fact_id)
            out.append(g)
    return out
