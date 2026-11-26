"""LLM revision classifier: (new fact, candidate old facts) -> revision type.

Per Algorithm 1 we return a single relation for the new fact against the
candidate set, plus the covered sub-condition when the relation is partial.
Priority partial > replace > delete; if no candidate conflicts, it is an insert.
"""
from __future__ import annotations

import json

from ..llm import LLMClient
from ..llm.prompts import classifier_messages
from ..schema import Condition, Fact, RevisionType
from .extractor import condition_from_llm


def _fact_str(f: Fact) -> str:
    return json.dumps(
        {"object": f.obj, "field": f.field_name, "value": f.value,
         "condition": f.condition.eq, "source_span": f.source_span[:200]},
        ensure_ascii=False,
    )


def _cands_str(cands: list[Fact]) -> str:
    return "\n".join(f"[{i}] {_fact_str(g)}" for i, g in enumerate(cands)) or "（无）"


class LLMClassifier:
    def __init__(self, client: LLMClient, model: str):
        self.client = client
        self.model = model

    def __call__(self, new_fact: Fact, cands: list[Fact]) -> tuple[RevisionType, Condition | None]:
        if not cands:
            return RevisionType.INSERT, None
        try:
            data = self.client.chat_json(
                classifier_messages(_fact_str(new_fact), _cands_str(cands)), self.model)
        except Exception as e:               # endpoint timeout / bad JSON: degrade to INSERT
            print(f"[classify] {new_fact.obj}/{new_fact.field_name}: failed ({type(e).__name__}); -> insert")
            return RevisionType.INSERT, None
        rels = data.get("relations", []) if isinstance(data, dict) else []

        covered = None
        chosen = RevisionType.INSERT
        rank = {RevisionType.PARTIAL: 3, RevisionType.REPLACE: 2,
                RevisionType.DELETE: 1, RevisionType.INSERT: 0}
        best = -1
        for rel in rels:
            name = str(rel.get("relation", "")).lower()
            if name in ("unrelated", "to_verify", ""):
                continue
            try:
                rt = RevisionType(name)
            except ValueError:
                continue
            if rank.get(rt, 0) > best:
                best = rank.get(rt, 0)
                chosen = rt
                if rt == RevisionType.PARTIAL:
                    covered = condition_from_llm(rel.get("covered_condition")) or new_fact.condition
        if chosen == RevisionType.PARTIAL and (covered is None or covered.is_top()):
            covered = new_fact.condition
        return chosen, covered
