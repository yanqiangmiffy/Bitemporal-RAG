"""LLM extractor: a notice -> candidate field-level facts (+ dates, cited ids).

Produces facts with placeholder intervals; Flim assigns the real valid/visible
intervals from the notice's effective/publish dates when it writes them.
"""
from __future__ import annotations

from ..llm import LLMClient
from ..llm.prompts import extractor_messages
from ..schema import Condition, Fact, Notice
from ..temporal import POS_INF, Interval, parse_date
from .dates import normalize_effective


def condition_from_llm(d: dict | None) -> Condition:
    if not d:
        return Condition.top()
    return Condition(eq={str(k): str(v) for k, v in d.items() if v not in (None, "")})


class LLMExtractor:
    def __init__(self, client: LLMClient, model: str):
        self.client = client
        self.model = model

    def extract(self, notice: Notice) -> list[Fact]:
        try:
            data = self.client.chat_json(extractor_messages(notice.title, notice.text), self.model)
        except Exception as e:               # malformed/odd JSON: skip this notice, keep the run alive
            print(f"[extract] {notice.doc_id}: JSON parse failed ({type(e).__name__}); 0 facts")
            return []
        if isinstance(data, list):           # some models answer with a bare list
            data = {"facts": data}
        if not isinstance(data, dict):
            return []

        # backfill notice dates / citations from the extractor if missing
        if notice.effective is None:
            eff = normalize_effective(notice, data.get("effective_date"))
            if eff is not None:
                notice.effective = eff
        if (not notice.published or notice.published >= POS_INF) and data.get("publish_date"):
            notice.published = parse_date(data["publish_date"])
        cited = data.get("cited_ids") or []
        if cited:
            notice.cited_ids = sorted(set(notice.cited_ids) | set(map(str, cited)))

        facts = []
        for fr in data.get("facts", []):
            if not fr.get("object") or not fr.get("field"):
                continue
            facts.append(Fact(
                obj=str(fr["object"]), field_name=str(fr["field"]),
                value=str(fr.get("value", "")),
                valid=Interval(POS_INF, POS_INF), visible=Interval(POS_INF, POS_INF),
                condition=condition_from_llm(fr.get("condition")),
                source_span=str(fr.get("source_span", "")),
                clause_id=(str(fr["clause_id"]) if fr.get("clause_id") else None),
                doc_id=notice.doc_id, domain=notice.domain,
            ))
        return facts
