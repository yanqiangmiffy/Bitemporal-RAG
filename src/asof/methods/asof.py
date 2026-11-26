"""As-of Retrieval (ours): Flim maintenance + bitemporal filtering + hybrid
retrieval over surviving facts + grounded generation. Ablation flags from
cfg['method'] toggle the visible-time axis, condition split and exact-id match.
"""
from __future__ import annotations

import time
import numpy as np

from ..config import Config
from ..extract import LLMClassifier, LLMExtractor
from ..llm import LLMClient
from ..llm.prompts import generator_messages
from ..retrieval import Retriever
from ..schema import Fact, Notice, Question
from ..store import BitemporalStore, FlimOptions, LocateImpact, flim_update
from ..store.flim import UpdateLog
from ..temporal import fmt_date
from .base import Answer, Method


def fact_text(f: Fact) -> str:
    cond = "" if f.condition.is_top() else f" | 条件:{f.condition}"
    return f"{f.obj} | {f.field_name} | {f.value}{cond} | {f.source_span}"


class AsofMethod(Method):
    name = "asof"

    def __init__(self, cfg: Config, llm: LLMClient, embedder=None, reranker=None):
        self.cfg = cfg
        m = cfg.get("method", {})
        self.store = BitemporalStore(use_visible_time=bool(m.get("use_visible_time", True)))
        self.locate = LocateImpact(
            use_exact_id=bool(m.get("use_exact_id_match", True)),
            vector_recall=self._vector_recall if embedder is not None else None,
            vector_topk=int(m.get("locate_vector_topk", 20)),
        )
        self.flim_opts = FlimOptions(use_condition_split=bool(m.get("use_condition_split", True)))
        self.extractor = LLMExtractor(llm, cfg.get("llm.extractor_model"))
        self.classifier = LLMClassifier(llm, cfg.get("llm.classifier_model"))
        self.llm = llm
        self.gen_model = cfg.get("llm.generator_model")
        self.embedder, self.reranker = embedder, reranker
        self.topk = int(cfg.get("retrieval.top_k_final", 8))

    def _vector_recall(self, fact, candidates, topk):
        vectors = self.embedder.encode([fact_text(g) for g in candidates])
        query = self.embedder.encode([fact_text(fact)], is_query=True)[0]
        order = np.argsort(-(vectors @ query), kind="stable")[:topk]
        return [candidates[int(i)] for i in order]

    def update(self, notice: Notice) -> UpdateLog:
        t0 = time.time()
        usage_before = dict(self.llm.stats)
        facts = self.extractor.extract(notice)
        log = flim_update(self.store, notice, facts, self.classifier, self.locate, self.flim_opts)
        log.seconds = time.time() - t0
        self._account_update_calls(log, usage_before)
        return log

    def _account_update_calls(self, log: UpdateLog, before: dict) -> None:
        """Measure successful calls during sequential maintenance, including cache hits.

        Looking up candidates after insertion counts newly inserted facts and cannot
        reconstruct which classifier calls actually ran. Use client counters instead.
        """
        log.llm_api_calls = self.llm.stats["api_calls"] - before["api_calls"]
        log.llm_cache_hits = self.llm.stats["cache_hits"] - before["cache_hits"]
        log.llm_calls = log.llm_api_calls + log.llm_cache_hits

    def answer(self, q: Question) -> Answer:
        if not q.question.strip():
            return Answer("无法确定")
        # (ii) as-of selection: prune to visible, valid, applicable facts (Eq. 2)
        surviving = self.store.query(q.ask_date, q.val_date, q.target_condition)
        if not surviving:
            return Answer("无法确定")
        # (iii) hybrid retrieval over the surviving facts
        ids = [f.fact_id for f in surviving]
        by_id = {f.fact_id: f for f in surviving}
        retr = Retriever(ids, [fact_text(by_id[i]) for i in ids], self.cfg,
                         embedder=self.embedder, reranker=self.reranker)
        query = q.question
        hits = retr.search(query, topk=self.topk) or [(i, fact_text(by_id[i])) for i in ids[: self.topk]]

        # Physical store IDs depend on rebuild/allocation history. Query-local
        # evidence IDs keep identical evidence prompts identical across stores.
        evidence_ids = {f"E{i + 1}": fid for i, (fid, _) in enumerate(hits)}
        block = "\n".join(f"[E{i + 1}] {text}" for i, (_, text) in enumerate(hits))
        try:
            data = self.llm.chat_json(
                generator_messages(q.question, fmt_date(q.ask_date), fmt_date(q.val_date), block),
                self.gen_model,
            )
        except Exception as e:               # generation timeout / bad JSON: abstain, don't crash
            print(f"[answer] {q.qid}: generation failed ({type(e).__name__}); abstain")
            return Answer("无法确定")
        data = data if isinstance(data, dict) else {}
        fid = data.get("fact_id")
        if isinstance(fid, list):            # some models return a list of ids
            fid = fid[0] if fid else None
        if not isinstance(fid, str):
            fid = None
        fid = evidence_ids.get(fid)
        f = by_id.get(fid)
        return Answer(
            text=str(data.get("answer", "")).strip(),
            cited_span=str(data.get("cited_span", "") or (f.source_span if f else "")),
            doc=(f.doc_id if f else (by_id[hits[0][0]].doc_id if hits else None)),
            fact_id=fid,
        )
