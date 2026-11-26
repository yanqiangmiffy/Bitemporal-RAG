"""Baselines, each isolating one hypothesis (paper Table: Compared methods):

  * hybrid_rag  : time-agnostic BM25+dense+rerank over all documents (lower bound)
  * latest_wins : answer from the newest-published retrieved document (fails Pre/Win/Unseen)
  * whole_doc   : retire a whole document once any successor in its family appears (fails Unch)
  * timelyrag   : rerank by temporal proximity to the business date (no asking-date visibility)
  * full_rebuild: the bitemporal oracle, re-extracted from scratch each notice (cost reference)
"""
from __future__ import annotations

import time

from ..config import Config
from ..llm import LLMClient
from ..llm.prompts import generator_messages, longctx_messages
from ..retrieval import Retriever
from ..schema import Notice, Question
from ..store import BitemporalStore, flim_update
from ..store.flim import UpdateLog
from ..temporal import POS_INF, Date, fmt_date
from .asof import AsofMethod
from .base import Answer, Method


class DocBaseline(Method):
    """Document-level RAG with a pluggable visibility policy and reorder hook."""

    name = "doc"

    def __init__(self, cfg: Config, llm: LLMClient, embedder=None, reranker=None):
        self.cfg = cfg
        self.llm = llm
        self.gen_model = cfg.get("llm.generator_model")
        self.embedder, self.reranker = embedder, reranker
        self.notices: dict[str, Notice] = {}
        self.topk = int(cfg.get("retrieval.top_k_final", 8))
        self._retr: Retriever | None = None

    def update(self, notice: Notice) -> None:
        self.notices[notice.doc_id] = notice
        self._retr = None  # rebuilt lazily after the stream is consumed

    def _retriever(self) -> Retriever:
        if self._retr is None:
            ids = list(self.notices)
            texts = [(self.notices[i].title + "\n" + self.notices[i].text)[:4000] for i in ids]
            self._retr = Retriever(ids, texts, self.cfg,
                                   embedder=self.embedder, reranker=self.reranker)
        return self._retr

    # policy hooks -------------------------------------------------------- #
    def _allowed(self, q: Question) -> set[str]:
        return set(self.notices)

    def _reorder(self, q: Question, hits: list[tuple[str, str]]) -> list[tuple[str, str]]:
        return hits

    def answer(self, q: Question) -> Answer:
        retr = self._retriever()
        hits = retr.search(q.question, topk=max(self.topk * 4, 20))
        allowed = self._allowed(q)
        hits = [h for h in hits if h[0] in allowed]
        hits = self._reorder(q, hits)[: self.topk]
        if not hits:
            return Answer("无法确定")
        block = "\n".join(f"[{d}] {t[:800]}" for d, t in hits)
        try:
            data = self.llm.chat_json(
                generator_messages(q.question, fmt_date(q.ask_date), fmt_date(q.val_date), block),
                self.gen_model)
        except Exception as e:               # generation timeout / bad JSON: abstain, don't crash
            print(f"[answer] {q.qid}: generation failed ({type(e).__name__}); abstain")
            return Answer("无法确定")
        data = data if isinstance(data, dict) else {}
        return Answer(text=str(data.get("answer", "")).strip(),
                      cited_span=str(data.get("cited_span", "")), doc=hits[0][0])

    def _pub(self, doc_id: str) -> Date:
        n = self.notices.get(doc_id)
        return n.published if n else POS_INF


class HybridRAG(DocBaseline):
    name = "hybrid_rag"


class LatestWins(DocBaseline):
    name = "latest_wins"

    def _reorder(self, q, hits):
        return sorted(hits, key=lambda h: self._pub(h[0]), reverse=True)


class WholeDocInvalidation(DocBaseline):
    name = "whole_doc"

    def _allowed(self, q):
        retired = set()
        items = list(self.notices.values())
        for n in items:
            for m in items:
                if m.family and m.family == n.family and m.published > n.published:
                    retired.add(n.doc_id)
                    break
        return set(self.notices) - retired


class TimelyRAG(DocBaseline):
    name = "timelyrag"

    def _reorder(self, q, hits):
        def key(h):
            p = self._pub(h[0])
            gap = abs((p - q.val_date).days) if p < POS_INF else 10 ** 9
            return (gap, -(p.toordinal() if p < POS_INF else 0))
        return sorted(hits, key=key)


class LongContextDump(DocBaseline):
    """Brute-force 'give the LLM everything' baseline -- the 大力出奇迹 upper bound.

    Puts *every* notice (full text, tagged with its publication date) into one
    long context and asks the question, with no bitemporal index at all. The
    prompt explicitly states the as-of semantics, so any failure is a reasoning
    gap, not a prompt handicap. ``visible_only`` optionally restricts the dump to
    notices published on/before the asking date (document-level visible-time
    filtering) -- isolating 'cannot filter the future' from 'cannot reason
    valid-time/conditions over many in-context docs'.
    """

    name = "longctx"
    visible_only = False
    per_doc_chars = 3000

    def answer(self, q: Question) -> Answer:
        ids = list(self.notices)
        if self.visible_only:
            ids = [i for i in ids if self.notices[i].published <= q.ask_date]
        if not ids:
            return Answer("无法确定")
        ids.sort(key=lambda i: self.notices[i].published)   # oldest first
        blocks = [
            f"[{i}]（发布日 {fmt_date(self.notices[i].published)}）{self.notices[i].title}\n"
            f"{self.notices[i].text[:self.per_doc_chars]}"
            for i in ids
        ]
        docs = "\n\n".join(blocks)
        try:
            data = self.llm.chat_json(
                longctx_messages(q.question, fmt_date(q.ask_date), fmt_date(q.val_date), docs),
                self.gen_model)
        except Exception as e:               # context overflow / timeout: abstain, don't crash
            print(f"[answer] {q.qid}: longctx generation failed ({type(e).__name__}); abstain")
            return Answer("无法确定")
        data = data if isinstance(data, dict) else {}
        d = data.get("doc")
        return Answer(text=str(data.get("answer", "")).strip(),
                      cited_span=str(data.get("cited_span", "")),
                      doc=str(d) if isinstance(d, str) else ids[-1])


class LongContextVisible(LongContextDump):
    """Long-context dump restricted to notices visible by the asking date."""

    name = "longctx_visible"
    visible_only = True


class FullRebuild(AsofMethod):
    """Bitemporal oracle: re-extract and rebuild the whole store on every notice.

    Query-equivalent to the incremental As-of store (Theorem 1), so its accuracy
    matches; its per-update cost grows with the corpus (the expensive reference).
    """

    name = "full_rebuild"

    def __init__(self, cfg: Config, llm: LLMClient, embedder=None, reranker=None):
        super().__init__(cfg, llm, embedder, reranker)
        self._seen: list[Notice] = []

    def update(self, notice: Notice) -> UpdateLog:
        self._seen.append(notice)
        t0 = time.time()
        usage_before = dict(self.llm.stats)
        self.store = BitemporalStore(use_visible_time=self.store.use_visible_time)
        total = UpdateLog(doc_id=notice.doc_id)
        for n in self._seen:
            facts = self.extractor.extract(n)
            log = flim_update(self.store, n, facts, self.classifier, self.locate, self.flim_opts)
            total.add(log)
        total.seconds = time.time() - t0
        self._account_update_calls(total, usage_before)
        return total
