"""Hybrid retrieval over a set of items (facts or notice passages).

BM25 (always available) fused with dense retrieval (if an Embedder is provided)
by reciprocal-rank fusion, then an optional cross-encoder rerank. The same class
serves the As-of method (small surviving-fact set, built per query) and the
document baselines (per-domain corpus, built once with cached vectors).
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..index.dense import DenseIndex
from ..index.sparse import BM25Index


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal-rank fusion of several ranked index lists."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])]


class Retriever:
    def __init__(
        self,
        ids: list[str],
        texts: list[str],
        cfg: Config,
        embedder=None,
        reranker=None,
        doc_vectors: np.ndarray | None = None,
    ):
        self.ids, self.texts, self.cfg = ids, texts, cfg
        self.embedder, self.reranker = embedder, reranker
        self.bm25 = BM25Index(texts) if texts else None
        if embedder is not None and doc_vectors is None and texts:
            doc_vectors = embedder.encode(texts, is_query=False)
        self.doc_vectors = doc_vectors
        self.dense = DenseIndex(doc_vectors) if doc_vectors is not None else None

    def search(self, query: str, topk: int | None = None) -> list[tuple[str, str]]:
        if not self.texts:
            return []
        r = self.cfg.get("retrieval", {})
        cand = int(r.get("top_k_candidates", 50))
        topk = topk or int(r.get("top_k_final", 8))

        rankings: list[list[int]] = []
        if r.get("bm25", True) and self.bm25 is not None:
            s = self.bm25.scores(query)
            rankings.append(list(np.argsort(-s)[:cand]))
        if r.get("dense", True) and self.dense is not None and self.embedder is not None:
            qv = self.embedder.encode([query], is_query=True)[0]
            rankings.append([i for i, _ in self.dense.search(qv, cand)])

        fused = rrf(rankings, int(r.get("rrf_k", 60))) if rankings else list(range(len(self.texts)))
        fused = fused[:cand]

        if r.get("rerank", True) and self.reranker is not None and fused:
            scores = self.reranker.score(query, [self.texts[i] for i in fused])
            fused = [i for i, _ in sorted(zip(fused, scores), key=lambda x: -x[1])]

        return [(self.ids[i], self.texts[i]) for i in fused[:topk]]
