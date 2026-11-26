"""Cross-encoder reranker.

Two backends: a local sentence-transformers CrossEncoder (GPU extra), or an
OpenAI-compatible rerank endpoint (SiliconFlow BAAI/bge-reranker-v2-m3) when
``embeddings.api_base`` is set. Scores are cached by (model, query, docs).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request

from ..config import Config
from ..llm.cache import Cache


class Reranker:
    def __init__(self, cfg: Config):
        self.model_name = cfg.get("retrieval.reranker_model", "BAAI/bge-reranker-large")
        self.api_base = cfg.get("embeddings.api_base")
        self.device = cfg.get("embeddings.device", "cpu")
        self._api = bool(self.api_base)
        self._key = os.environ.get(cfg.get("embeddings.api_key_env", "SILICONFLOW_API_KEY"), "")
        cache_key = "reranker_cache" if cfg.get("paths.reranker_cache") else "cache"
        self._cache = Cache(cfg.path(cache_key)) if cfg.get("llm.cache", True) else None
        self._model = None
        self.strict = bool(cfg.get("retrieval.strict", False))
        self.revision = cfg.get("retrieval.reranker_revision")
        self.cached_only = bool(cfg.get("retrieval.cached_only", False))

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        if self._api:
            return self._score_api(query, docs)
        return [float(s) for s in self._load().predict([(query, d) for d in docs])]

    # -- API backend --------------------------------------------------------- #
    def _score_api(self, query: str, docs: list[str]) -> list[float]:
        tag = json.dumps([2, "api" if self._api else "local", self.api_base,
                          self.model_name, self.revision])
        key = "rr:" + hashlib.sha256(
            (tag + "\x00" + query + "\x00" + "\x01".join(docs)).encode()).hexdigest()
        hit = self._cache.get(key) if self._cache is not None else None
        if hit is not None:
            return hit["s"]
        if self.cached_only:
            raise RuntimeError("Required reranker scores are not cached; GPU inference is disabled")
        body = json.dumps({"model": self.model_name, "query": query,
                           "documents": docs, "top_n": len(docs)}).encode()
        req = urllib.request.Request(
            self.api_base.rstrip("/") + "/rerank", data=body,
            headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"})
        for attempt in range(5):
            try:
                r = json.load(urllib.request.urlopen(req, timeout=60))
                if self.strict and (len(r["results"]) != len(docs) or
                                    {x["index"] for x in r["results"]} != set(range(len(docs)))):
                    raise ValueError("Incomplete rerank response")
                scores = [0.0] * len(docs)
                for res in r["results"]:
                    scores[res["index"]] = float(res["relevance_score"])
                if self._cache is not None:
                    self._cache.put(key, "rr", {"s": scores})
                return scores
            except Exception:
                time.sleep(min(2 ** attempt, 20))
        if self.strict:
            raise RuntimeError("Reranker API failed after retries")
        print("[rerank] API failed after retries; leaving fusion order unchanged")
        return [0.0] * len(docs)   # uniform -> rerank is a no-op
