"""Build and load a per-domain corpus index (dense vectors + ids + texts).

Used by the document-level baselines. The As-of method builds a tiny retriever
over surviving facts at query time instead. Dense vectors need the GPU extra;
without it the saved index is BM25-only (texts + ids).
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..jsonl import read_jsonl, write_json, write_jsonl


def _passages(cfg: Config, domain: str) -> tuple[list[str], list[str]]:
    path = cfg.path("canonical") / domain / "notices.jsonl"
    ids, texts = [], []
    for r in read_jsonl(path):
        ids.append(r["doc_id"])
        texts.append((r.get("title", "") + "\n" + r.get("text", ""))[:4000])
    return ids, texts


def build_domain_index(cfg: Config, domain: str) -> dict:
    ids, texts = _passages(cfg, domain)
    out = cfg.path("index") / domain
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "passages.jsonl", ({"doc_id": i, "text": t} for i, t in zip(ids, texts)))

    doc_vectors = None
    try:
        from .embeddings import Embedder

        emb = Embedder(cfg)
        doc_vectors = emb.encode(texts, is_query=False)
        np.save(out / "dense.npy", doc_vectors)
        dim = int(doc_vectors.shape[1]) if len(doc_vectors) else 0
    except Exception as e:  # torch/sentence-transformers missing -> BM25-only index
        dim = 0
        print(f"[index] dense skipped for {domain}: {e}")

    meta = {"domain": domain, "passages": len(ids), "dense_dim": dim,
            "embedding_model": cfg.get("embeddings.model")}
    write_json(out / "meta.json", meta)
    print(f"[index] {domain}: {len(ids)} passages, dense_dim={dim} -> {out}")
    return meta


def load_corpus_index(cfg: Config, domain: str, embedder=None, reranker=None):
    from ..retrieval.hybrid import Retriever

    out = cfg.path("index") / domain
    rows = list(read_jsonl(out / "passages.jsonl"))
    ids = [r["doc_id"] for r in rows]
    texts = [r["text"] for r in rows]
    doc_vectors = None
    dense_path = out / "dense.npy"
    if dense_path.exists() and embedder is not None:
        doc_vectors = np.load(dense_path)
    return Retriever(ids, texts, cfg, embedder=embedder, reranker=reranker, doc_vectors=doc_vectors)


__all__ = ["build_domain_index", "load_corpus_index"]
