"""Dense index: FAISS inner-product if available, else numpy brute force.

Vectors are assumed L2-normalized, so inner product == cosine similarity.
Corpora here are small (facts / per-domain notices), so numpy is fine on a Mac.
"""
from __future__ import annotations

import numpy as np

try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover
    _HAS_FAISS = False


class DenseIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self._faiss = None
        if _HAS_FAISS and len(self.vectors):
            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)

    def search(self, qvec: np.ndarray, topk: int) -> list[tuple[int, float]]:
        if len(self.vectors) == 0:
            return []
        q = np.ascontiguousarray(qvec.astype(np.float32).reshape(1, -1))
        topk = min(topk, len(self.vectors))
        if self._faiss is not None:
            sims, idx = self._faiss.search(q, topk)
            return list(zip(idx[0].tolist(), sims[0].tolist()))
        sims = (self.vectors @ q[0])
        order = np.argsort(-sims)[:topk]
        return [(int(i), float(sims[i])) for i in order]
