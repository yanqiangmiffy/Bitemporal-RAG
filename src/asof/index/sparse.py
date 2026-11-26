"""BM25 sparse retrieval (pure-Python via rank_bm25) with a CJK-aware tokenizer.

Chinese is tokenized to single characters; ASCII runs are kept as words. This
avoids a segmentation dependency and works well for BM25 over mixed zh/en text.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[A-Za-z0-9_]+|[一-鿿]")


def tokenize(text: str) -> list[str]:
    return [m.group().lower() for m in _TOKEN.finditer(text or "")]


class BM25Index:
    def __init__(self, texts: list[str]):
        self.tokenized = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized)

    def scores(self, query: str):
        return self.bm25.get_scores(tokenize(query))
