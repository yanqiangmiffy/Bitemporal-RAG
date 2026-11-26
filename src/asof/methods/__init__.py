"""Method registry / factory."""
from __future__ import annotations

from ..config import Config
from ..llm import LLMClient
from .asof import AsofMethod
from .base import Answer, Method
from .baselines import (
    DocBaseline,
    FullRebuild,
    HybridRAG,
    LatestWins,
    LongContextDump,
    LongContextVisible,
    TimelyRAG,
    WholeDocInvalidation,
)

_REGISTRY = {
    "asof": AsofMethod,
    "full_rebuild": FullRebuild,
    "hybrid_rag": HybridRAG,
    "latest_wins": LatestWins,
    "whole_doc": WholeDocInvalidation,
    "timelyrag": TimelyRAG,
    "longctx": LongContextDump,
    "longctx_visible": LongContextVisible,
}


def make_method(name: str, cfg: Config, llm: LLMClient, embedder=None, reranker=None) -> Method:
    if name not in _REGISTRY:
        raise ValueError(f"unknown method '{name}'; choices={list(_REGISTRY)}")
    return _REGISTRY[name](cfg, llm, embedder=embedder, reranker=reranker)


__all__ = [
    "Answer",
    "AsofMethod",
    "DocBaseline",
    "FullRebuild",
    "HybridRAG",
    "LatestWins",
    "LongContextDump",
    "LongContextVisible",
    "Method",
    "TimelyRAG",
    "WholeDocInvalidation",
    "make_method",
]
