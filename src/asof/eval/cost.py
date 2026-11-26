"""Maintenance-cost accounting over the notice stream."""
from __future__ import annotations

from ..store.flim import UpdateLog


def summarize_cost(logs: list[UpdateLog]) -> dict:
    if not logs:
        return {"updates": 0}
    total_s = sum(x.seconds for x in logs)
    calls = sum(x.llm_calls for x in logs)
    writes = sum(x.writes for x in logs)
    affected = sum(x.affected for x in logs)
    cum, running = [], 0.0
    for x in logs:
        running += x.seconds
        cum.append(round(running, 4))
    n = len(logs)
    return {
        "updates": n,
        "total_seconds": round(total_s, 4),
        "mean_update_seconds": round(total_s / n, 4),
        "llm_calls": calls,
        "mean_llm_calls": round(calls / n, 3),
        "llm_api_calls": sum(x.llm_api_calls for x in logs),
        "llm_cache_hits": sum(x.llm_cache_hits for x in logs),
        "llm_call_scope": "successful maintenance requests, including cached responses; excludes failed attempts",
        "writes": writes,
        "affected_facts": affected,
        "cumulative_seconds": cum,     # the Fig. cost-curve series
    }
