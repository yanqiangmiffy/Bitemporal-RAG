"""Run a method over a domain's notice stream + question set; write a run JSON.

Replays notices in publication order (maintenance), then answers the chosen
split and scores. Ablation runs merge a variant YAML over the config and run the
As-of method under each variant.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from ..config import Config, _deep_merge
from ..jsonl import read_jsonl, write_json
from ..llm import LLMClient
from ..methods import make_method
from ..schema import Notice, Question
from .cost import summarize_cost
from .metrics import evaluate


def _load_domain(cfg: Config, domain: str) -> tuple[list[Notice], list[Question]]:
    base = cfg.path("canonical") / domain
    notices = [Notice.from_dict(r) for r in read_jsonl(base / "notices.jsonl")]
    questions = [Question.from_dict(r) for r in read_jsonl(base / "questions.jsonl")]
    notices.sort(key=lambda n: (n.published, n.doc_id))
    return notices, questions


def _make_models(cfg: Config):
    """Embedder/reranker if available; None on CPU-only boxes (BM25-only)."""
    emb = rr = None
    r = cfg.get("retrieval", {})
    try:
        if r.get("dense", True):
            from ..index.embeddings import Embedder
            emb = Embedder(cfg)
        if r.get("rerank", True):
            from ..index.reranker import Reranker
            rr = Reranker(cfg)
    except Exception as e:
        print(f"[runner] dense/rerank unavailable ({e}); using BM25-only retrieval")
        emb = rr = None
    return emb, rr


def run_experiment(cfg: Config, domain: str, method_name: str,
                   split: str = "test", run_name: str | None = None) -> dict:
    notices, questions = _load_domain(cfg, domain)
    llm = LLMClient(cfg)
    emb, rr = _make_models(cfg)
    method = make_method(method_name, cfg, llm, emb, rr)

    logs = []
    for n in notices:
        lg = method.update(n)
        if lg is not None:
            logs.append(lg)
    method.finalize_setup()

    qs = [q for q in questions if q.split == split] or questions
    preds = [method.answer(q) for q in qs]
    metrics = evaluate(qs, preds, joint=bool(cfg.get("eval.joint", True)))
    cost = summarize_cost(logs)

    run = run_name or method_name
    result = {
        "domain": domain, "method": method_name, "run": run,
        "n_notices": len(notices), "n_questions": len(qs),
        "llm_provider": cfg.get("llm.provider"),
        "method_flags": cfg.get("method", {}),
        "metrics": metrics, "cost": cost,
    }
    out = cfg.path("runs")
    write_json(out / f"{domain}__{run}.json", result)
    print(f"[{domain}/{run}] overall={metrics['overall']}  by_type={metrics['by_type']}  "
          f"mean_update={cost.get('mean_update_seconds')}s  llm_calls={cost.get('llm_calls')}")
    return result


def run_ablation(cfg: Config, domain: str, variants_dir: Path) -> list[dict]:
    results = []
    for vf in sorted(Path(variants_dir).glob("*.yaml")):
        with open(vf, "r", encoding="utf-8") as f:
            variant = yaml.safe_load(f) or {}
        merged = Config(_deep_merge(dict(cfg), variant))
        merged["_root"] = cfg["_root"]
        results.append(run_experiment(merged, domain, "asof", run_name=f"ablation__{vf.stem}"))
    return results
