#!/usr/bin/env python
"""Maintenance-cost microbenchmark: Flim (incremental) vs full rebuild (Prop 1).

Pure-Python, no LLM: we drive the *actual* store/flim/locate code with a
synthetic revision stream and a deterministic classifier, so this measures index
maintenance cost in isolation (not extraction). Claim under test: a Flim update
costs time proportional to the number of *affected* facts (≈ constant per
notice), independent of store size; a full rebuild re-processes the whole prefix,
so its per-update cost grows linearly with the corpus and its cumulative cost is
quadratic.

Also reports the query-time context budget: a long-context baseline must feed the
whole (visible) corpus every query, which grows with |D|, whereas as-of retrieval
feeds a fixed top-k facts.

Writes data/bench/maintenance.json. Run: .venv/bin/python scripts/bench_maintenance.py
"""
from __future__ import annotations

import json
import random
import time
from datetime import date, timedelta
from pathlib import Path

import _bootstrap  # noqa: F401  (adds src/ to path)

from asof.schema import Condition, Fact, Notice, RevisionType
from asof.store import BitemporalStore, FlimOptions, LocateImpact, flim_update
from asof.temporal import POS_INF, Interval

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/bench/maintenance.json"

FIELDS = ["rate", "ratio", "office", "threshold", "limit"]
RNG = random.Random(20261125)


def deterministic_classifier(f: Fact, cands: list[Fact]):
    """Rule classifier (no LLM): revise a compatible same-(obj,field) belief,
    else insert. This is the common insert/replace case; it is enough to drive
    realistic affected-set sizes for the maintenance-cost measurement."""
    for g in cands:
        if g.obj == f.obj and g.field_name == f.field_name and g.condition.compatible(f.condition):
            return RevisionType.REPLACE, None
    return RevisionType.INSERT, None


def make_stream(n_notices: int, objects: int, facts_per_notice: int = 3) -> list[tuple[Notice, list[Fact]]]:
    """A stream where each notice revises a few existing (object, field) facts
    (so Flim has a bounded, roughly constant affected set) and occasionally
    introduces a new object (so the store grows)."""
    base = date(2020, 1, 1)
    obj_pool = [f"obj{idx:04d}" for idx in range(objects)]
    stream = []
    for i in range(n_notices):
        pub = base + timedelta(days=7 * i)
        doc_id = f"n{i:05d}"
        facts = []
        for _ in range(facts_per_notice):
            o = RNG.choice(obj_pool)
            fld = RNG.choice(FIELDS)
            facts.append(Fact(
                obj=o, field_name=fld, value=f"v{RNG.randint(0, 999)}",
                valid=Interval(POS_INF, POS_INF), visible=Interval(POS_INF, POS_INF),
                condition=Condition.top(), source_span="syn", doc_id=doc_id,
            ))
        stream.append((Notice(doc_id=doc_id, published=pub, text="syn", domain="bench"), facts))
    return stream


def run(n_notices=320, objects=120, step=20):
    locate = LocateImpact(use_exact_id=True, vector_recall=None)
    opts = FlimOptions(use_condition_split=True)
    stream = make_stream(n_notices, objects)

    # ---- Flim: process the whole stream once, incrementally -----------------
    flim_store = BitemporalStore(use_visible_time=True)
    flim_ms, flim_cum, flim_affected, sizes = [], [], [], []
    cum = 0.0
    for notice, facts in stream:
        # fresh Fact objects (flim mutates/copies); re-extract would do this too
        fs = [f.copy_with() for f in facts]
        t0 = time.perf_counter()
        log = flim_update(flim_store, notice, fs, deterministic_classifier, locate, opts)
        dt = (time.perf_counter() - t0) * 1000
        cum += dt
        flim_ms.append(dt); flim_cum.append(cum)
        flim_affected.append(log.affected); sizes.append(len(flim_store))

    # ---- Full rebuild on EVERY notice (what a redeploy-on-update system pays) #
    # At notice i the whole size-i corpus is rebuilt from scratch; the cumulative
    # maintenance cost is therefore sum_i rebuild(i) -- quadratic in |D|.
    checkpoints = list(range(step, n_notices + 1, step))
    rebuild_one_at_ckpt: dict[int, float] = {}
    rebuild_cum_series, rebuild_cum_at_ckpt = [], []
    run_cum = 0.0
    for i in range(1, n_notices + 1):
        store = BitemporalStore(use_visible_time=True)
        t0 = time.perf_counter()
        for notice, facts in stream[:i]:
            fs = [f.copy_with() for f in facts]
            flim_update(store, notice, fs, deterministic_classifier, locate, opts)
        dt = (time.perf_counter() - t0) * 1000       # cost to rebuild a size-i corpus
        run_cum += dt
        rebuild_cum_series.append(run_cum)
        if i in checkpoints:
            rebuild_one_at_ckpt[i] = dt
            rebuild_cum_at_ckpt.append(run_cum)
    rebuild_ms = [rebuild_one_at_ckpt[i] for i in checkpoints]
    rebuild_cum = rebuild_cum_at_ckpt

    # ---- Query-time context budget (chars) ---------------------------------
    # long-context must feed the whole visible corpus; as-of feeds fixed top-k.
    chars_per_notice = 600          # representative of the real notices
    topk, chars_per_fact = 8, 180
    qctx = {str(i): {"longctx_chars": i * chars_per_notice,
                     "asof_chars": topk * chars_per_fact} for i in checkpoints}

    result = {
        "params": {"n_notices": n_notices, "objects": objects, "step": step,
                   "facts_per_notice": 3},
        "sizes": sizes,
        "flim_per_update_ms": flim_ms,
        "flim_cumulative_ms": flim_cum,
        "flim_affected": flim_affected,
        "checkpoints": checkpoints,
        "rebuild_at_size_ms": rebuild_ms,        # cost of ONE rebuild of a size-i corpus
        "rebuild_cumulative_ms": rebuild_cum,    # cost of rebuilding every `step` notices
        "query_context_chars": qctx,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ---- console summary ----------------------------------------------------
    import statistics as sstat
    print(f"stream: {n_notices} notices, {objects} objects, store size={sizes[-1]} versions")
    print(f"Flim per-update: mean={sstat.mean(flim_ms):.3f} ms  "
          f"first10={sstat.mean(flim_ms[:10]):.3f}  last10={sstat.mean(flim_ms[-10:]):.3f} ms  "
          f"(flat => independent of store size)")
    print(f"Flim affected/update: mean={sstat.mean(flim_affected):.2f}")
    print(f"{'notices':>8} {'store':>7} {'flim_cum_ms':>12} {'rebuild_one_ms':>15} {'rebuild_cum_ms':>15}")
    for j, i in enumerate(checkpoints):
        print(f"{i:>8} {sizes[i-1]:>7} {flim_cum[i-1]:>12.1f} {rebuild_ms[j]:>15.1f} {rebuild_cum[j]:>15.1f}")
    print(f"\nquery context @ {checkpoints[-1]} notices: "
          f"longctx={checkpoints[-1]*600} chars vs as-of={8*180} chars "
          f"(~{checkpoints[-1]*600/(8*180):.0f}x)")


if __name__ == "__main__":
    run()
