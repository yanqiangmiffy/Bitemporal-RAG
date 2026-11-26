#!/usr/bin/env python
"""Synthetic 'condition axis' domain: a broad rule narrowed by partial revisions.

Purpose: the housing data revises whole fields, so it under-tests the
*applicability-condition* axis (the task is not only about dates). Here a single
broad fact ("every family: 800/month") is narrowed by later notices that change
only a sub-population (three-child -> 1500, two-child -> 1200). Answering an
*unchanged* sub-population after such a revision requires the Flim condition-split
(partial): without it, a partial revision clobbers the whole prior rule. Short,
clean notice texts so extraction is reliable; gold is defined by the verified
ledger + the as-of semantics (independent of the method under test).

Writes data/canonical/cond_demo/{notices.jsonl,questions.jsonl}.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/canonical/cond_demo"


def D(s): return date(*map(int, s.split("-")))


NOTICES = [
    dict(doc_id="childcare_2024", published="2024-01-01", effective="2024-01-01",
         title="全市育儿补贴发放标准公告",
         text="为支持生育，现公布全市育儿补贴标准。自2024年1月1日起，本市每个有未成年子女的家庭，"
              "每月发放育儿补贴800元。"),
    dict(doc_id="childcare_20240701", published="2024-07-01", effective="2024-07-01",
         title="关于调整三孩家庭育儿补贴的通知",
         text="自2024年7月1日起，三孩家庭育儿补贴上调至每月1500元，其他家庭维持每月800元不变。"),
    dict(doc_id="childcare_20250101", published="2025-01-01", effective="2025-03-01",
         title="关于调整二孩家庭育儿补贴的通知",
         text="自2025年3月1日起，二孩家庭育儿补贴上调至每月1200元。"),
]

# Per-(child-order) value timeline: (value, visible_from=pub, valid_from=eff, doc)
LEDGER = {
    "一孩": [("800元", "2024-01-01", "2024-01-01", "childcare_2024")],
    "二孩": [("800元", "2024-01-01", "2024-01-01", "childcare_2024"),
             ("1200元", "2025-01-01", "2025-03-01", "childcare_20250101")],
    "三孩": [("800元", "2024-01-01", "2024-01-01", "childcare_2024"),
             ("1500元", "2024-07-01", "2024-07-01", "childcare_20240701")],
}
SPAN = {"800元": "每月发放育儿补贴800元", "1500元": "三孩家庭育儿补贴上调至每月1500元",
        "1200元": "二孩家庭育儿补贴上调至每月1200元"}


def asof(tl, ask, val):
    vis = [v for v in tl if D(v[1]) <= ask and D(v[2]) <= val]
    return max(vis, key=lambda v: (D(v[2]), D(v[1]))) if vis else None


# (qid, qtype, child, ask, val)  -- gold computed from the ledger
SPECS = [
    ("cd-00", "Unch",   "一孩", "2025-06-01", "2025-06-01"),  # unchanged after both partials
    ("cd-01", "Unch",   "一孩", "2024-09-01", "2024-09-01"),
    ("cd-02", "Unch",   "三孩", "2025-06-01", "2025-06-01"),  # n3 changed 二孩, 三孩 untouched
    ("cd-03", "Pre",    "三孩", "2024-09-01", "2024-03-01"),
    ("cd-04", "Post",   "三孩", "2024-09-01", "2024-09-01"),
    ("cd-05", "Unseen", "三孩", "2024-03-01", "2024-09-01"),  # before n2 published
    ("cd-06", "Pre",    "二孩", "2025-06-01", "2024-12-01"),
    ("cd-07", "Post",   "二孩", "2025-06-01", "2025-06-01"),
    ("cd-08", "Win",    "二孩", "2025-02-01", "2025-02-01"),  # n3 known, not yet effective
    ("cd-09", "Unseen", "二孩", "2024-12-01", "2025-06-01"),  # before n3 published
    ("cd-10", "Post",   "二孩", "2025-06-01", "2025-03-01"),
]

QTEXT = {
    "Pre":  "在{v}，{c}家庭每月的育儿补贴是多少？（提问时间{a}）",
    "Post": "在{v}，{c}家庭每月的育儿补贴是多少？（提问时间{a}）",
    "Unch": "在{a}查询：业务日期{v}时，{c}家庭每月的育儿补贴是多少？",
    "Win":  "在{a}查询：业务日期{v}（新标准已公布但尚未生效）时，{c}家庭每月的育儿补贴是多少？",
    "Unseen": "在{a}（相关上调公告尚未发布）询问业务日期{v}时{c}家庭的每月育儿补贴，应是多少？",
    "Inap": "在{a}，育儿补贴公告是否规定了{c}的标准？业务日期{v}。",
}


def build():
    qs = []
    for qid, qt, child, a, v in SPECS:
        ans = asof(LEDGER[child], D(a), D(v))
        gold, doc = ans[0], ans[3]
        qs.append(dict(qid=qid, domain="cond_demo", qtype=qt,
                       question=QTEXT[qt].format(a=a, v=v, c=child),
                       ask_date=a, val_date=v, gold_answer=gold, gold_span=SPAN[gold],
                       gold_doc=doc, target_condition={"eq": {}, "ne": {}},
                       family=f"childcare_{child}", split="test",
                       meta={"status": "synthetic condition-axis diagnostic"}))
    # Inap: a sub-population with no policy
    qs.append(dict(qid="cd-11", domain="cond_demo", qtype="Inap",
                   question="在2025-06-01，育儿补贴公告是否规定了无子女家庭的标准？业务日期2025-06-01。",
                   ask_date="2025-06-01", val_date="2025-06-01", gold_answer="无法确定",
                   gold_span="", gold_doc=None, target_condition={"eq": {}, "ne": {}},
                   family="inap", split="test", meta={"status": "synthetic condition-axis diagnostic"}))
    return qs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "notices.jsonl", "w", encoding="utf-8") as f:
        for n in NOTICES:
            n = dict(n, domain="cond_demo", family="childcare",
                     cited_ids=[], clause_ids=[])
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    qs = build()
    with open(OUT / "questions.jsonl", "w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"wrote {len(NOTICES)} notices, {len(qs)} questions -> {OUT}")
    print("by type:", dict(Counter(q["qtype"] for q in qs)))


if __name__ == "__main__":
    main()
