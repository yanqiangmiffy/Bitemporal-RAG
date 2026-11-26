"""Beijing housing-fund: dated text/html notices + a curated seed question set.

The seed questions encode *documented* value changes (first-home fund rate
2.85% -> 2.60% effective 2025-05-08; second-home min down-payment 30% -> 25%
effective 2025-12-24). They are marked split='dev' and meta.status='auto-seed'
and MUST be verified against the source text before being reported (see paper
annotation protocol). They exist so the pipeline is runnable end-to-end.
"""
from __future__ import annotations

from pathlib import Path

from ...schema import Condition, Notice, QType, Question
from ...temporal import parse_date as D
from .common import date_from_filename, extract_first_date, load_text_any

DOMAIN = "beijing_housing"


def _family(name: str) -> str:
    if name.startswith("downpayment"):
        return "beijing_downpayment"
    if name.startswith("pboc"):
        return "pboc_rate"
    return "beijing_rate"


def _load_notices(root: Path) -> list[Notice]:
    notices: dict[str, Notice] = {}
    for p in sorted(root.glob("*")):
        if p.suffix.lower() not in (".txt", ".html", ".htm"):
            continue
        if p.stem == "rate_history":          # after-the-fact summary, not a notice
            continue
        text = load_text_any(p)
        if not text.strip():
            continue
        pub = extract_first_date(text) or date_from_filename(p.name)
        if pub is None:
            continue
        doc_id = p.stem
        # prefer .txt over .html when both carry the same notice
        if doc_id in notices and p.suffix.lower() != ".txt":
            continue
        notices[doc_id] = Notice(
            doc_id=doc_id, published=D(pub), title=p.stem,
            text=text, domain=DOMAIN, family=_family(p.name),
        )
    return list(notices.values())


# Curated seed items (documented values; verify before reporting).
_SEED = [
    # first-home fund rate, >5yr: 2.85% (2024-05-18) -> 2.60% (eff 2025-05-08)
    ("bjh-rate-pre",  QType.PRE,    "2025-07-01", "2025-01-01", "2.85%",
     "rate_20240518", "首套个人住房公积金贷款5年以上利率为2.85%"),
    ("bjh-rate-post", QType.POST,   "2025-07-01", "2025-07-01", "2.60%",
     "rate_20250508", "2025年5月8日（含）之后新发放的……5年以上首套……2.6%"),
    ("bjh-rate-win",  QType.WIN,    "2025-05-08", "2025-05-07", "2.85%",
     "rate_20240518", "2025年5月8日之前发放的贷款执行原利率2.85%"),
    ("bjh-rate-unseen", QType.UNSEEN, "2025-05-01", "2025-07-01", "2.85%",
     "rate_20240518", "提问日2025-05-01时下调公告尚未发布，仍为2.85%"),
    # second-home min down-payment: 30% (2025-08-08) -> 25% (eff 2025-12-24)
    ("bjh-down-pre",  QType.PRE,    "2026-02-01", "2025-10-01", "30%",
     "downpayment_20250808", "第二套住房公积金贷款最低首付比例为30%"),
    ("bjh-down-post", QType.POST,   "2026-02-01", "2026-02-01", "25%",
     "downpayment_20251224", "第二套住房公积金贷款最低首付比例下调为25%"),
    ("bjh-down-unseen", QType.UNSEEN, "2025-12-01", "2026-02-01", "30%",
     "downpayment_20250808", "提问日2025-12-01时下调至25%的公告尚未发布，仍为30%"),
    # a field the fund-rate notice does not govern
    ("bjh-inap", QType.INAP, "2025-07-01", "2025-07-01", "无法确定",
     "rate_20250508", "商业性个人住房贷款利率不由公积金利率公告规定"),
]


def _seed_questions() -> list[Question]:
    out = []
    for qid, qt, ask, val, gold, doc, span in _SEED:
        cond = Condition.of(loan="公积金") if "rate" in qid or "down" in qid else Condition.top()
        out.append(Question(
            qid=qid, domain=DOMAIN, qtype=qt, question="", ask_date=D(ask), val_date=D(val),
            gold_answer=gold, gold_span=span, gold_doc=doc, target_condition=cond,
            family="beijing_rate" if "rate" in qid else "beijing_downpayment",
            split="dev", meta={"status": "auto-seed; verify against source text"},
        ))
    # fill a human-readable question string
    tmpl = {
        QType.PRE: "业务日期{v}（提问日{a}）适用的{what}是多少？",
        QType.POST: "业务日期{v}（提问日{a}）适用的{what}是多少？",
        QType.WIN: "提问日{a}、业务日期{v}（新规尚未生效）适用的{what}是多少？",
        QType.UNSEEN: "提问日{a}（相关下调公告尚未发布）询问业务日期{v}的{what}，应是多少？",
        QType.INAP: "提问日{a}，公积金利率公告是否规定了商业贷款利率？",
    }
    for q in out:
        what = "首套公积金贷款5年以上利率" if "rate" in q.qid else "第二套公积金贷款最低首付比例"
        from ...temporal import fmt_date
        q.question = tmpl[q.qtype].format(a=fmt_date(q.ask_date), v=fmt_date(q.val_date), what=what)
    return out


def load(datasets_root: str | Path) -> tuple[list[Notice], list[Question]]:
    root = Path(datasets_root) / DOMAIN
    return _load_notices(root), _seed_questions()
