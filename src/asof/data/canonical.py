"""Write the unified (canonical) benchmark: one dir per domain with a notice
stream and a question set, both sorted into a stable, replayable order."""
from __future__ import annotations

from pathlib import Path

from ..jsonl import write_json, write_jsonl
from ..schema import Notice, Question
from ..temporal import fmt_date


def write_canonical(
    domain: str, notices: list[Notice], questions: list[Question], out_root: str | Path
) -> dict:
    out = Path(out_root) / domain
    # Notices in publication order (the stream Flim replays).
    notices = sorted(notices, key=lambda n: (n.published, n.doc_id))
    write_jsonl(out / "notices.jsonl", (n.to_dict() for n in notices))
    write_jsonl(out / "questions.jsonl", (q.to_dict() for q in questions))

    by_type: dict[str, int] = {}
    for q in questions:
        by_type[q.qtype.value] = by_type.get(q.qtype.value, 0) + 1
    stats = {
        "domain": domain,
        "notices": len(notices),
        "questions": len(questions),
        "questions_by_type": by_type,
        "span": [fmt_date(notices[0].published), fmt_date(notices[-1].published)] if notices else [],
    }
    write_json(out / "stats.json", stats)
    return stats
