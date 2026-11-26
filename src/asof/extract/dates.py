"""Effective-date normalization.

Explicit dates are parsed directly. Relative forms ("自下季度起", "次月1日")
are a known limitation (paper §Conclusion/Limitations); we handle the common
"自YYYY年M月D日起" pattern and leave harder cases to annotation.
"""
from __future__ import annotations

import re

from ..schema import Notice
from ..temporal import Date, parse_date

_EFF = re.compile(r"自\s*(20\d{2}[年\-/.]\d{1,2}[月\-/.]\d{1,2})\s*[日]?\s*起")


def normalize_effective(notice: Notice, llm_value: str | None) -> Date | None:
    if llm_value and str(llm_value).lower() not in ("null", "none", ""):
        try:
            return parse_date(llm_value)
        except Exception:
            pass
    m = _EFF.search(notice.text or "")
    if m:
        try:
            return parse_date(m.group(1))
        except Exception:
            return None
    return None
