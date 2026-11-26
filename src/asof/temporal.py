"""Dates and half-open intervals on the two bitemporal axes.

A fact carries two half-open intervals (paper Eq. 1):
  * valid-time   [valid_start, valid_end)   -- when the value holds in the world
  * visible-time [vis_start,  vis_end)      -- when the system believed this version

We represent a date as ``datetime.date`` and use ``date.max`` as +infinity
(an open-ended / currently-held interval) and ``date.min`` as -infinity.
Serialization maps these sentinels to the strings "INF" / "-INF".
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

from dateutil import parser as _dup

POS_INF = _dt.date.max          # 9999-12-31, used for open-ended intervals (t_e = ∞)
NEG_INF = _dt.date.min          # 0001-01-01

Date = _dt.date


def parse_date(s: str | _dt.date | None) -> Date:
    """Parse a date. ``None``/"" -> POS_INF; "INF"/"-INF" sentinels supported.

    Accepts ISO strings and common Chinese forms like '2025年5月8日' / '2025-05-08'.
    """
    if s is None or s == "":
        return POS_INF
    if isinstance(s, _dt.date):
        return s
    t = str(s).strip()
    if t in ("INF", "+INF", "inf", "∞"):
        return POS_INF
    if t in ("-INF", "-inf", "-∞"):
        return NEG_INF
    t = (t.replace("年", "-").replace("月", "-").replace("日", "")
          .replace("/", "-").replace(".", "-").strip("-"))
    return _dup.parse(t, yearfirst=True).date()


def fmt_date(d: Date) -> str:
    if d >= POS_INF:
        return "INF"
    if d <= NEG_INF:
        return "-INF"
    return d.isoformat()


@dataclass(frozen=True)
class Interval:
    """A half-open interval [start, end) on one time axis."""

    start: Date
    end: Date

    def contains(self, p: Date) -> bool:
        return self.start <= p < self.end

    def overlaps(self, other: Interval) -> bool:
        return self.start < other.end and other.start < self.end

    def is_open(self) -> bool:
        return self.end >= POS_INF

    def to_strs(self) -> list[str]:
        return [fmt_date(self.start), fmt_date(self.end)]

    @staticmethod
    def of(start, end) -> Interval:
        return Interval(parse_date(start), parse_date(end))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{fmt_date(self.start)}, {fmt_date(self.end)})"
