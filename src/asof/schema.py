"""Canonical records: Condition lattice, Fact, Notice, Question.

These mirror the paper's formulation (§Methodology / Problem Formulation):
a fact is  <o, phi, c, v, [valid), [visible), s, rho>  and an as-of query fixes
(t_ask, t_val, c0). Everything is JSON-serializable for the canonical benchmark.
"""
from __future__ import annotations

import enum
import itertools
from dataclasses import asdict, dataclass, field
from typing import Any

from .temporal import POS_INF, Date, Interval, fmt_date, parse_date


class RevisionType(str, enum.Enum):
    INSERT = "insert"
    REPLACE = "replace"
    DELETE = "delete"
    PARTIAL = "partial"       # new value holds only for part of the old condition
    UNRELATED = "unrelated"


class QType(str, enum.Enum):
    """The six diagnostic question types (paper §Experiments/Benchmark)."""
    PRE = "Pre"          # rule before the revision (historical)
    POST = "Post"        # rule after it
    UNCH = "Unch"        # a field in the same document that was NOT changed
    WIN = "Win"          # published but not yet effective (valid-time window)
    INAP = "Inap"        # object the rule does not apply to
    UNSEEN = "Unseen"    # asked before the revision was published (visible-time axis)


# --------------------------------------------------------------------------- #
# Condition lattice                                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Condition:
    """A conjunction of equality constraints, with optional per-key exclusions.

    A condition selects the cases a fact applies to (audience, product, loan
    type, disbursement window, doc version, ...). ``eq`` pins a dimension to a
    value; ``ne`` lists excluded values on a dimension (used to express the
    *retained* branch of a partial split: "c(g) minus c(f)").

    Two conditions are *compatible* (``~``) iff they can describe the same case,
    i.e. they have a common lower bound (meet) in the lattice.  ``top`` is the
    most general condition (empty constraints).
    """

    eq: dict[str, str] = field(default_factory=dict)
    ne: dict[str, frozenset[str]] = field(default_factory=dict)

    # -- construction -------------------------------------------------------- #
    @staticmethod
    def top() -> Condition:
        return Condition()

    @staticmethod
    def of(**kwargs: str) -> Condition:
        return Condition(eq=dict(kwargs))

    def is_top(self) -> bool:
        return not self.eq and not self.ne

    # -- lattice operations -------------------------------------------------- #
    def compatible(self, other: Condition) -> bool:
        """c ~ c': no shared dimension contradicts, and no eq violates an ne."""
        for k in self.eq.keys() & other.eq.keys():
            if self.eq[k] != other.eq[k]:
                return False
        for k, v in self.eq.items():
            if v in other.ne.get(k, frozenset()):
                return False
        for k, v in other.eq.items():
            if v in self.ne.get(k, frozenset()):
                return False
        return True

    def meet(self, other: Condition) -> Condition | None:
        """Greatest lower bound (merged constraints), or None if incompatible."""
        if not self.compatible(other):
            return None
        eq = {**self.eq, **other.eq}
        ne: dict[str, set[str]] = {k: set(v) for k, v in self.ne.items()}
        for k, v in other.ne.items():
            ne.setdefault(k, set()).update(v)
        # drop exclusions on dimensions now pinned by an equality
        ne = {k: frozenset(v) for k, v in ne.items() if k not in eq and v}
        return Condition(eq=eq, ne=ne)

    def difference(self, cover: Condition) -> list[Condition]:
        """Disjoint conjunctions whose union is ``self \\ cover``.

        The complement of (A AND B) is a union, not (NOT A AND NOT B).
        Prefixing each next complement with the preceding covered atoms avoids
        overlapping retained branches. An empty list means fully covered.
        """
        if self.meet(cover) is None:
            return [self]
        atoms = [(Condition.of(**{k: v}), Condition(ne={k: frozenset({v})}))
                 for k, v in sorted(cover.eq.items())]
        atoms += [(Condition(ne={k: frozenset({v})}), Condition.of(**{k: v}))
                  for k, values in sorted(cover.ne.items()) for v in sorted(values)]
        retained, prefix = [], self
        for atom, complement in atoms:
            branch = prefix.meet(complement)
            if branch is not None:
                retained.append(branch)
            prefix = prefix.meet(atom)
            if prefix is None:
                break
        return retained

    def minus(self, cover: Condition) -> Condition:
        """Single-branch convenience; use ``difference`` for a general split."""
        retained = self.difference(cover)
        if len(retained) != 1:
            raise ValueError("Condition difference needs zero or multiple branches; use difference()")
        return retained[0]

    # -- serialization ------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {"eq": dict(self.eq), "ne": {k: sorted(v) for k, v in self.ne.items()}}

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> Condition:
        if not d:
            return Condition.top()
        return Condition(
            eq=dict(d.get("eq", {})),
            ne={k: frozenset(v) for k, v in d.get("ne", {}).items()},
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if self.is_top():
            return "⊤"
        parts = [f"{k}={v}" for k, v in self.eq.items()]
        parts += [f"{k}∉{{{','.join(sorted(v))}}}" for k, v in self.ne.items()]
        return "∧".join(parts)


_ids = itertools.count(1)


# --------------------------------------------------------------------------- #
# Fact                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class Fact:
    """A field-level, bitemporal fact version (paper Eq. 1)."""

    obj: str                      # o  — object (procedure, loan type, function)
    field_name: str               # phi — field (office, rate, parameter)
    value: str                    # v
    valid: Interval               # [valid_start, valid_end)
    visible: Interval             # [vis_start, vis_end)  — the key fix vs a stamp
    condition: Condition = field(default_factory=Condition.top)  # c
    source_span: str = ""         # s  — supporting text span
    rho: str | None = None        # revising notice id (None for original)
    doc_id: str | None = None     # provenance for exact-id LocateImpact
    clause_id: str | None = None
    domain: str | None = None
    fact_id: str = ""

    def __post_init__(self) -> None:
        if not self.fact_id:
            self.fact_id = f"f{next(_ids):06d}"

    @property
    def is_current_belief(self) -> bool:
        """t_tau_end == ∞ : this version is the system's current belief."""
        return self.visible.end >= POS_INF

    @property
    def okey(self) -> tuple[str, str]:
        """(object, field) key used by LocateImpact vector recall."""
        return (self.obj, self.field_name)

    def copy_with(self, **changes: Any) -> Fact:
        data = {
            "obj": self.obj, "field_name": self.field_name, "value": self.value,
            "valid": self.valid, "visible": self.visible, "condition": self.condition,
            "source_span": self.source_span, "rho": self.rho, "doc_id": self.doc_id,
            "clause_id": self.clause_id, "domain": self.domain,
        }
        data.update(changes)
        f = Fact(**data)          # fresh fact_id
        return f

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id, "obj": self.obj, "field": self.field_name,
            "value": self.value, "condition": self.condition.to_dict(),
            "valid": self.valid.to_strs(), "visible": self.visible.to_strs(),
            "source_span": self.source_span, "rho": self.rho,
            "doc_id": self.doc_id, "clause_id": self.clause_id, "domain": self.domain,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Fact:
        f = Fact(
            obj=d["obj"], field_name=d["field"], value=d["value"],
            valid=Interval.of(*d["valid"]), visible=Interval.of(*d["visible"]),
            condition=Condition.from_dict(d.get("condition")),
            source_span=d.get("source_span", ""), rho=d.get("rho"),
            doc_id=d.get("doc_id"), clause_id=d.get("clause_id"),
            domain=d.get("domain"),
        )
        if d.get("fact_id"):
            f.fact_id = d["fact_id"]
        return f


# --------------------------------------------------------------------------- #
# Notice (an incoming document in the publication stream)                     #
# --------------------------------------------------------------------------- #
@dataclass
class Notice:
    doc_id: str
    published: Date               # publication date p (when it becomes knowable)
    text: str
    title: str = ""
    effective: Date | None = None  # effective date t_eff (may be None -> inferred)
    domain: str | None = None
    family: str | None = None     # version-chain / event family
    cited_ids: list[str] = field(default_factory=list)   # doc numbers it amends
    clause_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["published"] = fmt_date(self.published)
        d["effective"] = None if self.effective is None else fmt_date(self.effective)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Notice:
        return Notice(
            doc_id=d["doc_id"], published=parse_date(d["published"]), text=d.get("text", ""),
            title=d.get("title", ""),
            effective=None if d.get("effective") in (None, "", "INF") else parse_date(d["effective"]),
            domain=d.get("domain"), family=d.get("family"),
            cited_ids=list(d.get("cited_ids", [])), clause_ids=list(d.get("clause_ids", [])),
        )


# --------------------------------------------------------------------------- #
# Question (a diagnostic benchmark item)                                      #
# --------------------------------------------------------------------------- #
@dataclass
class Question:
    qid: str
    domain: str
    qtype: QType
    question: str
    ask_date: Date                # t_ask
    val_date: Date                # t_val (business date)
    gold_answer: str
    gold_span: str = ""           # gold source span (for the Joint metric)
    gold_doc: str | None = None   # gold source document id
    target_condition: Condition = field(default_factory=Condition.top)  # c0
    family: str | None = None     # event id — keep items of one event together in a split
    split: str = "test"           # train | dev | test
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid, "domain": self.domain, "qtype": self.qtype.value,
            "question": self.question, "ask_date": fmt_date(self.ask_date),
            "val_date": fmt_date(self.val_date), "gold_answer": self.gold_answer,
            "gold_span": self.gold_span, "gold_doc": self.gold_doc,
            "target_condition": self.target_condition.to_dict(),
            "family": self.family, "split": self.split, "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Question:
        return Question(
            qid=d["qid"], domain=d["domain"], qtype=QType(d["qtype"]),
            question=d["question"], ask_date=parse_date(d["ask_date"]),
            val_date=parse_date(d["val_date"]), gold_answer=d["gold_answer"],
            gold_span=d.get("gold_span", ""), gold_doc=d.get("gold_doc"),
            target_condition=Condition.from_dict(d.get("target_condition")),
            family=d.get("family"), split=d.get("split", "test"), meta=d.get("meta", {}),
        )
