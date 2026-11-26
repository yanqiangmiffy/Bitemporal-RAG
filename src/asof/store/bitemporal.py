"""The bitemporal fact store: append-only on the visible axis + as-of query.

This implements the paper's corrected model. A revision never rewrites a
version's valid interval in place; it *closes* the old belief on the visible
axis (t_tau_end <- p) and *appends* successors visible from p.  The as-of query
(Eq. 2) is an interval containment on BOTH axes, which is what lets a query
asked before p still see the pre-revision belief.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..schema import Condition, Fact
from ..temporal import NEG_INF, POS_INF, Date, Interval


@dataclass
class BitemporalStore:
    facts: list[Fact] = field(default_factory=list)
    # When False the store degrades to a single timeline (ablation: -visible axis):
    # every version is visible for all t_ask and closings truncate valid in place.
    use_visible_time: bool = True

    def __post_init__(self) -> None:
        # Secondary index: object -> its current-belief versions. Keeps
        # LocateImpact (and current_beliefs) proportional to the facts of ONE
        # object rather than the whole store, so a Flim update is O(|a|+k),
        # independent of |D| (paper Prop. 1). Rebuilt from any preloaded facts.
        self._by_obj: dict[str, list[Fact]] = {}
        for f in self.facts:
            if self._is_current(f):
                self._by_obj.setdefault(f.obj, []).append(f)

    def _is_current(self, f: Fact) -> bool:
        return True if not self.use_visible_time else f.is_current_belief

    # -- writes -------------------------------------------------------------- #
    def append(self, fact: Fact) -> Fact:
        if not self.use_visible_time:
            fact.visible = Interval(NEG_INF, POS_INF)   # always "current belief"
        self.facts.append(fact)
        if self._is_current(fact):
            self._by_obj.setdefault(fact.obj, []).append(fact)
        return fact

    def close_visible(self, g: Fact, p: Date) -> None:
        """Transaction-time close: record that belief g held only before p.

        With the visible axis on, this sets t_tau_end <- p and leaves the valid
        interval untouched (append-only). With it off, there is no visible axis,
        so a close must truncate valid in place (the pre-fix overwrite model).
        """
        if self.use_visible_time:
            g.visible = Interval(g.visible.start, p)
            lst = self._by_obj.get(g.obj)          # g is no longer a current belief
            if lst is not None:
                self._by_obj[g.obj] = [x for x in lst if x is not g]

    # -- as-of selection (Eq. 2) -------------------------------------------- #
    def query(
        self,
        t_ask: Date,
        t_val: Date,
        c0: Condition | None = None,
        use_visible_time: bool | None = None,
    ) -> list[Fact]:
        c0 = c0 or Condition.top()
        use_vis = self.use_visible_time if use_visible_time is None else use_visible_time
        out = []
        for f in self.facts:
            if use_vis and not f.visible.contains(t_ask):
                continue
            if not f.valid.contains(t_val):
                continue
            if not f.condition.compatible(c0):
                continue
            out.append(f)
        return out

    # -- views --------------------------------------------------------------- #
    def current_beliefs(self) -> list[Fact]:
        """Versions still believed now (t_tau_end == ∞)."""
        if not self.use_visible_time:
            return list(self.facts)
        return [f for lst in self._by_obj.values() for f in lst]

    def current_for_obj(self, obj: str) -> list[Fact]:
        """Current-belief versions of one object (indexed; O(facts of obj))."""
        if not self.use_visible_time:
            return [f for f in self.facts if f.obj == obj]
        return list(self._by_obj.get(obj, []))

    def __len__(self) -> int:
        return len(self.facts)

    def to_rows(self) -> list[dict]:
        return [f.to_dict() for f in self.facts]
