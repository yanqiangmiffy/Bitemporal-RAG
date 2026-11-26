"""Common method interface. A method consumes the notice stream via update()
and answers questions via answer()."""
from __future__ import annotations

from dataclasses import dataclass

from ..schema import Notice, Question
from ..store.flim import UpdateLog


@dataclass
class Answer:
    text: str
    cited_span: str = ""
    doc: str | None = None
    fact_id: str | None = None


class Method:
    name: str = "base"

    def update(self, notice: Notice) -> UpdateLog | None:
        """Consume one incoming notice (maintenance step)."""
        return None

    def answer(self, q: Question) -> Answer:
        raise NotImplementedError

    # optional one-time setup hook (e.g. register notices for baselines)
    def finalize_setup(self) -> None:
        pass
