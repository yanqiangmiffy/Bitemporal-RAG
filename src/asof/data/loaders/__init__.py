"""Registry of domain loaders. Each module exposes DOMAIN and load(root)."""
from __future__ import annotations

from . import beijing_housing, campus, cn_state_council, python_pathlib, safe_fx

_MODULES = (campus, cn_state_council, beijing_housing, python_pathlib, safe_fx)
LOADERS = {m.DOMAIN: m.load for m in _MODULES}

__all__ = ["LOADERS"]
