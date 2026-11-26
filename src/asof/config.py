"""Config loading: deep-merge a base YAML with provider/ablation overrides.

    cfg = load_config("configs/default.yaml",
                      overrides=["configs/providers/deepseek.yaml",
                                 "configs/ablations/no_visible_time.yaml"])
    cfg["llm"]["extractor_model"]        # dotted access also works: cfg.get("llm.provider")
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict with dotted ``get`` and attribute-style section access."""

    def get(self, dotted: str, default: Any = None) -> Any:  # type: ignore[override]
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    @property
    def root(self) -> Path:
        return Path(self["_root"])

    def path(self, key: str) -> Path:
        """Resolve a configured path relative to the repo root."""
        p = Path(self["paths"][key])
        return p if p.is_absolute() else (self.root / p).resolve()


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(base: str | Path, overrides: list[str | Path] | None = None) -> Config:
    base = Path(base).resolve()
    with open(base, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for ov in overrides or []:
        with open(ov, "r", encoding="utf-8") as f:
            data = _deep_merge(data, yaml.safe_load(f) or {})
    # repo root = parent of the configs/ directory of the base file
    data["_root"] = str(base.parent.parent)
    return Config(data)


def api_key(cfg: Config) -> str:
    """Read the LLM api key from the env var named by llm.api_key_env."""
    env = cfg.get("llm.api_key_env", "DEEPSEEK_API_KEY")
    key = os.environ.get(env, "")
    if not key:
        # vLLM accepts any token; default to a placeholder so local runs work.
        key = "EMPTY" if cfg.get("llm.provider") == "vllm" else ""
    return key
