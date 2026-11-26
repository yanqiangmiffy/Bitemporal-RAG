"""Unified chat client.

One OpenAI-compatible client serves DeepSeek, vLLM and OpenAI (they differ only
by base_url/model/key). An Anthropic branch is included for the Claude provider.
All calls go through the SQLite cache first, so identical requests are free and
runs are reproducible.
"""
from __future__ import annotations

import json
import re
import time
import threading
from typing import Any

from ..config import Config, api_key
from .cache import Cache, request_key

Message = dict[str, str]


class LLMClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.provider = cfg.get("llm.provider", "deepseek")
        self.base_url = cfg.get("llm.base_url")
        self.temperature = float(cfg.get("llm.temperature", 0.0))
        self.max_tokens = int(cfg.get("llm.max_tokens", 2048))
        self.max_retries = int(cfg.get("llm.max_retries", 5))
        self.top_p = float(cfg.get("llm.top_p", 1.0))
        # provider-specific extra body (e.g. Qwen3: disable the <think> trace)
        self.extra_body = cfg.get("llm.extra_body") or None
        self.cache = Cache(cfg.path("cache")) if cfg.get("llm.cache", True) else None
        self._client = None  # lazy
        self._replica_clients = {}
        self._replica_next = 0
        self._backend_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self.stats = {"cache_hits": 0, "api_calls": 0, "prompt_tokens": 0,
                      "completion_tokens": 0, "api_seconds": 0.0}
        self.last_call = threading.local()

    # -- lazy backend -------------------------------------------------------- #
    def _backend(self):
        replicas = self.cfg.get("llm.replica_urls") or []
        if replicas:
            if self.provider == "anthropic" or not self.cfg.get("llm.revision"):
                raise ValueError("Replica routing requires an OpenAI-compatible model with a fixed weight revision")
            with self._backend_lock:
                endpoint = replicas[self._replica_next % len(replicas)]
                self._replica_next += 1
                if endpoint not in self._replica_clients:
                    self._replica_clients[endpoint] = self._openai_backend(endpoint)
                return self._replica_clients[endpoint]
        if self._client is not None:
            return self._client
        key = api_key(self.cfg)
        if self.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=key)
        else:  # deepseek / vllm / openai — all OpenAI-compatible
            self._client = self._openai_backend(self.base_url)
        return self._client

    def _openai_backend(self, endpoint):
        from openai import OpenAI
        return OpenAI(base_url=endpoint, api_key=api_key(self.cfg) or "EMPTY",
                      timeout=float(self.cfg.get("llm.request_timeout", 90.0)), max_retries=0)

    # -- core call ----------------------------------------------------------- #
    def chat(
        self,
        messages: list[Message],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        payload = {
            "cache_version": 2, "base_url": (self.base_url or "").rstrip("/"),
            "provider": self.provider, "model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens, "json": json_mode,
            "top_p": self.top_p, "extra_body": self.extra_body,
        }
        if self.cfg.get("llm.revision"):
            payload["model_revision"] = self.cfg.get("llm.revision")
        key = request_key("chat", payload)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                with self._stats_lock:
                    self.stats["cache_hits"] += 1
                self.last_call.info = {"cache_hit": True, "request_key": key,
                                       **hit.get("usage", {})}
                return hit["text"]

        text = _strip_think(self._call_backend(messages, model, temperature, max_tokens, json_mode))
        if not text:
            raise RuntimeError("LLM returned empty content")
        if json_mode:
            parse_json(text)
        usage = getattr(self.last_call, "info", {})
        self.last_call.info = {**usage, "cache_hit": False, "request_key": key}

        if self.cache is not None:
            self.cache.put(key, "chat", {"text": text, "usage": usage})
        return text

    def _call_backend(self, messages, model, temperature, max_tokens, json_mode) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if self.provider == "anthropic":
                    sys = "\n".join(m["content"] for m in messages if m["role"] == "system")
                    turns = [m for m in messages if m["role"] != "system"]
                    resp = self._backend().messages.create(
                        model=model, system=sys or None, messages=turns,
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    return resp.content[0].text
                kwargs: dict[str, Any] = dict(
                    model=model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens,
                    top_p=self.top_p,
                )
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body
                started = time.perf_counter()
                backend = self._backend()
                resp = backend.chat.completions.create(**kwargs)
                self.last_call.info = {
                    "api_seconds": time.perf_counter() - started,
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                    "finish_reason": resp.choices[0].finish_reason,
                    "endpoint": str(getattr(backend, "base_url", self.base_url)),
                }
                with self._stats_lock:
                    self.stats["api_calls"] += 1
                    self.stats["api_seconds"] += time.perf_counter() - started
                    if resp.usage:
                        self.stats["prompt_tokens"] += resp.usage.prompt_tokens
                        self.stats["completion_tokens"] += resp.usage.completion_tokens
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")

    # -- json convenience ---------------------------------------------------- #
    def chat_json(self, messages: list[Message], model: str, **kw: Any) -> Any:
        """Call and parse a JSON object/array, tolerating ```json fences."""
        raw = self.chat(messages, model, json_mode=True, **kw)
        return parse_json(raw)


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove a reasoning model's <think>...</think> trace (e.g. Qwen3)."""
    if not text:
        return text
    text = _THINK.sub("", text)
    # unclosed trace (truncated by max_tokens): drop everything up to a lone </think>
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def parse_json(raw: str) -> Any:
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # last resort: grab the outermost {...} or [...]
        m = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise
