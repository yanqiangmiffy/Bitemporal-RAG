import copy

import numpy as np

from asof.config import Config
from asof.index.embeddings import Embedder
from asof.llm.client import LLMClient
from types import SimpleNamespace


def config(tmp_path):
    return Config({"_root": str(tmp_path), "paths": {"cache": "inference.sqlite"},
                   "llm": {"provider": "vllm", "base_url": "https://one.example/v1",
                           "cache": True, "top_p": 0.8, "extra_body": {
                               "chat_template_kwargs": {"enable_thinking": False}}},
                   "embeddings": {"provider": "siliconflow", "api_base": "http://localhost/v1",
                                  "model": "BAAI/bge-m3", "normalize": True, "strict": True}})


def test_llm_cache_isolates_endpoint_sampling_thinking_and_weight_revision(tmp_path):
    cfg = config(tmp_path)
    messages = [{"role": "user", "content": "hello"}]
    calls = []

    def make(current):
        client = LLMClient(current)
        client._call_backend = lambda *args: calls.append(args) or '{"ok": true}'
        return client

    client = make(cfg)
    client.chat_json(messages, "model-a", max_tokens=64)
    client.chat_json(messages, "model-a", max_tokens=64)
    assert len(calls) == 1
    for field, value in [("base_url", "https://two.example/v1"), ("top_p", 0.9),
                         ("revision", "sha256:replaced-weights-with-same-served-name"),
                         ("extra_body", {"chat_template_kwargs": {"enable_thinking": True}})]:
        other = Config(copy.deepcopy(dict(cfg)))
        other["llm"][field] = value
        make(other).chat_json(messages, "model-a", max_tokens=64)
    assert len(calls) == 5


def test_embedding_cache_populates_an_initially_empty_database(tmp_path):
    emb = Embedder(config(tmp_path))
    calls = []
    emb._call = lambda texts: calls.append(texts) or np.ones((len(texts), 1024)).tolist()
    first = emb.encode(["document"])
    replay = emb.encode(["document"])
    assert len(calls) == 1
    np.testing.assert_array_equal(first, replay)
    assert len(emb._cache) == 1


def test_identical_weight_replicas_reuse_clients_and_alternate_requests(tmp_path, monkeypatch):
    import openai
    cfg = config(tmp_path)
    cfg["llm"].update(revision="fixed-weight-sha256", replica_urls=["http://a/v1", "http://b/v1"])
    created = []
    def fake(**kwargs):
        created.append(kwargs["base_url"])
        return SimpleNamespace(base_url=kwargs["base_url"])
    monkeypatch.setattr(openai, "OpenAI", fake)
    client = LLMClient(cfg)
    assert [client._backend().base_url for _ in range(4)] == ["http://a/v1", "http://b/v1"] * 2
    assert created == ["http://a/v1", "http://b/v1"]
