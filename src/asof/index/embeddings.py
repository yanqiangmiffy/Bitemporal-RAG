"""Dense embedding encoder.

Two backends, chosen by config:
  * local  (default): sentence-transformers (e.g. BAAI/bge-large-zh-v1.5), needs
    torch; lazy import so this module is safe on a CPU-only box.
  * api    (embeddings.provider: siliconflow / embeddings.api_base set): an
    OpenAI-compatible embeddings endpoint (e.g. SiliconFlow BAAI/bge-m3). No GPU.
Vectors are cached in SQLite. Local encoding reuses vectors across runs without
loading the model on a cache hit.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np

from ..config import Config
from ..llm.cache import Cache


class Embedder:
    def __init__(self, cfg: Config):
        e = cfg.get("embeddings", {})
        self.model_name = e.get("model", "BAAI/bge-large-zh-v1.5")
        self.revision = e.get("revision")
        self.provider = e.get("provider")
        self.api_base = e.get("api_base")
        self.device = e.get("device", "cpu")
        self.batch_size = int(e.get("batch_size", 64))
        self.normalize = bool(e.get("normalize", True))
        self.strict = bool(e.get("strict", False))
        self.query_instruction = e.get("query_instruction", "")
        self._api = bool(self.provider == "siliconflow" or self.api_base)
        self._key = os.environ.get(e.get("api_key_env", "SILICONFLOW_API_KEY"), "")
        self._cache = Cache(cfg.path("cache")) if cfg.get("llm.cache", True) else None
        self._dim = 1024  # bge-m3
        self._model = None
        self._local_namespace = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs = {"device": self.device, "local_files_only": Path(self.model_name).is_dir()}
            if self.revision:
                kwargs["revision"] = self.revision
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    @property
    def dim(self) -> int:
        return self._dim if self._api else self._load().get_sentence_embedding_dimension()

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if is_query and self.query_instruction and not self._api:
            texts = [self.query_instruction + t for t in texts]
        if self._api:
            return self._encode_api(texts)
        if not texts or self._cache is None:
            return self._encode_local(texts)

        keys = {text: self._local_key(text) for text in dict.fromkeys(texts)}
        cached = {text: self._cache.get_bytes(key) for text, key in keys.items()}
        vectors = {
            text: np.frombuffer(blob, dtype="<f4")
            for text, blob in cached.items() if blob is not None
        }
        missing = [text for text in keys if text not in vectors]
        if missing:
            encoded = self._encode_local(missing)
            if encoded.ndim != 2 or len(encoded) != len(missing) or not np.isfinite(encoded).all():
                raise ValueError("Local encoder returned invalid embeddings")
            vectors.update(zip(missing, encoded, strict=True))
            self._cache.put_bytes_many("emb_local", (
                (keys[text], np.asarray(vectors[text], dtype="<f4").tobytes()) for text in missing
            ))
        return np.stack([vectors[text] for text in texts]).astype(np.float32, copy=False)

    def _encode_local(self, texts: list[str]) -> np.ndarray:
        vecs = self._load().encode(
            texts, batch_size=self.batch_size, normalize_embeddings=self.normalize,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def _local_key(self, text: str) -> str:
        if self._local_namespace is None:
            model = Path(self.model_name)
            files = []
            if model.is_dir():
                model = model.resolve()
                for path in sorted(model.rglob("*")):
                    if path.is_file():
                        info = path.stat()
                        files.append((path.relative_to(model).as_posix(), info.st_size,
                                      info.st_mtime_ns))
            identity = [str(model) if files else self.model_name, self.revision, self.normalize, files]
            self._local_namespace = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        payload = (self._local_namespace + "\x00" + text).encode()
        return "emb-local-v1:" + hashlib.sha256(payload).hexdigest()

    # -- API backend --------------------------------------------------------- #
    def _key_of(self, t: str) -> str:
        # key by (backend, model, text): different models AND backends (local vs
        # API) never share a cache entry, even for the same model name.
        tag = json.dumps([2, self.provider or 'local', self.api_base, self.model_name,
                          self.revision, self.normalize])
        return "emb:" + hashlib.sha256((tag + "\x00" + t).encode()).hexdigest()

    def _encode_api(self, texts: list[str]) -> np.ndarray:
        out: list[list[float] | None] = [None] * len(texts)
        miss, midx = [], []
        for i, t in enumerate(texts):
            hit = self._cache.get(self._key_of(t)) if self._cache is not None else None
            if hit is not None:
                out[i] = hit["v"]
            else:
                miss.append(t); midx.append(i)
        for b in range(0, len(miss), self.batch_size):
            batch = miss[b:b + self.batch_size]
            vecs = self._call(batch)                       # [] on failure -> zeros
            if self.strict and (len(vecs) != len(batch) or
                                np.asarray(vecs).shape != (len(batch), self._dim) or
                                not np.isfinite(np.asarray(vecs)).all()):
                raise RuntimeError("Embedding endpoint returned incomplete or invalid vectors")
            for j, t in enumerate(batch):
                v = vecs[j] if j < len(vecs) else [0.0] * self._dim
                out[midx[b + j]] = v
                if self._cache is not None and vecs:
                    self._cache.put(self._key_of(t), "emb", {"v": v})
        arr = np.asarray([o if o is not None else [0.0] * self._dim for o in out], dtype=np.float32)
        if self.normalize and arr.size:
            arr = arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8, None)
        return arr

    def _call(self, batch: list[str]) -> list[list[float]]:
        body = json.dumps({"input": batch, "model": self.model_name}).encode()
        req = urllib.request.Request(
            self.api_base.rstrip("/") + "/embeddings", data=body,
            headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"})
        for attempt in range(5):
            try:
                r = json.load(urllib.request.urlopen(req, timeout=60))
                return [d["embedding"] for d in r["data"]]
            except Exception:
                time.sleep(min(2 ** attempt, 20))
        if self.strict:
            raise RuntimeError("Embedding API failed after retries")
        print("[embed] API failed after retries; using zero vectors (BM25 still active)")
        return []
