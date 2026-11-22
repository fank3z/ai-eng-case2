"""Embeddings з кешем (stretch: кешування embeddings за хешем тексту).
Бекенди:
- fastembed  — локальна мультимовна модель (ONNX, CPU), за замовчуванням;
- gemini     — Gemini Embedding API (free tier);
- tfidf      — запасний варіант без завантаження моделей (char n-gram TF-IDF + SVD), для smoke-тестів."""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("embeddings")


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class Embedder:
    name = "base"

    def __init__(self, cache_dir: str = ".cache/emb"):
        self.cache_path = Path(cache_dir) / f"{self.name.replace('/', '_')}.pkl"
        self.cache = pickle.loads(self.cache_path.read_bytes()) if self.cache_path.exists() else {}

    def _key(self, text, kind):
        return hashlib.sha256(f"{kind}|{text}".encode()).hexdigest()

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        keys = [self._key(t, kind) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        if missing:
            vecs = self._embed([texts[i] for i in missing], kind)
            for i, v in zip(missing, vecs):
                self.cache[keys[i]] = np.asarray(v, dtype=np.float32)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_bytes(pickle.dumps(self.cache))
        return _l2(np.stack([self.cache[k] for k in keys]))

    def fit(self, corpus: list[str]):  # лише для tfidf
        pass

    def _embed(self, texts, kind):
        raise NotImplementedError


class FastEmbedder(Embedder):
    def __init__(self, model: str, models_dir: str = ".models", **kw):
        self.name = f"fastembed:{model}"
        super().__init__(**kw)
        self.model_name, self.models_dir, self._m = model, models_dir, None

    def _embed(self, texts, kind):
        if self._m is None:
            from fastembed import TextEmbedding
            self._m = TextEmbedding(model_name=self.model_name, cache_dir=self.models_dir)
        prefix = {"query": "query: ", "passage": "passage: "}[kind] if "e5" in self.model_name else ""
        return list(self._m.embed([prefix + t for t in texts]))


class GeminiEmbedder(Embedder):
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:batchEmbedContents"

    def __init__(self, model: str = "gemini-embedding-001", **kw):
        self.name = f"gemini:{model}"
        super().__init__(**kw)
        self.model = model

    def _embed(self, texts, kind):
        import httpx
        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        out = []
        for i in range(0, len(texts), 50):
            batch = texts[i:i + 50]
            body = {"requests": [{"model": f"models/{self.model}", "taskType": task,
                                  "content": {"parts": [{"text": t}]}} for t in batch]}
            for attempt in range(5):
                r = httpx.post(self.URL.format(m=self.model), json=body, timeout=60,
                               headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
                if r.status_code in (429, 500, 503):
                    time.sleep(8 * 2 ** attempt); continue
                r.raise_for_status(); break
            out += [e["values"] for e in r.json()["embeddings"]]
        return out


class TfidfEmbedder(Embedder):
    """Без моделей: char 3–5-грами + SVD(256). Не семантичний, але дозволяє прогнати пайплайн офлайн."""
    name = "tfidf-char-svd"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.cache = {}  # залежить від корпусу -> не кешуємо між запусками
        self._vec = self._svd = None

    def fit(self, corpus):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
        x = self._vec.fit_transform(corpus)
        self._svd = TruncatedSVD(n_components=min(256, x.shape[1] - 1, x.shape[0] - 1), random_state=0).fit(x)

    def embed(self, texts, kind="passage"):
        return _l2(self._svd.transform(self._vec.transform(texts)).astype(np.float32))


def build_embedder(cfg: dict) -> Embedder:
    b = cfg["backend"]
    if b == "fastembed":
        return FastEmbedder(cfg["model"], cfg.get("models_dir", ".models"))
    if b == "gemini":
        return GeminiEmbedder(cfg.get("gemini_model", "gemini-embedding-001"))
    if b == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Невідомий embeddings backend: {b}")
