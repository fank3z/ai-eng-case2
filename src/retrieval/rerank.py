"""Reranking кандидатів: локальний cross-encoder (fastembed) або без reranking."""
from __future__ import annotations

import logging

log = logging.getLogger("rerank")


class CrossEncoderReranker:
    def __init__(self, model: str, models_dir: str = ".models"):
        self.model_name, self.models_dir, self._m = model, models_dir, None
        self.name = f"cross-encoder:{model}"

    def scores(self, query: str, texts: list[str]) -> list[float]:
        if self._m is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._m = TextCrossEncoder(model_name=self.model_name, cache_dir=self.models_dir)
        return [float(s) for s in self._m.rerank(query, texts)]


class OverlapReranker:
    """Запасний офлайн-reranker для smoke-тестів: частка термінів запиту, присутніх у чанку."""
    name = "lexical-overlap"

    def scores(self, query, texts):
        from .bm25 import tokenize
        q = set(tokenize(query))
        return [len(q & set(tokenize(t))) / max(1, len(q)) for t in texts]


def build_reranker(cfg: dict):
    b = cfg.get("backend", "none")
    if b == "cross_encoder":
        return CrossEncoderReranker(cfg["model"], cfg.get("models_dir", ".models"))
    if b == "overlap":
        return OverlapReranker()
    return None
