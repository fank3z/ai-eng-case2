"""Hybrid search: BM25 + векторний пошук -> Reciprocal Rank Fusion -> (опційно) reranking.
Режими: bm25 | vector | hybrid | hybrid_rerank."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .bm25 import BM25
from .embeddings import build_embedder
from .rerank import build_reranker
from .store import VectorStore, match_filters

log = logging.getLogger("search")


@dataclass
class Hit:
    chunk: dict
    score: float
    ranks: dict


def load_chunks(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


class Searcher:
    def __init__(self, chunks: list[dict], cfg: dict, index_dir: Path | None = None, rebuild: bool = False):
        self.cfg, self.chunks = cfg, chunks
        texts = [c["index_text"] for c in chunks]
        self.bm25 = BM25(texts)
        self.embedder = build_embedder(cfg["embeddings"])
        self.embedder.fit(texts)
        stale = True
        if index_dir and not rebuild and (index_dir / "payloads.json").exists():
            st = VectorStore.load(index_dir)
            stale = [p["chunk_hash"] for p in st.payloads] != [c["chunk_hash"] for c in chunks] \
                or getattr(self.embedder, "name", "") == "tfidf-char-svd"
            if not stale:
                self.store = st
        if stale:
            vecs = self.embedder.embed(texts, "passage")
            self.store = VectorStore(vecs.shape[1])
            self.store.add(vecs, [{k: v for k, v in c.items() if k not in ("text", "index_text")} for c in chunks])
            if index_dir:
                self.store.save(index_dir)
        self.reranker = build_reranker(cfg.get("rerank", {}))

    def search(self, query: str, k: int = 5, mode: str = "hybrid_rerank", filters: dict | None = None) -> list[Hit]:
        sc = self.cfg["search"]
        n_cand = sc.get("candidates", 30)
        allowed = {i for i, c in enumerate(self.chunks) if match_filters(c, filters)}
        if not allowed:
            return []
        ranks: dict[int, dict] = {}
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            s = self.bm25.scores(query)
            order = sorted((i for i in allowed if s[i] > 0), key=lambda i: -s[i])[:n_cand]
            for r, i in enumerate(order):
                ranks.setdefault(i, {})["bm25"] = r + 1
        if mode in ("vector", "hybrid", "hybrid_rerank"):
            qv = self.embedder.embed([query], "query")[0]
            for r, (i, _) in enumerate(self.store.search(qv, n_cand, filters)):
                ranks.setdefault(i, {})["vector"] = r + 1
        rrf_k = sc.get("rrf_k", 60)
        w = sc.get("weights", {"bm25": 1.0, "vector": 1.0})
        fused = {i: sum(w.get(src, 1.0) / (rrf_k + r) for src, r in rk.items()) for i, rk in ranks.items()}
        cands = sorted(fused, key=lambda i: -fused[i])
        if mode == "hybrid_rerank" and self.reranker is not None:
            top = cands[: sc.get("rerank_top", 20)]
            scores = self.reranker.scores(query, [self.chunks[i]["index_text"] for i in top])
            for i, s in zip(top, scores):
                ranks[i]["rerank_score"] = s
            ranked = sorted(zip(top, scores), key=lambda x: -x[1])
            return [Hit(self.chunks[i], s, ranks[i]) for i, s in ranked[:k]]
        return [Hit(self.chunks[i], fused[i], ranks[i]) for i in cands[:k]]
