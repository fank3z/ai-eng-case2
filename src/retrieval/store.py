"""Векторна база: FAISS (IndexFlatIP, косинусна близькість на нормалізованих векторах) + payload-метадані.
Якщо faiss не встановлено — той самий інтерфейс на numpy. Фільтри за метаданими застосовуються
як пре-фільтр (дозволені id) до ранжування."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


def match_filters(payload: dict, filters: dict | None) -> bool:
    """filters: {"status": "current", "access": ["noc", "field"], "date_from": "2026-01-01"}"""
    for k, v in (filters or {}).items():
        if v is None:
            continue
        if k == "date_from":
            if payload.get("date", "") < v:
                return False
        elif k == "date_to":
            if payload.get("date", "") > v:
                return False
        elif isinstance(v, (list, tuple, set)):
            if payload.get(k) not in v:
                return False
        elif payload.get(k) != v:
            return False
    return True


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.payloads: list[dict] = []
        self.backend = "faiss" if faiss else "numpy"
        self._index = faiss.IndexFlatIP(dim) if faiss else None
        self._mat = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray, payloads: list[dict]):
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self._index is not None:
            self._index.add(vectors)
        self._mat = np.vstack([self._mat, vectors])
        self.payloads += payloads

    def search(self, qvec: np.ndarray, k: int, filters: dict | None = None) -> list[tuple[int, float]]:
        allowed = [i for i, p in enumerate(self.payloads) if match_filters(p, filters)]
        if not allowed:
            return []
        q = np.ascontiguousarray(qvec.reshape(1, -1), dtype=np.float32)
        if self._index is not None:
            # пре-фільтр через IDSelector, якщо доступний; інакше пошук по всіх + відсікання
            try:
                sel = faiss.IDSelectorBatch(np.array(allowed, dtype=np.int64))
                D, I = self._index.search(q, min(k, len(allowed)), params=faiss.SearchParameters(sel=sel))
                return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i >= 0]
            except Exception:  # стара версія faiss без SearchParameters
                D, I = self._index.search(q, len(self.payloads))
                ok = set(allowed)
                return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i in ok][:k]
        sims = self._mat[allowed] @ q[0]
        order = np.argsort(-sims)[:k]
        return [(allowed[j], float(sims[j])) for j in order]

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self._mat)
        if self._index is not None:
            faiss.write_index(self._index, str(path / "index.faiss"))
        (path / "payloads.json").write_text(json.dumps(self.payloads, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        mat = np.load(path / "vectors.npy")
        st = cls(mat.shape[1])
        st.add(mat, json.loads((path / "payloads.json").read_text(encoding="utf-8")))
        return st
