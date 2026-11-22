"""Очищення, нормалізація та дедуплікація документів."""
from __future__ import annotations

import hashlib
import re
import unicodedata

from ..textutil import APOSTROPHES

INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"), None)  # soft hyphen, zero-width
SPACES = re.compile(r"[\u00a0\u2007\u2009\u202f\t]")
BOILERPLATE_LINES = [
    re.compile(r"^\s*(сторінка|page)\s*\d+(\s*(з|of)\s*\d+)?\s*$", re.I),
    re.compile(r"^\s*©.*$"),
    re.compile(r"^\s*(для внутрішнього використання|confidential)\s*$", re.I),
]


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(INVISIBLE)
    for a in APOSTROPHES:
        text = text.replace(a, "'")
    text = SPACES.sub(" ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)                  # перенос слова через рядок
    lines = [ln.rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if not any(p.match(ln) for p in BOILERPLATE_LINES)]
    text = "\n".join(re.sub(r" {2,}", " ", ln) for ln in lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def content_hash(text: str) -> str:
    """Хеш змісту, нечутливий до пробілів і регістру -> точні дублікати навіть з різним форматуванням."""
    key = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def shingles(text: str, n: int = 5) -> set:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def deduplicate(docs: list[dict], near_threshold: float = 0.85) -> list[dict]:
    """docs: [{'doc_hash', 'text', 'date', ...}] -> проставляє duplicate_of / dedup_reason.
    Пріоритет залишити: новіша дата, потім коротший шлях (оригінал, а не «копія»)."""
    order = sorted(range(len(docs)), key=lambda i: (docs[i].get("date", ""), -len(docs[i]["source"])), reverse=True)
    kept: list[int] = []
    sh = {}
    for i in order:
        d = docs[i]
        d["duplicate_of"], d["dedup_reason"] = None, None
        for j in kept:
            if docs[j]["doc_hash"] == d["doc_hash"]:
                d["duplicate_of"], d["dedup_reason"] = docs[j]["source"], "exact (content hash)"
                break
            sh.setdefault(j, shingles(docs[j]["text"]))
            sh.setdefault(i, shingles(d["text"]))
            sim = jaccard(sh[i], sh[j])
            if sim >= near_threshold:
                d["duplicate_of"], d["dedup_reason"] = docs[j]["source"], f"near-duplicate (Jaccard 5-shingles = {sim:.2f})"
                break
        if d["duplicate_of"] is None:
            kept.append(i)
    return docs
