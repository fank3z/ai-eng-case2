"""Дві стратегії chunking:
- structure: по розділах документа (заголовки), таблиці й списки не розрізаються,
  великі розділи діляться по межах блоків; кожен чанк отримує контекстний заголовок.
- fixed: фіксоване вікно токенів з перекриттям (baseline для порівняння).
"""
from __future__ import annotations

import hashlib
import re

from ..textutil import count_tokens


def parse_blocks(text: str):
    """markdown -> [(section_path, block_text)]; блок = абзац / список / таблиця."""
    path: list[str] = []
    out = []
    for raw in re.split(r"\n{2,}", text):
        blk = raw.strip()
        if not blk:
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", blk.split("\n", 1)[0])
        if m and "\n" not in blk:
            lvl = len(m.group(1))
            path = path[: lvl - 1] + [m.group(2).strip()]
            continue
        if m:  # заголовок + список одразу під ним (без порожнього рядка)
            head, rest = blk.split("\n", 1)
            lvl = len(m.group(1))
            path = path[: lvl - 1] + [head.lstrip("#").strip()]
            blk = rest.strip()
        out.append((tuple(path), blk))
    return out


def _split_table(block: str, max_tokens: int) -> list[str]:
    lines = block.splitlines()
    head, body = lines[:2], lines[2:]
    parts, cur = [], []
    for row in body:
        if cur and count_tokens("\n".join(head + cur + [row])) > max_tokens:
            parts.append("\n".join(head + cur)); cur = []
        cur.append(row)
    if cur:
        parts.append("\n".join(head + cur))
    return parts


def chunk_structure(text: str, max_tokens: int = 300, min_tokens: int = 50) -> list[dict]:
    blocks = parse_blocks(text)
    sections: dict[tuple, list[str]] = {}
    for path, blk in blocks:
        sections.setdefault(path, []).append(blk)
    chunks = []
    for path, blks in sections.items():
        cur: list[str] = []
        for blk in blks:
            pieces = [blk]
            if count_tokens(blk) > max_tokens and blk.startswith("|"):
                pieces = _split_table(blk, max_tokens)     # таблицю ріжемо по рядках, шапку повторюємо
            for pc in pieces:
                if cur and count_tokens("\n\n".join(cur + [pc])) > max_tokens:
                    chunks.append({"section_path": list(path), "text": "\n\n".join(cur)}); cur = []
                cur.append(pc)
        if cur:
            chunks.append({"section_path": list(path), "text": "\n\n".join(cur)})
    # дрібні розділи (< min_tokens) зливаємо з наступним розділом того ж документа, якщо вміщається в max_tokens
    merged: list[dict] = []
    for ch in chunks:
        prev = merged[-1] if merged else None
        if prev and count_tokens(prev["text"]) < min_tokens and \
                count_tokens(prev["text"] + ch["text"]) + 10 <= max_tokens:
            prev["text"] = f"[{prev['section_path'][-1] if prev['section_path'] else ''}]\n{prev['text']}\n\n" \
                           f"[{ch['section_path'][-1] if ch['section_path'] else ''}]\n{ch['text']}" \
                           if not prev.get("merged") else \
                           f"{prev['text']}\n\n[{ch['section_path'][-1] if ch['section_path'] else ''}]\n{ch['text']}"
            prev["sections"] = prev.get("sections", [prev["section_path"]]) + [ch["section_path"]]
            prev["merged"] = True
        else:
            merged.append(dict(ch))
    return merged


def chunk_fixed(text: str, size: int = 120, overlap: int = 30) -> list[dict]:
    """Baseline: вікно `size` слів-токенів з перекриттям, ігноруючи структуру."""
    section_at = []
    words = []
    path: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if m:
            path = path[: len(m.group(1)) - 1] + [m.group(2).strip()]
        for w in line.split():
            words.append(w); section_at.append(tuple(path))
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, max(1, len(words)), step):
        piece = words[start:start + size]
        if not piece:
            break
        chunks.append({"section_path": list(section_at[start]), "text": " ".join(piece)})
        if start + size >= len(words):
            break
    return chunks


def finalize(doc_meta: dict, raw_chunks: list[dict], strategy: str) -> list[dict]:
    out = []
    title = doc_meta["title"]
    for i, ch in enumerate(raw_chunks):
        paths = ch.get("sections", [ch["section_path"]])
        names = [" > ".join([s for s in p if s != title]) or "(вступ)" for p in paths]
        section = "; ".join(dict.fromkeys(names))
        status = "чинна" if doc_meta.get("status") == "current" else "СКАСОВАНА"
        header = f"Документ: {title} | Версія: {doc_meta.get('version', '')} ({status}) | Розділ: {section}"
        out.append({
            "chunk_id": f"{doc_meta['doc_id']}#{i:02d}",
            "doc_id": doc_meta["doc_id"], "title": title, "section": section,
            "source": doc_meta["source"], "format": doc_meta["format"],
            "version": doc_meta.get("version", ""), "date": doc_meta.get("date", ""),
            "status": doc_meta.get("status", "current"), "superseded_by": doc_meta.get("superseded_by"),
            "doc_type": doc_meta.get("doc_type", ""), "access": doc_meta.get("access", "noc"),
            "doc_hash": doc_meta["doc_hash"], "strategy": strategy,
            "text": ch["text"],
            "index_text": header + "\n" + ch["text"],      # contextual chunk header (для BM25 і embeddings)
            "chunk_hash": hashlib.sha256(ch["text"].encode()).hexdigest()[:16],
            "n_tokens": count_tokens(ch["text"]),
        })
    return out
