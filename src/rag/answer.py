"""RAG: пошук -> контекст у межах бюджету токенів -> відповідь моделі з цитуванням джерел.
Захист від вигадування на трьох рівнях:
1) немає знайдених чанків після фільтрів -> «не знаю» без виклику LLM;
2) (опційно) найкращий reranker score нижче порогу -> «не знаю» без виклику LLM;
3) промпт + схема з полем not_found; посилання на неіснуючі джерела відкидаються."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ..prompts import Prompt
from ..retrieval.search import Searcher
from ..textutil import count_tokens
from .llm import LLM

log = logging.getLogger("rag")
NOT_FOUND_TEXT = "У доступних документах відповіді на це питання немає."

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "not_found": {"type": "boolean", "description": "true, якщо в джерелах немає відповіді"},
        "answer": {"type": "string", "description": "Відповідь українською; порожньо, якщо not_found"},
        "claims": {"type": "array", "description": "Кожне твердження відповіді з посиланнями на джерела",
                   "items": {"type": "object",
                             "properties": {"text": {"type": "string"},
                                            "sources": {"type": "array", "items": {"type": "string"},
                                                        "description": "Ідентифікатори джерел, напр. S1"}},
                             "required": ["text", "sources"]}},
    },
    "required": ["not_found", "answer", "claims"],
}


@dataclass
class RagAnswer:
    question: str
    answer: str
    not_found: bool
    claims: list = field(default_factory=list)
    citations: list = field(default_factory=list)      # [{sid, chunk_id, doc_id, title, section, version}]
    context: list = field(default_factory=list)        # чанки, що пішли в контекст
    retrieved: list = field(default_factory=list)      # усі знайдені chunk_id
    dropped_by_budget: list = field(default_factory=list)
    context_tokens: int = 0
    guard: str | None = None                           # чому «не знаю» без LLM
    invalid_sources: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    error: str | None = None

    def render(self) -> str:
        if self.not_found:
            return NOT_FOUND_TEXT
        lines = [self.answer, "", "Джерела:"]
        for c in self.citations:
            lines.append(f"[{c['sid']}] {c['title']} — {c['section']} (версія {c['version']}, {c['source']})")
        return "\n".join(lines)


def build_context(hits, budget_tokens: int):
    ctx, dropped, used = [], [], 0
    for h in hits:
        block_tokens = count_tokens(h.chunk["text"]) + 25   # + заголовок джерела
        if used + block_tokens > budget_tokens:
            dropped.append(h.chunk["chunk_id"]); continue
        ctx.append(h); used += block_tokens
    return ctx, dropped, used


def format_sources(ctx) -> str:
    parts = []
    for i, h in enumerate(ctx, 1):
        c = h.chunk
        status = "чинна" if c["status"] == "current" else f"СКАСОВАНА, замінена {c.get('superseded_by') or ''}".strip()
        parts.append(f"[S{i}] {c['title']} — {c['section']} | версія {c['version']} від {c['date']} ({status})\n{c['text']}")
    return "\n\n".join(parts)


def answer(question: str, searcher: Searcher, llm: LLM, prompt: Prompt, cfg: dict,
           filters: dict | None = None) -> RagAnswer:
    rc = cfg["rag"]
    filters = cfg["search"].get("default_filters", {}) if filters is None else filters
    hits = searcher.search(question, k=rc["top_k"], mode=rc.get("mode", "hybrid_rerank"), filters=filters)
    res = RagAnswer(question, "", True, retrieved=[h.chunk["chunk_id"] for h in hits])
    if not hits:
        res.guard = "no_hits_after_filters"
        return res
    thr = rc.get("min_rerank_score")
    if thr is not None and "rerank_score" in hits[0].ranks and hits[0].ranks["rerank_score"] < thr:
        res.guard = f"low_rerank_score ({hits[0].ranks['rerank_score']:.2f} < {thr})"
        return res
    ctx, res.dropped_by_budget, res.context_tokens = build_context(hits, rc["context_budget_tokens"])
    res.context = [h.chunk["chunk_id"] for h in ctx]
    system, user = prompt.render(question=question, sources=format_sources(ctx))
    try:
        out = llm.generate(system, user, ANSWER_SCHEMA, temperature=rc.get("temperature", 0.0))
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}"
        log.error("LLM помилка: %s", res.error)
        return res
    d = out.data
    res.input_tokens, res.output_tokens, res.latency_s, res.cached = out.input_tokens, out.output_tokens, out.latency_s, out.cached
    res.not_found = bool(d.get("not_found")) or not d.get("answer", "").strip()
    if res.not_found:
        return res
    res.answer = d.get("answer", "").strip()
    valid = {f"S{i}" for i in range(1, len(ctx) + 1)}
    used = []
    for cl in d.get("claims", []):
        srcs = [s.strip("[] ") for s in cl.get("sources", [])]
        bad = [s for s in srcs if s not in valid]
        res.invalid_sources += bad
        good = [s for s in srcs if s in valid]
        res.claims.append({"text": cl.get("text", ""), "sources": good})
        used += good
    for sid in dict.fromkeys(used + re.findall(r"\[(S\d+)\]", res.answer)):
        if sid in valid:
            c = ctx[int(sid[1:]) - 1].chunk
            res.citations.append({"sid": sid, **{k: c[k] for k in ["chunk_id", "doc_id", "title", "section", "version", "source"]}})
    return res
