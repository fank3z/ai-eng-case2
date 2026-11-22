"""Scoring-функції для RAG.
Детерміновані: key facts, правильне джерело, «не знаю» де треба, заборонені (застарілі) значення, валідність цитат.
LLM-as-a-judge: коректність відносно еталону; faithfulness — чи кожне твердження підтверджене контекстом."""
from __future__ import annotations

from src.textutil import norm_for_match

# ---------------- детерміновані ----------------

def contains_alt(text: str, fact: str) -> bool:
    """fact може містити альтернативи через «|»: '2,0|2.0'."""
    t = norm_for_match(text)
    return any(norm_for_match(a) in t for a in fact.split("|"))


def key_fact_recall(answer: str, key_facts: list[str]) -> float | None:
    if not key_facts:
        return None
    return sum(contains_alt(answer, f) for f in key_facts) / len(key_facts)


def source_hit(cited_doc_ids: list[str], expected: list[str]) -> float | None:
    """1, якщо серед цитат є хоча б один очікуваний документ."""
    if not expected:
        return None
    return float(any(d in expected for d in cited_doc_ids))


def source_precision(cited_doc_ids: list[str], expected: list[str]) -> float | None:
    if not expected or not cited_doc_ids:
        return None
    return sum(d in expected for d in cited_doc_ids) / len(cited_doc_ids)


def abstain_correct(not_found: bool, answerable: bool) -> float:
    return float(not_found != answerable)


def forbidden_ok(answer: str, must_not_contain: list[str]) -> float | None:
    if not must_not_contain:
        return None
    return float(not any(contains_alt(answer, x) for x in must_not_contain))


def citations_valid(claims: list[dict], invalid_sources: list[str]) -> float | None:
    """Кожне твердження має ≥1 посилання, і немає посилань на неіснуючі джерела."""
    if not claims:
        return None
    return float(all(c["sources"] for c in claims) and not invalid_sources)


# ---------------- LLM-as-a-judge ----------------
JUDGE_SCHEMA = {"type": "object",
                "properties": {"reason": {"type": "string"}, "verdict": {"type": "string", "enum": ["pass", "fail"]}},
                "required": ["reason", "verdict"]}
FAITH_SCHEMA = {"type": "object", "properties": {"claims": {"type": "array", "items": {
    "type": "object", "properties": {"claim": {"type": "string"}, "reason": {"type": "string"},
                                     "supported": {"type": "boolean"}},
    "required": ["claim", "reason", "supported"]}}}, "required": ["claims"]}


def judge_correctness(llm, prompt, question: str, expected: str, answer_text: str) -> dict:
    system, user = prompt.render(question=question, expected=expected, answer=answer_text)
    r = llm.generate(system, user, JUDGE_SCHEMA, temperature=0.0)
    v = r.data.get("verdict")
    return {"verdict": v if v in ("pass", "fail") else "fail", "reason": r.data.get("reason", ""),
            "tokens": r.input_tokens + r.output_tokens, "in": r.input_tokens, "out": r.output_tokens}


def judge_faithfulness(llm, prompt, claims: list[str], context: str) -> dict:
    if not claims:
        return {"score": None, "details": []}
    system, user = prompt.render(claims="\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1)), context=context)
    r = llm.generate(system, user, FAITH_SCHEMA, temperature=0.0)
    det = r.data.get("claims", [])
    sup = [bool(d.get("supported")) for d in det][: len(claims)]
    sup += [False] * (len(claims) - len(sup))          # суддя пропустив твердження -> вважаємо непідтвердженим
    return {"score": sum(sup) / len(sup), "details": det, "in": r.input_tokens, "out": r.output_tokens}
