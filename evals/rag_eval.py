"""Прогін RAG на тестсеті + scoring + пороги для CI."""
from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

from src.prompts import load_prompt
from src.rag.answer import NOT_FOUND_TEXT, answer, format_sources
from src.retrieval.search import Hit
from . import scoring as S

log = logging.getLogger("eval.rag")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _pct(xs, p):
    if not xs:
        return None
    s = sorted(xs); k = (len(s) - 1) * p / 100; lo = int(k); hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_one(cfg, testset, searcher, llm, judge_llm, prompt_version, out_dir: Path) -> dict:
    ev = cfg["eval"]
    prompt = load_prompt("rag", prompt_version)
    jp = load_prompt("judge_correctness", ev["judge_correctness_prompt"])
    fp = load_prompt("judge_faithfulness", ev["judge_faithfulness_prompt"])
    recs = []
    for it in testset:
        r = answer(it["question"], searcher, llm, prompt, cfg, filters=it.get("filters"))
        text = NOT_FOUND_TEXT if r.not_found else r.answer
        cited = [c["doc_id"] for c in r.citations]
        rec = {"id": it["id"], "type": it["type"], "question": it["question"], "answerable": it["answerable"],
               "expected_answer": it["expected_answer"], "answer": text, "not_found": r.not_found,
               "claims": r.claims, "citations": r.citations, "retrieved": r.retrieved, "context": r.context,
               "dropped_by_budget": r.dropped_by_budget, "context_tokens": r.context_tokens, "guard": r.guard,
               "invalid_sources": r.invalid_sources, "error": r.error,
               "prompt_version": prompt_version, "model": llm.model, "judge_model": judge_llm.model,
               "in_tokens": r.input_tokens, "out_tokens": r.output_tokens, "latency_s": round(r.latency_s, 3),
               "cached": r.cached}
        # детерміновані
        rec["s_abstain"] = S.abstain_correct(r.not_found, it["answerable"])
        rec["s_key_facts"] = S.key_fact_recall(text, it["key_facts"]) if it["answerable"] else None
        rec["s_source_hit"] = S.source_hit(cited, it["expected_sources"]) if it["answerable"] else None
        rec["s_source_precision"] = S.source_precision(cited, it["expected_sources"]) if it["answerable"] else None
        rec["s_forbidden_ok"] = S.forbidden_ok(text, it.get("must_not_contain", []))
        rec["s_citations_valid"] = S.citations_valid(r.claims, r.invalid_sources)
        # LLM-as-a-judge
        try:
            j = S.judge_correctness(judge_llm, jp, it["question"], it["expected_answer"], text)
            rec["judge_verdict"], rec["judge_reason"] = j["verdict"], j["reason"]
            rec["judge_tokens"] = j["in"] + j["out"]
            if not r.not_found and r.claims:
                from src.rag.answer import format_sources
                from src.retrieval.search import Hit
                ctx_text = format_sources([Hit(next(c for c in searcher.chunks if c['chunk_id'] == cid), 0, {}) for cid in r.context])
                f = S.judge_faithfulness(judge_llm, fp, [c["text"] for c in r.claims], ctx_text)
                rec["s_faithfulness"], rec["faithfulness_details"] = f["score"], f["details"]
                rec["judge_tokens"] += f.get("in", 0) + f.get("out", 0)
            else:
                rec["s_faithfulness"] = None
        except Exception as e:
            rec["judge_verdict"], rec["judge_error"], rec["s_faithfulness"] = None, f"{type(e).__name__}: {e}", None
            log.error("judge %s: %s", it["id"], e)
        recs.append(rec)
        log.info("%s %-22s nf=%s judge=%s kf=%s", it["id"], it["type"], r.not_found, rec.get("judge_verdict"),
                 rec["s_key_facts"])

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"rag_{prompt_version}.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ans = [r for r in recs if r["answerable"]]
    una = [r for r in recs if not r["answerable"]]
    lat = [r["latency_s"] for r in recs if r["latency_s"] > 0]
    pr = cfg.get("pricing_usd_per_1m", {}).get(llm.model, {"input": 0, "output": 0})
    cost = sum(r["in_tokens"] * pr["input"] + r["out_tokens"] * pr["output"] for r in recs) / 1e6
    summ = {
        "prompt_version": prompt_version, "model": llm.model, "judge_model": judge_llm.model,
        "n": len(recs), "n_answerable": len(ans), "n_unanswerable": len(una),
        "abstain_accuracy": _mean([r["s_abstain"] for r in recs]),
        "idk_on_unanswerable": _mean([r["s_abstain"] for r in una]),
        "false_refusal_rate": _mean([1 - r["s_abstain"] for r in ans]),
        "key_fact_recall": _mean([r["s_key_facts"] for r in ans]),
        "source_hit": _mean([r["s_source_hit"] for r in ans]),
        "source_precision": _mean([r["s_source_precision"] for r in ans]),
        "forbidden_ok": _mean([r["s_forbidden_ok"] for r in recs]),
        "citations_valid": _mean([r["s_citations_valid"] for r in recs]),
        "judge_pass": _mean([1.0 if r.get("judge_verdict") == "pass" else 0.0 for r in recs if r.get("judge_verdict")]),
        "faithfulness": _mean([r["s_faithfulness"] for r in recs]),
        "latency_p50_s": _pct(lat, 50), "latency_p95_s": _pct(lat, 95),
        "avg_context_tokens": _mean([r["context_tokens"] for r in recs if r["context_tokens"]]),
        "avg_in_tokens": _mean([r["in_tokens"] for r in recs if r["in_tokens"]]),
        "cost_per_1000_questions_usd": 1000 * cost / len(recs),
        "errors": sum(1 for r in recs if r["error"] or r.get("judge_error")),
    }
    return summ


def check_thresholds(summary: dict, thresholds: dict) -> list[str]:
    fails = []
    for k, thr in thresholds.items():
        v = summary.get(k)
        if v is None or v < thr:
            fails.append(f"{k} = {v if v is None else round(v, 3)} < {thr}")
    return fails


def write_md(summaries: list[dict], out_dir: Path):
    keys = [("abstain_accuracy", "«Не знаю» де треба / відповідь де є"), ("idk_on_unanswerable", "«Не знаю» на питаннях без відповіді"),
            ("false_refusal_rate", "Хибні відмови ↓"), ("key_fact_recall", "Ключові факти"), ("source_hit", "Правильне джерело в цитатах"),
            ("source_precision", "Точність цитат"), ("forbidden_ok", "Без застарілих значень"), ("citations_valid", "Валідні цитати"),
            ("judge_pass", "LLM-judge: коректність"), ("faithfulness", "Faithfulness"),
            ("latency_p50_s", "Latency p50, с"), ("latency_p95_s", "Latency p95, с"), ("avg_context_tokens", "Контекст, токенів (сер.)"),
            ("cost_per_1000_questions_usd", "$ / 1000 питань")]
    md = ["### Якість відповідей RAG\n", "| Метрика | " + " | ".join(f"prompt {s['prompt_version']}" for s in summaries) + " |",
          "|---|" + "---|" * len(summaries)]
    for k, label in keys:
        vals = []
        for s in summaries:
            v = s.get(k)
            vals.append("—" if v is None else (f"{v:.4f}" if k.startswith("cost") else f"{v:.2f}"))
        md.append(f"| {label} | " + " | ".join(vals) + " |")
    s0 = summaries[0]
    md.append(f"\n_Питань: {s0['n']} (з відповіддю: {s0['n_answerable']}, без відповіді: {s0['n_unanswerable']}). "
              f"Модель відповідей: `{s0['model']}`, суддя: `{s0['judge_model']}`. Вартість — лише виклики моделі відповідей, "
              f"за цінами з config.yaml (на free tier фактично $0)._")
    (out_dir / "rag.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "rag_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("\n" + "\n".join(md))
