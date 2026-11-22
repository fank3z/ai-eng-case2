"""Якість пошуку: recall@k і MRR@10 на питаннях з відомим правильним чанком.
Правильний чанк = чанк з очікуваного документа, що містить evidence-фразу (не залежить від chunking)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.retrieval.search import Searcher, load_chunks
from src.textutil import norm_for_match

log = logging.getLogger("eval.retrieval")


def is_relevant(chunk: dict, item: dict) -> bool:
    return chunk["doc_id"] in item["expected_sources"] and \
        any(norm_for_match(e) in norm_for_match(chunk["text"]) for e in item["evidence"])


def evaluate(searcher: Searcher, items: list[dict], mode: str, filters_override="item", default_filters=None):
    ks, rr, per = (1, 3, 5), [], []
    hits_at = {k: 0 for k in ks}
    stale_top1 = 0
    for it in items:
        flt = it.get("filters", default_filters) if filters_override == "item" else filters_override
        res = searcher.search(it["question"], k=10, mode=mode, filters=flt)
        rank = next((r for r, h in enumerate(res, 1) if is_relevant(h.chunk, it)), None)
        for k in ks:
            hits_at[k] += bool(rank and rank <= k)
        rr.append(1 / rank if rank else 0.0)
        if res and res[0].chunk["status"] != "current" and it["type"] != "trap_version_explicit":
            stale_top1 += 1
        per.append({"id": it["id"], "rank": rank, "top3": [h.chunk["chunk_id"] for h in res[:3]]})
    n = len(items)
    m = {f"recall@{k}": hits_at[k] / n for k in ks}
    m["mrr@10"] = sum(rr) / n
    m["stale_top1"] = stale_top1
    return m, per


def run(cfg: dict, testset: list[dict], out_dir: Path) -> dict:
    items = [t for t in testset if t["answerable"] and t["evidence"]]
    default_filters = cfg["search"]["default_filters"]
    rows, details = [], {}
    proc, idx = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["index"])
    for chunking, fn in [("structure", "chunks.jsonl"), ("fixed", "chunks_fixed.jsonl")]:
        s = Searcher(load_chunks(proc / fn), cfg, idx / chunking)
        modes = ["bm25", "vector", "hybrid"] + (["hybrid_rerank"] if s.reranker else [])
        for mode in modes:
            m, per = evaluate(s, items, mode, "item", default_filters)
            rows.append({"chunking": chunking, "mode": mode, "filters": "status+access", **m})
            details[f"{chunking}/{mode}"] = per
        if chunking == "structure":   # ефект фільтра за метаданими на пастках зі старими версіями
            mode = modes[-1]
            m, per = evaluate(s, items, mode, {"access": ["noc", "field", "management"]})
            rows.append({"chunking": chunking, "mode": mode, "filters": "без фільтрів", **m})
            details[f"{chunking}/{mode}/nofilter"] = per
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"n_questions": len(items), "embeddings": cfg["embeddings"], "rerank": cfg.get("rerank"), "rows": rows,
           "details": details}
    (out_dir / "retrieval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    md = [f"### Якість retrieval ({len(items)} питань з відомим правильним чанком)\n",
          "| Chunking | Режим | Фільтри | Recall@1 | Recall@3 | Recall@5 | MRR@10 | Скасовані док. у top-1 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['chunking']} | {r['mode']} | {r['filters']} | {r['recall@1']:.2f} | {r['recall@3']:.2f} | "
                  f"{r['recall@5']:.2f} | {r['mrr@10']:.3f} | {r['stale_top1']} |")
    md.append(f"\n_Embeddings: `{cfg['embeddings'].get('model', cfg['embeddings']['backend'])}` ({cfg['embeddings']['backend']}); "
              f"reranker: `{(cfg.get('rerank') or {}).get('model', (cfg.get('rerank') or {}).get('backend'))}`._")
    (out_dir / "retrieval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("\n" + "\n".join(md))
    return res
