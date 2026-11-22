"""RAG на базі знань NOC + evals.

  python main.py all                 # ingest -> index -> eval-retrieval -> eval -> report  (= make run)
  python main.py ingest | index | eval-retrieval | eval | report
  python main.py ask "питання" [--role management] [--prompt v1]
  python main.py search "питання" [--mode bm25|vector|hybrid|hybrid_rerank]
  python main.py calibrate export|score
Прапорці: --offline (без моделей і ключа: tfidf + mock LLM), --ci (лише основний промпт + пороги).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

log = logging.getLogger("main")


def load_cfg(path: str, offline: bool) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if offline:
        cfg = copy.deepcopy(cfg)
        cfg["embeddings"] = {"backend": "tfidf"}
        cfg["rerank"] = {"backend": "overlap"}
        cfg["paths"]["index"] = "index_offline"
        cfg["paths"]["results"] = "evals/results_offline"
    return cfg


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def get_searcher(cfg, chunking="structure"):
    from src.retrieval.search import Searcher, load_chunks
    fn = "chunks.jsonl" if chunking == "structure" else f"chunks_{chunking}.jsonl"
    return Searcher(load_chunks(Path(cfg["paths"]["processed"]) / fn), cfg, Path(cfg["paths"]["index"]) / chunking)


def get_llms(cfg, offline):
    from src.rag.llm import GeminiLLM, build_llm
    if not offline:
        ok, why = GeminiLLM.available()
        if not ok:
            return None, None, why
    return build_llm(cfg["llm"]["answer"], offline), build_llm(cfg["llm"]["judge"], offline), ""


def cmd_ingest(cfg, a):
    from evals import chunking_compare
    from src.ingest import pipeline
    pipeline.run(cfg)
    log.info("\n" + chunking_compare.run(Path(cfg["paths"]["processed"]), Path(cfg["paths"]["results"])))


def cmd_index(cfg, a):
    from src.retrieval.search import Searcher, load_chunks
    for ch, fn in [("structure", "chunks.jsonl"), ("fixed", "chunks_fixed.jsonl")]:
        s = Searcher(load_chunks(Path(cfg["paths"]["processed"]) / fn), cfg, Path(cfg["paths"]["index"]) / ch, rebuild=True)
        log.info("Індекс %s: %d векторів, backend=%s", ch, len(s.store.payloads), s.store.backend)


def cmd_search(cfg, a):
    s = get_searcher(cfg)
    for h in s.search(a.text, k=5, mode=a.mode, filters=cfg["search"]["default_filters"]):
        print(f"{h.score:.3f}  {h.chunk['chunk_id']:40s} {h.chunk['section']}  {h.ranks}")


def cmd_ask(cfg, a):
    from src.prompts import load_prompt
    from src.rag.answer import answer
    llm, _, why = get_llms(cfg, a.offline)
    if llm is None:
        sys.exit(why)
    flt = dict(cfg["search"]["default_filters"])
    if a.role == "management":
        flt["access"] = ["noc", "field", "management"]
    r = answer(a.text, get_searcher(cfg), llm, load_prompt("rag", a.prompt or cfg["rag"]["prompt"]), cfg, flt)
    print(r.render())
    print(f"\n[контекст: {r.context_tokens} ток., чанки {r.context}, відкинуто бюджетом: {r.dropped_by_budget}, guard: {r.guard}]")


def cmd_eval_retrieval(cfg, a):
    from evals import retrieval_eval
    res = retrieval_eval.run(cfg, read_jsonl("evals/testset.jsonl"), Path(cfg["paths"]["results"]))
    main_row = [r for r in res["rows"] if r["chunking"] == "structure" and r["filters"] == "status+access"][-1]
    thr = cfg["eval"]["thresholds"].get("retrieval_recall@5")
    if a.ci and thr is not None and main_row["recall@5"] < thr:
        print(f"::error::retrieval recall@5 = {main_row['recall@5']:.2f} < {thr}")
        return 1
    return 0


def cmd_eval(cfg, a):
    from evals import rag_eval
    llm, judge, why = get_llms(cfg, a.offline)
    if llm is None:
        print(f"::warning::LLM-evals пропущено: {why}")
        return 0 if a.ci and a.allow_no_llm else 1
    out = Path(cfg["paths"]["results"])
    s = get_searcher(cfg)
    versions = [cfg["rag"]["prompt"]] if a.ci else cfg["eval"]["prompts"]
    summaries = [rag_eval.run_one(cfg, read_jsonl("evals/testset.jsonl"), s, llm, judge, v, out) for v in versions]
    summaries.sort(key=lambda x: x["prompt_version"])
    rag_eval.write_md(summaries, out)
    main_s = next(x for x in summaries if x["prompt_version"] == cfg["rag"]["prompt"])
    thr = {k: v for k, v in cfg["eval"]["thresholds"].items() if not k.startswith("retrieval")}
    fails = rag_eval.check_thresholds(main_s, thr)
    for f in fails:
        print(f"::error::{f}")
    if a.offline:
        return 0  # mock-метрики не є результатом
    return 1 if fails else 0


def cmd_calibrate(cfg, a):
    from evals import calibrate
    res = Path(cfg["paths"]["results"]) / f"rag_{cfg['rag']['prompt']}.jsonl"
    if a.text == "export":
        calibrate.export(res)
    else:
        calibrate.score(res, Path(cfg["paths"]["results"]))


def cmd_report(cfg, a):
    from evals import report
    if getattr(a, "offline", False):
        log.info("offline: REPORT.md не оновлюється (mock-метрики не є результатом)")
        return 0
    report.fill(Path("REPORT.md"), Path(cfg["paths"]["results"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "ingest", "index", "search", "ask", "eval-retrieval", "eval", "calibrate", "report"])
    ap.add_argument("text", nargs="?", default="")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--ci", action="store_true")
    ap.add_argument("--allow-no-llm", action="store_true")
    ap.add_argument("--mode", default="hybrid_rerank")
    ap.add_argument("--role", default="noc")
    ap.add_argument("--prompt")
    a = ap.parse_args()
    load_dotenv()
    cfg = load_cfg(a.config, a.offline)
    Path(cfg["paths"]["results"]).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(Path(cfg["paths"]["results"]) / "run.log", encoding="utf-8")])
    for noisy in ("httpx", "httpcore", "huggingface_hub", "fastembed"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    steps = {"ingest": cmd_ingest, "index": cmd_index, "search": cmd_search, "ask": cmd_ask,
             "eval-retrieval": cmd_eval_retrieval, "eval": cmd_eval, "calibrate": cmd_calibrate, "report": cmd_report}
    if a.cmd == "all":
        rc = 0
        for name in ["ingest", "index", "eval-retrieval", "eval", "report"]:
            log.info("===== %s =====", name)
            rc |= steps[name](cfg, a) or 0
        if not a.offline:
            cmd_calibrate(cfg, argparse.Namespace(text="score"))
            cmd_report(cfg, a)
        sys.exit(rc)
    sys.exit(steps[a.cmd](cfg, a) or 0)


if __name__ == "__main__":
    main()
