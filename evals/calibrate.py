"""Калібрування LLM-судді на ручній розмітці.
1) export: формує evals/human_labels.jsonl з відповідями останнього прогону (вердикт судді НЕ показується).
2) Ви заповнюєте human_verdict: "pass" | "fail" (і за бажанням human_faithful: true/false) щонайменше для 10 відповідей.
3) score: згода судді з вами (accuracy, Cohen's kappa, матриця помилок, список розбіжностей)."""
from __future__ import annotations

import json
from pathlib import Path

LABELS = Path("evals/human_labels.jsonl")


def _read(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def export(results: Path):
    old = {r["id"]: r for r in _read(LABELS)}
    rows = []
    for r in _read(results):
        prev = old.get(r["id"], {})
        same = prev.get("answer") == r["answer"]
        rows.append({"id": r["id"], "question": r["question"], "expected_answer": r["expected_answer"],
                     "answer": r["answer"],
                     "human_verdict": prev.get("human_verdict") if same else None,
                     "human_faithful": prev.get("human_faithful") if same else None,
                     "note": prev.get("note", "") if same else ""})
    LABELS.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    n = sum(1 for x in rows if x["human_verdict"])
    print(f"{LABELS}: {len(rows)} відповідей, розмічено {n}. Заповніть human_verdict (pass/fail) для ≥10.")


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    labels = {"pass", "fail"}
    pe = sum((sum(a == l for a, _ in pairs) / n) * (sum(b == l for _, b in pairs) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score(results: Path, out_dir: Path) -> dict | None:
    human = {r["id"]: r for r in _read(LABELS) if r.get("human_verdict") in ("pass", "fail")}
    judged = {r["id"]: r for r in _read(results)}
    pairs, dis = [], []
    for i, h in human.items():
        j = judged.get(i)
        if not j or j.get("judge_verdict") not in ("pass", "fail") or j["answer"] != h["answer"]:
            continue
        pairs.append((h["human_verdict"], j["judge_verdict"]))
        if h["human_verdict"] != j["judge_verdict"]:
            dis.append({"id": i, "human": h["human_verdict"], "judge": j["judge_verdict"], "judge_reason": j.get("judge_reason"),
                        "note": h.get("note", "")})
    if len(pairs) < 10:
        print(f"Розмічено лише {len(pairs)} відповідей, що збігаються з останнім прогоном (потрібно ≥10).")
        return None
    cm = {f"human_{a}/judge_{b}": sum(1 for x, y in pairs if x == a and y == b) for a in ("pass", "fail") for b in ("pass", "fail")}
    res = {"n": len(pairs), "agreement": sum(a == b for a, b in pairs) / len(pairs), "cohen_kappa": kappa(pairs),
           "confusion": cm, "disagreements": dis}
    # faithfulness: людина vs суддя (твердження підтверджені повністю)
    fp = [(h["human_faithful"], (judged[i].get("s_faithfulness") or 0) >= 0.999) for i, h in human.items()
          if h.get("human_faithful") is not None and i in judged and judged[i].get("s_faithfulness") is not None]
    if fp:
        res["faithfulness_agreement"] = sum(a == b for a, b in fp) / len(fp)
        res["faithfulness_n"] = len(fp)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["### Калібрування LLM-судді\n",
          f"Розмічено вручну: **{res['n']}** відповідей. Згода судді з людиною: **{res['agreement']:.0%}**, "
          f"Cohen's κ = **{res['cohen_kappa']:.2f}**.\n",
          "| | суддя pass | суддя fail |", "|---|---|---|",
          f"| людина pass | {cm['human_pass/judge_pass']} | {cm['human_pass/judge_fail']} |",
          f"| людина fail | {cm['human_fail/judge_pass']} | {cm['human_fail/judge_fail']} |"]
    if "faithfulness_agreement" in res:
        md.append(f"\nFaithfulness: згода з ручною оцінкою {res['faithfulness_agreement']:.0%} (n={res['faithfulness_n']}).")
    if dis:
        md.append("\n**Розбіжності:**")
        md += [f"- {d['id']}: людина `{d['human']}`, суддя `{d['judge']}` — {d['judge_reason']}" for d in dis]
    (out_dir / "calibration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return res
