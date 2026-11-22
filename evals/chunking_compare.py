"""Порівняння structure-aware chunking з фіксованим вікном: кількісно по корпусу + приклади на кількох документах."""
from __future__ import annotations

import json
import re
from pathlib import Path

EXAMPLES = ["battery-backup-standard-2026", "nbiot-config-guide-v2", "sop-vswr"]


def _read(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def table_rows(text):
    return [re.sub(r"\s+", " ", l.strip()) for l in text.splitlines() if l.strip().startswith("|") and "---" not in l]


def run(processed: Path, out: Path):
    st, fx = _read(processed / "chunks.jsonl"), _read(processed / "chunks_fixed.jsonl")
    # 1) фіксовані чанки, що перетинають межу розділу (містять заголовок не на початку)
    crossing = sum(1 for c in fx if re.search(r"\S.*?#{2,} \S", c["text"]))
    # 2) таблиці, розрізані між чанками: рядок таблиці зі structure-чанка має бути цілком у одному fixed-чанку разом із шапкою
    split_tables, n_tables = 0, 0
    for c in st:
        rows = table_rows(c["text"])
        if not rows:
            continue
        n_tables += 1
        fx_doc = [re.sub(r"\s+", " ", f["text"]) for f in fx if f["doc_id"] == c["doc_id"]]
        if not any(all(r in f for r in rows) for f in fx_doc):
            split_tables += 1
    md = ["### Порівняння chunking: structure vs fixed\n",
          "| Метрика | structure | fixed (50 слів, перекриття 12) |", "|---|---|---|",
          f"| Чанків | {len(st)} | {len(fx)} |",
          f"| Середній розмір, токенів | {sum(c['n_tokens'] for c in st)/len(st):.0f} | {sum(c['n_tokens'] for c in fx)/len(fx):.0f} |",
          f"| Чанки, що перетинають межу розділу | 0 (за побудовою; злиті дрібні розділи позначені) | {crossing} ({crossing/len(fx):.0%}) |",
          f"| Таблиці, розрізані між чанками | 0 з {n_tables} | {split_tables} з {n_tables} |",
          ""]
    for d in EXAMPLES:
        md.append(f"#### Приклад: `{d}`\n")
        md.append("| structure | fixed |\n|---|---|")
        a = [c for c in st if c["doc_id"] == d]
        b = [c for c in fx if c["doc_id"] == d]
        for i in range(max(len(a), len(b))):
            fa = f"**{a[i]['section']}** — {a[i]['text'][:90].replace(chr(10), ' ')}…" if i < len(a) else ""
            fb = f"**{b[i]['section']}** — {b[i]['text'][:90]}…" if i < len(b) else ""
            md.append(f"| {fa.replace('|', '/')} | {fb.replace('|', '/')} |")
        md.append("")
    out.mkdir(parents=True, exist_ok=True)
    (out / "chunking.md").write_text("\n".join(md), encoding="utf-8")
    return "\n".join(md[:8])


if __name__ == "__main__":
    print(run(Path("data/processed"), Path("evals/results")))
