"""Вставляє таблиці з evals/results/*.md у REPORT.md між маркерами."""
from pathlib import Path


def fill(report: Path, results: Path):
    if not report.exists():
        return
    txt = report.read_text(encoding="utf-8")
    for name in ["chunking", "retrieval", "rag", "calibration"]:
        start, end = f"<!-- {name.upper()}:START -->", f"<!-- {name.upper()}:END -->"
        f = results / f"{name}.md"
        if start in txt and end in txt and f.exists():
            head, rest = txt.split(start, 1)
            txt = f"{head}{start}\n{f.read_text(encoding='utf-8')}{end}{rest.split(end, 1)[1]}"
    report.write_text(txt, encoding="utf-8")
