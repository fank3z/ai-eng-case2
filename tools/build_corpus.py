"""Генерує «сирий» корпус data/raw/ у різних форматах (MD / HTML / DOCX / PDF)
з вихідних текстів tools/corpus_src/*.md.

Корпус синтетичний: база знань NOC вигаданого оператора NetCo. Значення параметрів
вигадані й не відповідають жодній реальній мережі. Генератор навмисно додає «бруд»,
який має прибрати ingestion: навігацію/cookie-банер/скрипти в HTML, колонтитули в
DOCX і PDF, кодування windows-1251, точний дублікат і майже-дублікат.

Запускати не обов'язково: згенеровані файли вже лежать у data/raw/.
"""
from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

import yaml

SRC = Path(__file__).parent / "corpus_src"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"
FONT = "/usr/share/fonts/liberation/LiberationSans-Regular.ttf"
FONT_B = "/usr/share/fonts/liberation/LiberationSans-Bold.ttf"


def parse(path: Path):
    raw = path.read_text(encoding="utf-8")
    _, fm, body = raw.split("---\n", 2)
    meta = yaml.safe_load(fm)
    blocks, lines = [], body.strip().splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1; continue
        if ln.startswith("#"):
            lvl = len(ln) - len(ln.lstrip("#"))
            blocks.append(("h", lvl, ln[lvl:].strip())); i += 1
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                if not all(re.fullmatch(r"-+", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
        elif re.match(r"^(- |\d+\. )", ln):
            ordered = bool(re.match(r"^\d+\. ", ln)); items = []
            while i < len(lines) and re.match(r"^(- |\d+\. )", lines[i]):
                items.append(re.sub(r"^(- |\d+\. )", "", lines[i])); i += 1
            blocks.append(("list", ordered, items))
        else:
            buf = []
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||- |\d+\. )", lines[i]):
                buf.append(lines[i]); i += 1
            blocks.append(("p", " ".join(buf)))
    return meta, blocks, raw


def plain(s):  # прибрати markdown-жирний для не-md форматів
    return s.replace("**", "")


def meta_json(meta):
    keys = ["doc_id", "version", "date", "status", "doc_type", "access", "superseded_by"]
    return json.dumps({k: str(meta[k]) for k in keys if k in meta}, ensure_ascii=False)


# ---------------- HTML ----------------
def to_html(meta, blocks, variant=0):
    out = []
    for b in blocks:
        if b[0] == "h":
            out.append(f"<h{b[1]}>{html.escape(b[2])}</h{b[1]}>")
        elif b[0] == "p":
            out.append(f"<p>{html.escape(plain(b[1]))}</p>")
        elif b[0] == "list":
            tag = "ol" if b[1] else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{html.escape(plain(x))}</li>" for x in b[2]) + f"</{tag}>")
        elif b[0] == "table":
            head, *rows = b[1]
            out.append("<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in head) + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows)
                       + "</tbody></table>")
    charset = meta.get("encoding", "utf-8")
    m = json.loads(meta_json(meta))
    metas = "\n".join(f'<meta name="{k}" content="{html.escape(v)}">' for k, v in m.items())
    menu = "Головна | Регламенти | Каталоги аварій | Контакти" if variant == 0 else "Головна | База знань | Пошук | Вихід"
    return f"""<!DOCTYPE html>
<html lang="uk"><head><meta charset="{charset}">
<title>{html.escape(meta['title'])} — Wiki NOC NetCo</title>
{metas}
<style>body{{font-family:sans-serif}}</style>
<script>window.analytics=window.analytics||[];analytics.push(['page']);</script>
</head><body>
<header class="site-header"><div class="logo">NetCo Wiki</div><nav class="top-nav">{menu}</nav></header>
<div class="cookie-banner">Ми використовуємо cookies для покращення роботи wiki. <button>Прийняти</button></div>
<aside class="sidebar"><ul><li>Останні зміни</li><li>Мої сторінки</li><li>Допомога</li></ul></aside>
<main><article>
{chr(10).join(out)}
</article></main>
<footer class="site-footer">© 2026 NetCo. Усі права захищено. Сторінку востаннє редаговано {meta['date']}.</footer>
<script>document.querySelector('.cookie-banner button').onclick=()=>{{}};</script>
</body></html>"""


# ---------------- DOCX ----------------
def to_docx(meta, blocks, path):
    from docx import Document
    doc = Document()
    sec = doc.sections[0]
    sec.header.paragraphs[0].text = "NetCo · Служба експлуатації мережі · Для внутрішнього використання"
    sec.footer.paragraphs[0].text = f"Документ {meta['doc_id']} · версія {meta['version']}"
    for b in blocks:
        if b[0] == "h":
            doc.add_heading(b[2], level=b[1] - 1 if b[1] > 1 else 0)
        elif b[0] == "p":
            doc.add_paragraph(plain(b[1]))
        elif b[0] == "list":
            for x in b[2]:
                doc.add_paragraph(plain(x), style="List Number" if b[1] else "List Bullet")
        elif b[0] == "table":
            rows = b[1]
            t = doc.add_table(rows=len(rows), cols=len(rows[0]))
            t.style = "Table Grid"
            for r, row in enumerate(rows):
                for c, val in enumerate(row):
                    t.cell(r, c).text = val
    cp = doc.core_properties
    cp.title, cp.version, cp.comments = meta["title"], str(meta["version"]), meta_json(meta)
    doc.save(path)


# ---------------- PDF ----------------
def to_pdf(meta, blocks, path):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (ListFlowable, ListItem, PageBreak, Paragraph,
                                    SimpleDocTemplate, Spacer, Table, TableStyle)
    if "LS" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("LS", FONT)); pdfmetrics.registerFont(TTFont("LSB", FONT_B))
    body = ParagraphStyle("b", fontName="LS", fontSize=10, leading=14)
    h1 = ParagraphStyle("h1", fontName="LSB", fontSize=16, leading=20, spaceAfter=8)
    h2 = ParagraphStyle("h2", fontName="LSB", fontSize=13, leading=17, spaceBefore=8, spaceAfter=4)
    story, n_h2 = [], 0
    for b in blocks:
        if b[0] == "h":
            if b[1] == 2:
                n_h2 += 1
                if n_h2 > 1 and n_h2 % 2 == 1:
                    story.append(PageBreak())  # багатосторінкові PDF -> колонтитули повторюються
            story.append(Paragraph(html.escape(b[2]), h1 if b[1] == 1 else h2))
        elif b[0] == "p":
            story.append(Paragraph(html.escape(plain(b[1])), body))
        elif b[0] == "list":
            story.append(ListFlowable([ListItem(Paragraph(html.escape(plain(x)), body)) for x in b[2]],
                                      bulletType="1" if b[1] else "bullet", bulletFontName="LS"))
        elif b[0] == "table":
            t = Table([[Paragraph(html.escape(c), body) for c in r] for r in b[1]])
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
            story.append(t)
        story.append(Spacer(1, 4))

    def deco(c, d):
        c.setFont("LS", 8)
        c.drawString(40, A4[1] - 30, "NetCo · Служба експлуатації мережі · Для внутрішнього використання")
        c.drawRightString(A4[0] - 40, 25, f"Сторінка {d.page}")
    SimpleDocTemplate(str(path), pagesize=A4, title=meta["title"], subject=str(meta["version"]),
                      keywords=meta_json(meta), author="NetCo NOC").build(story, onFirstPage=deco, onLaterPages=deco)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for src in sorted(SRC.glob("*.md")):
        meta, blocks, raw = parse(src)
        fmt, stem = meta["format"], meta["doc_id"]
        if fmt == "md":
            (OUT / f"{stem}.md").write_text(raw, encoding="utf-8")
        elif fmt == "html":
            enc = meta.get("encoding", "utf-8")
            (OUT / f"{stem}.html").write_bytes(to_html(meta, blocks).encode(enc))
        elif fmt == "docx":
            to_docx(meta, blocks, OUT / f"{stem}.docx")
        elif fmt == "pdf":
            to_pdf(meta, blocks, OUT / f"{stem}.pdf")
    # «бруд» для дедуплікації
    shutil.copy(OUT / "sop-vswr.md", OUT / "sop-vswr (копія).md")                 # точний дублікат
    meta, blocks, _ = parse(SRC / "faq-noc.md")
    meta = {**meta, "encoding": "utf-8"}
    blocks = [(b[0], b[1].replace("Рекомендовано", "Рекомендується")) if b[0] == "p" else b for b in blocks]
    (OUT / "faq-noc-archive.html").write_text(to_html(meta, blocks, variant=1), encoding="utf-8")  # майже-дублікат
    print(f"Згенеровано {len(list(OUT.iterdir()))} файлів у {OUT}")


if __name__ == "__main__":
    main()
