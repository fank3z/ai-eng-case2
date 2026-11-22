"""Витягування тексту зі збереженням структури -> уніфікований markdown
(# заголовки, - списки, | таблиці |) + метадані документа."""
from __future__ import annotations

import contextlib
import io
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RawDoc:
    path: Path
    fmt: str
    text: str                     # структурований markdown
    meta: dict = field(default_factory=dict)


def _md_table(rows: list[list[str]]) -> str:
    rows = [[" ".join(c.split()) for c in r] for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ""
    head, *body = rows
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


# ---------------- Markdown ----------------
def load_md(path: Path) -> RawDoc:
    raw = path.read_text(encoding="utf-8")
    meta = {}
    if raw.startswith("---\n"):
        _, fm, raw = raw.split("---\n", 2)
        meta = yaml.safe_load(fm) or {}
    return RawDoc(path, "md", raw, {k: str(v) for k, v in meta.items()})


# ---------------- HTML ----------------
BOILERPLATE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "noscript", "form", "button"]
BOILERPLATE_CLASSES = re.compile(r"cookie|banner|sidebar|breadcrumb|menu|share", re.I)


def load_html(path: Path) -> RawDoc:
    from bs4 import BeautifulSoup, UnicodeDammit
    raw = path.read_bytes()
    dammit = UnicodeDammit(raw, ["utf-8", "windows-1251"])  # визначення кодування
    soup = BeautifulSoup(dammit.unicode_markup, "html.parser")
    meta = {m["name"]: m.get("content", "") for m in soup.find_all("meta") if m.get("name")}
    meta["encoding_detected"] = dammit.original_encoding or "utf-8"
    h1 = soup.find("h1")
    if "title" not in meta:
        meta["title"] = h1.get_text(" ", strip=True) if h1 else (
            soup.title.get_text().rsplit(" — ", 1)[0].strip() if soup.title else path.stem)
    for tag in soup(BOILERPLATE_TAGS):
        tag.decompose()
    for tag in soup.find_all(class_=BOILERPLATE_CLASSES):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    out = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "ul", "ol", "table"], recursive=True):
        if el.find_parent(["table", "ul", "ol"]) and el.name != "table":
            continue
        if el.name.startswith("h"):
            out.append("#" * int(el.name[1]) + " " + el.get_text(" ", strip=True))
        elif el.name == "p":
            out.append(el.get_text(" ", strip=True))
        elif el.name in ("ul", "ol"):
            items = [li.get_text(" ", strip=True) for li in el.find_all("li", recursive=False)]
            out.append("\n".join((f"{i}. " if el.name == "ol" else "- ") + x for i, x in enumerate(items, 1)))
        elif el.name == "table":
            rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])] for tr in el.find_all("tr")]
            out.append(_md_table(rows))
    return RawDoc(path, "html", "\n\n".join(out), meta)


# ---------------- DOCX ----------------
def load_docx(path: Path) -> RawDoc:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    doc = Document(path)
    cp = doc.core_properties
    meta = {}
    try:
        meta = json.loads(cp.comments or "{}")
    except json.JSONDecodeError:
        pass
    meta.setdefault("title", cp.title or path.stem)
    meta.setdefault("version", cp.version or "")
    out, counters = [], Counter()
    # обхід тіла в порядку документа: абзаци й таблиці; колонтитули (section.header/footer) не беремо
    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            p = Paragraph(child, doc)
            txt, style = p.text.strip(), (p.style.name if p.style is not None else "")
            if not txt:
                continue
            if style == "Title":
                counters.clear(); out.append("# " + txt)
            elif style.startswith("Heading"):
                counters.clear()
                lvl = int(re.sub(r"\D", "", style) or 1) + 1
                out.append("#" * lvl + " " + txt)
            elif "List Number" in style:
                counters[id(out)] += 1
                out.append(f"{counters[id(out)]}. {txt}")
            elif "List" in style:
                out.append("- " + txt)
            else:
                counters.clear()
                out.append(txt)
        elif tag == "tbl":
            t = Table(child, doc)
            out.append(_md_table([[c.text for c in r.cells] for r in t.rows]))
    # склеюємо підряд ідучі пункти списку в один блок
    text = "\n\n".join(out)
    text = re.sub(r"\n\n(?=(- |\d+\. ))", "\n", text)
    return RawDoc(path, "docx", text, meta)


# ---------------- PDF ----------------
def load_pdf(path: Path) -> RawDoc:
    import fitz
    doc = fitz.open(path)
    meta = {}
    try:
        meta = json.loads(doc.metadata.get("keywords") or "{}")
    except json.JSONDecodeError:
        pass
    meta.setdefault("title", doc.metadata.get("title") or path.stem)
    n_pages = len(doc)
    items = []   # (page, y, kind, payload)
    for pno, page in enumerate(doc):
        h = page.rect.height
        with contextlib.redirect_stdout(io.StringIO()):   # глушимо рекламне повідомлення PyMuPDF
            tables = page.find_tables().tables
        tboxes = [fitz.Rect(tb.bbox) for tb in tables]
        for tb in tables:
            items.append((pno, tb.bbox[1], "table", _md_table([[c or "" for c in r] for r in tb.extract()])))
        for b in page.get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                txt = "".join(s["text"] for s in ln["spans"]).strip()
                if not txt:
                    continue
                r = fitz.Rect(ln["bbox"])
                if any(tb.contains(r) or tb.intersects(r) and (tb & r).get_area() > 0.5 * r.get_area() for tb in tboxes):
                    continue  # рядок усередині таблиці — таблицю вже взято цілком
                size = max(s["size"] for s in ln["spans"])
                zone = "top" if r.y1 < 0.07 * h else "bottom" if r.y0 > 0.93 * h else \
                       "edge" if (r.y1 < 0.15 * h or r.y0 > 0.85 * h) else "body"
                items.append((pno, r.y0, "line", {"text": txt, "size": size, "zone": zone, "x": r.x0}))
    # маркери списків («•», «1») — окремі спани з іншим розміром шрифту: приклеюємо до рядка поруч
    bullets = [it for it in items if it[2] == "line" and re.fullmatch(r"•|\d+\.?", it[3]["text"])]
    for b in bullets:
        near = [it for it in items if it[2] == "line" and it is not b and it[0] == b[0]
                and abs(it[1] - b[1]) < 6 and it[3]["x"] > b[3]["x"]]
        if near:
            tgt = min(near, key=lambda it: abs(it[1] - b[1]))
            mark = "- " if b[3]["text"] == "•" else b[3]["text"].rstrip(".") + ". "
            tgt[3]["text"] = mark + tgt[3]["text"]
            tgt[3]["list"] = True
            items.remove(b)
    items.sort(key=lambda x: (x[0], x[1]))
    # колонтитули: верхня/нижня зона сторінки + рядки біля краю, що повторюються на >=50% сторінок
    norm = lambda s: re.sub(r"\d+", "#", s)
    pages_with = {}
    for pno, _, kind, d in items:
        if kind == "line" and d["zone"] != "body":
            pages_with.setdefault(norm(d["text"]), set()).add(pno)
    def is_furniture(d):
        if d["zone"] in ("top", "bottom"):
            return True
        return d["zone"] == "edge" and n_pages > 1 and len(pages_with.get(norm(d["text"]), ())) >= max(2, n_pages / 2)
    items = [it for it in items if it[2] == "table" or not is_furniture(it[3])]
    # розмір основного тексту = найчастіший розмір, зважений кількістю символів
    weights = Counter()
    for _, _, k, d in items:
        if k == "line":
            weights[round(d["size"], 1)] += len(d["text"])
    body_size = weights.most_common(1)[0][0] if weights else 10
    out, para = [], []
    def flush():
        if para:
            out.append(" ".join(para)); para.clear()
    for _, _, kind, d in items:
        if kind == "table":
            flush(); out.append(d); continue
        t = d["text"]
        lvl = 0 if d.get("list") else 1 if d["size"] >= body_size + 5 else 2 if d["size"] >= body_size + 2 else 0
        if lvl:
            flush()
            if out and out[-1].startswith("#" * lvl + " ") and not out[-1].startswith("#" * (lvl + 1)):
                out[-1] += " " + t          # заголовок, перенесений на 2 рядки
            else:
                out.append("#" * lvl + " " + t)
        elif d.get("list"):
            flush(); out.append(t)
        else:
            if para and para[-1].endswith("-") and not para[-1].endswith(" -") and len(para[-1]) > 1:
                para[-1] = para[-1][:-1] + t       # перенос слова: «сло-» + «во»
            else:
                para.append(t)
    flush()
    text = re.sub(r"\n\n(?=(- |\d+\. ))", "\n", "\n\n".join(out))
    return RawDoc(path, "pdf", text, meta)


LOADERS = {".md": load_md, ".markdown": load_md, ".html": load_html, ".htm": load_html,
           ".docx": load_docx, ".pdf": load_pdf}


def load(path: Path) -> RawDoc | None:
    fn = LOADERS.get(path.suffix.lower())
    return fn(path) if fn else None
