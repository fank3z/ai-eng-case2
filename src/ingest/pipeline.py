"""Ingestion: data/raw -> очищення -> дедуплікація -> chunking -> data/processed/{chunks*.jsonl, manifest.json}."""
from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path

from .chunking import chunk_fixed, chunk_structure, finalize
from .clean import clean_text, content_hash, deduplicate
from .loaders import LOADERS, load

log = logging.getLogger("ingest")


def run(cfg: dict) -> dict:
    raw_dir, out_dir = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ccfg = cfg["chunking"]
    docs, skipped = [], []
    for path in sorted(raw_dir.iterdir()):
        if path.suffix.lower() not in LOADERS:
            skipped.append(str(path)); continue
        try:
            rd = load(path)
        except Exception as e:  # битий файл не має валити весь ingestion
            log.error("Не вдалося прочитати %s: %s", path.name, e); skipped.append(str(path)); continue
        text = clean_text(rd.text)
        meta = {k: str(v) for k, v in rd.meta.items() if v not in (None, "")}
        meta.setdefault("doc_id", path.stem)
        docs.append({**meta, "source": str(path.as_posix()), "format": rd.fmt, "text": text,
                     "doc_hash": content_hash(text), "n_chars": len(text)})
    deduplicate(docs, ccfg.get("near_dup_threshold", 0.85))

    outputs = {"structure": [], "fixed": []}
    for d in docs:
        if d["duplicate_of"]:
            log.info("ДУБЛІКАТ %s -> %s (%s)", d["source"], d["duplicate_of"], d["dedup_reason"]); continue
        outputs["structure"] += finalize(d, chunk_structure(d["text"], ccfg["max_tokens"]), "structure")
        outputs["fixed"] += finalize(d, chunk_fixed(d["text"], ccfg["fixed_size"], ccfg["fixed_overlap"]), "fixed")
        d["n_chunks"] = sum(1 for c in outputs["structure"] if c["doc_id"] == d["doc_id"])

    for name, chunks in outputs.items():
        fn = out_dir / ("chunks.jsonl" if name == "structure" else f"chunks_{name}.jsonl")
        with fn.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    def stats(ch):
        t = [c["n_tokens"] for c in ch]
        return {"n_chunks": len(t), "tokens_mean": round(statistics.mean(t), 1), "tokens_median": statistics.median(t),
                "tokens_min": min(t), "tokens_max": max(t)}
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunking": ccfg,
        "n_files": len(docs) + len(skipped), "n_docs_unique": sum(1 for d in docs if not d["duplicate_of"]),
        "n_duplicates": sum(1 for d in docs if d["duplicate_of"]), "skipped": skipped,
        "formats": {f: sum(1 for d in docs if d["format"] == f) for f in sorted({d["format"] for d in docs})},
        "chunk_stats": {k: stats(v) for k, v in outputs.items()},
        "documents": [{k: d.get(k) for k in ["doc_id", "title", "source", "format", "version", "date", "status",
                                              "superseded_by", "doc_type", "access", "doc_hash", "n_chars",
                                              "n_chunks", "duplicate_of", "dedup_reason", "encoding_detected"]}
                      for d in docs],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Ingestion: %d файлів, %d унікальних, %d дублікатів; чанків structure=%d, fixed=%d",
             manifest["n_files"], manifest["n_docs_unique"], manifest["n_duplicates"],
             len(outputs["structure"]), len(outputs["fixed"]))
    return manifest
