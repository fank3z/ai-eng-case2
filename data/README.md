# Дані
- `raw/` — сирий корпус (34 файли, 4 формати). Синтетичний, згенерований `tools/build_corpus.py` з `tools/corpus_src/`.
- `processed/manifest.json` — маніфест: метадані, хеші, дублікати, статистика чанків.
- `processed/chunks.jsonl` — основні чанки (structure); `chunks_fixed.jsonl` — baseline (fixed).
Перегенерувати: `make ingest`.
