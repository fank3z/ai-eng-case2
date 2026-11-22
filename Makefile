PY ?= python

.PHONY: setup run ingest index eval-retrieval eval ask search label calibrate report smoke ci clean

setup:            ## встановити залежності
	$(PY) -m pip install -r requirements.txt

run:              ## повний прогін: ingest -> index -> retrieval eval -> RAG eval (v1, v2) -> калібрування -> REPORT.md
	$(PY) main.py all

ingest:
	$(PY) main.py ingest
index:
	$(PY) main.py index
eval-retrieval:
	$(PY) main.py eval-retrieval
eval:
	$(PY) main.py eval
report:
	$(PY) main.py report

ask:              ## make ask Q="Скільки триває зміна NOC?"
	$(PY) main.py ask "$(Q)"
search:
	$(PY) main.py search "$(Q)"

label:            ## сформувати evals/human_labels.jsonl для ручної розмітки
	$(PY) main.py calibrate export
calibrate:        ## порахувати згоду LLM-судді з ручною розміткою
	$(PY) main.py calibrate score && $(PY) main.py report

smoke:            ## перевірка всього пайплайну без моделей і ключа (метрики mock не є результатом)
	$(PY) main.py all --offline

ci:               ## те, що виконує GitHub Actions
	$(PY) main.py ingest
	$(PY) main.py index
	$(PY) main.py eval-retrieval --ci
	$(PY) main.py eval --ci --allow-no-llm

clean:
	rm -rf index index_offline .cache evals/results_offline data/processed/*.jsonl
