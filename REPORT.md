# REPORT — Кейс 2: RAG на базі знань NOC + evals

## 1. Налаштування
- Корпус: 34 файли (MD/HTML/DOCX/PDF) → 32 унікальні документи; дублікати: 1 точний, 1 майже-дублікат.
- Chunking: structure-aware (розділ = чанк, 60–300 токенів, таблиці цілі) vs fixed (50 слів, перекриття 12).
- Пошук: BM25 + `paraphrase-multilingual-MiniLM-L12-v2` (FAISS) → RRF → `jina-reranker-v2-base-multilingual`; фільтри `status`, `access`.
- Відповіді: `gemini-3.5-flash-lite`, T=0, бюджет контексту 1200 токенів; суддя: `gemini-3.8-flash`.
- Тестсет: 26 питань (22 з відповіддю, 4 без).

## 2. Chunking
<!-- CHUNKING:START -->
_Заповнюється автоматично після `make run`._
<!-- CHUNKING:END -->

## 3. Retrieval
<!-- RETRIEVAL:START -->
_Заповнюється автоматично після `make run`._
<!-- RETRIEVAL:END -->

## 4. Відповіді
<!-- RAG:START -->
_Заповнюється автоматично після `make run`._
<!-- RAG:END -->

## 5. Калібрування LLM-судді
<!-- CALIBRATION:START -->
_Після `make label` → ручна розмітка ≥10 відповідей → `make calibrate`._
<!-- CALIBRATION:END -->

## 6. Розбір характерних помилок
> 3–5 прикладів з `evals/results/rag_v2.jsonl` і `retrieval.json`: що мало бути → що вийшло → чому → що змінити.

1. …
2. …
3. …

## 7. Висновки і що покращити далі
> Яка конфігурація пошуку найкраща і чому; чи окупився reranker; що дали фільтри; де система ламається; наступні кроки.
