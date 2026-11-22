### Якість retrieval (22 питань з відомим правильним чанком)

| Chunking | Режим | Фільтри | Recall@1 | Recall@3 | Recall@5 | MRR@10 | Скасовані док. у top-1 |
|---|---|---|---|---|---|---|---|
| structure | bm25 | status+access | 0.77 | 0.86 | 0.91 | 0.830 | 0 |
| structure | vector | status+access | 0.41 | 0.68 | 0.91 | 0.599 | 0 |
| structure | hybrid | status+access | 0.64 | 0.91 | 0.91 | 0.779 | 0 |
| structure | hybrid_rerank | status+access | 0.86 | 0.95 | 0.95 | 0.909 | 0 |
| structure | hybrid_rerank | без фільтрів | 0.82 | 0.95 | 0.95 | 0.879 | 2 |
| fixed | bm25 | status+access | 0.68 | 0.91 | 0.91 | 0.773 | 0 |
| fixed | vector | status+access | 0.41 | 0.77 | 0.91 | 0.613 | 0 |
| fixed | hybrid | status+access | 0.73 | 0.91 | 0.91 | 0.818 | 0 |
| fixed | hybrid_rerank | status+access | 0.64 | 0.91 | 0.91 | 0.778 | 0 |

_Embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (fastembed); reranker: `jinaai/jina-reranker-v2-base-multilingual`._
