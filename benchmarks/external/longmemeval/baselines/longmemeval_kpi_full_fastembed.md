# LongMemEval KPI (full, embedding=fastembed)

- generated_at: `2026-10-02T03:31:35+00:00`
- source: `longmemeval_s_cleaned.json`
- samples: **500**
- embedding: `fastembed` (`BAAI/bge-small-en-v1.5`)
- compare_to: `longmemeval_kpi_full_ngram.md`
- note: Product M7 hybrid seed rerank

| Mode | R@5 | R@10 | NDCG@10 | avg_ms | questions |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical_baseline | 0.950 | 0.974 | 0.863 | 78.728 | 500 |
| hybrid | 0.962 | 0.980 | 0.875 | 2191.704 | 500 |
| embedding_baseline | 0.934 | 0.974 | 0.848 | 1.029 | 500 |

## Notes

- Product M7 optional dense full-corpus KPI after seed rerank.
- Do not treat as Layer B architecture proof.
