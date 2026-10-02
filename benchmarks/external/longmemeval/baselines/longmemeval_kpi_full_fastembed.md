# LongMemEval KPI (full, embedding=fastembed)

- generated_at: `2026-10-02T00:13:09+00:00`
- source: `longmemeval_s_cleaned.json`
- samples: **500** (full LongMemEval-S cleaned)
- embedding: `fastembed` (`BAAI/bge-small-en-v1.5`)
- compare_to: `longmemeval_kpi_full_ngram.md`

| Mode | R@5 | R@10 | NDCG@10 | avg_ms | questions |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical_baseline | 0.950 | 0.974 | 0.863 | 77.069 | 500 |
| embedding_baseline | 0.934 | 0.974 | 0.848 | 1996.259 | 500 |
| hybrid | 0.952 | 0.982 | 0.775 | 212.303 | 500 |

## Notes

- Product M6 optional dense full-corpus KPI.
- vs ngram full: hybrid R@5 **0.938 → 0.952**; embedding_baseline **0.740 → 0.934**.
- Reproduce: `pip install 'memory-path-engine[embed]' && mpe bench longmemeval ... --embedding fastembed --limit 0`
- Do not treat as Layer B architecture proof.
