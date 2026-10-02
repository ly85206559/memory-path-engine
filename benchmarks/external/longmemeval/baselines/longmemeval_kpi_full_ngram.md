# LongMemEval KPI (full, embedding=ngram)

- generated_at: `2026-10-02T03:11:48+00:00`
- source: `longmemeval_s_cleaned.json`
- samples: **500** (full LongMemEval-S cleaned)
- embedding: `ngram` (default)
- note: Product M7 hybrid seed rerank

| Mode | R@5 | R@10 | NDCG@10 | avg_ms | questions |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical_baseline | 0.950 | 0.974 | 0.863 | 74.233 | 500 |
| embedding_baseline | 0.740 | 0.836 | 0.579 | 388.495 | 500 |
| weighted_graph | 0.882 | 0.928 | 0.684 | 465.414 | 500 |
| hybrid | 0.950 | 0.974 | 0.863 | 718.036 | 500 |
| activation_spreading_v1 | 0.892 | 0.922 | 0.784 | 440.620 | 500 |

## Notes

- Product M7 headline public KPI for Layer A.
- hybrid matches lexical: R@5=0.950 / R@10=0.974 / NDCG@10=0.863 (was NDCG 0.767 pre-M7).
- Reproduce: `mpe bench longmemeval --dataset .../longmemeval_s_cleaned.json --label full --limit 0`
- Do not treat as Layer B architecture proof.
