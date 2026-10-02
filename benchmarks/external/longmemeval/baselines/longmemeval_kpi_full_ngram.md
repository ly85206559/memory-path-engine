# LongMemEval KPI (full, embedding=ngram)

- generated_at: `2026-10-01T23:53:18+00:00`
- source: `longmemeval_s_cleaned.json`
- samples: **500** (full LongMemEval-S cleaned)
- embedding: `ngram` (default)

| Mode | R@5 | R@10 | NDCG@10 | avg_ms | questions |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical_baseline | 0.950 | 0.974 | 0.863 | 97.396 | 500 |
| embedding_baseline | 0.740 | 0.836 | 0.579 | 497.959 | 500 |
| weighted_graph | 0.882 | 0.928 | 0.684 | 501.004 | 500 |
| hybrid | 0.938 | 0.978 | 0.767 | 618.154 | 500 |
| activation_spreading_v1 | 0.892 | 0.922 | 0.784 | 446.976 | 500 |

## Notes

- Product M6 headline public KPI for Layer A.
- hybrid R@5=0.938 / R@10=0.978 on the full cleaned LongMemEval-S split.
- Reproduce: `mpe bench longmemeval --dataset .../longmemeval_s_cleaned.json --label full --limit 0`
- Do not treat as Layer B architecture proof.
