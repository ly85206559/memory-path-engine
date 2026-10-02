# LongMemEval KPI (medium50_turn)

- generated_at: `2026-10-02T11:03:55+00:00`
- source: `longmemeval_s_cleaned.json`
- granularity: `turn`
- embedding: `ngram`
- note: Product M11 turn mid-slice

| Mode | R@5 | R@10 | NDCG@10 | avg_ms | questions |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical_baseline | 0.840 | 0.920 | 0.714 | 57.754 | 50 |
| embedding_baseline | 0.620 | 0.680 | 0.479 | 404.704 | 50 |
| weighted_graph | 0.780 | 0.820 | 0.670 | 114.396 | 50 |
| hybrid | 0.840 | 0.920 | 0.714 | 552.518 | 50 |
| activation_spreading_v1 | 0.760 | 0.800 | 0.631 | 108.098 | 50 |

## Notes

- Product M11 turn-granularity mid-slice (50q) public recall KPI.
- Turn units are finer than session; R@5 is lower than session medium50 (0.98) as expected.
- hybrid matches lexical_baseline on this ngram turn slice.
- Do not treat as Layer B architecture proof.
