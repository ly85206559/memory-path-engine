# LongMemEval KPI (medium30)

- generated_at: `2026-10-01T15:18:59.454655+00:00`
- source: `longmemeval_s_cleaned.json`

| Mode | R@5 | R@10 | NDCG@10 | avg_ms | questions |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical_baseline | 1.000 | 1.000 | 0.988 | 87.310 | 30 |
| embedding_baseline | 0.700 | 0.800 | 0.641 | 485.932 | 30 |
| weighted_graph | 0.933 | 0.933 | 0.783 | 154.844 | 30 |
| hybrid | 1.000 | 1.000 | 0.932 | 723.459 | 30 |
| activation_spreading_v1 | 0.933 | 0.933 | 0.909 | 129.453 | 30 |

## Notes

- Public recall KPI after score-ordered ranking + hybrid BM25/ngram boost.
- Do not treat as Layer B architecture proof.
