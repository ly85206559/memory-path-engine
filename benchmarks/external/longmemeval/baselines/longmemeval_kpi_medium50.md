# LongMemEval KPI (medium50)

- generated_at: `2026-10-02T02:45:44+00:00`
- source: `longmemeval_s_cleaned.json`
- note: Product M7 hybrid seed rerank

| Mode | R@5 | R@10 | NDCG@10 | avg_ms | questions |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical_baseline | 0.980 | 1.000 | 0.948 | 72.671 | 50 |
| hybrid | 0.980 | 1.000 | 0.948 | 702.588 | 50 |
| embedding_baseline | 0.680 | 0.760 | 0.603 | 387.552 | 50 |
| weighted_graph | 0.900 | 0.920 | 0.734 | 462.727 | 50 |
| activation_spreading_v1 | 0.920 | 0.920 | 0.864 | 436.039 | 50 |

## Notes

- Public recall KPI after M7 hybrid seed-score rerank.
- hybrid NDCG@10 matches lexical on this slice.
- Do not treat as Layer B architecture proof.
