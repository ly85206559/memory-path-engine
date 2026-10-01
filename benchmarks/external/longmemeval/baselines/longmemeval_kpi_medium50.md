# LongMemEval KPI (medium50)

- generated_at: `2026-10-01T15:18:59.455899+00:00`
- source: `longmemeval_s_cleaned.json`

| Mode | R@5 | R@10 | NDCG@10 | avg_ms | questions |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical_baseline | 0.980 | 1.000 | 0.948 | 76.399 | 50 |
| embedding_baseline | 0.680 | 0.760 | 0.603 | 441.715 | 50 |
| weighted_graph | 0.900 | 0.920 | 0.736 | 149.460 | 50 |
| hybrid | 0.960 | 1.000 | 0.871 | 625.392 | 50 |
| activation_spreading_v1 | 0.880 | 0.920 | 0.827 | 126.751 | 50 |

## Notes

- Public recall KPI after score-ordered ranking + hybrid BM25/ngram boost.
- Do not treat as Layer B architecture proof.
