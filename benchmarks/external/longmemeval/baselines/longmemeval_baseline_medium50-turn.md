# LongMemEval baseline (medium50-turn)

- generated_at: `2026-10-02T11:03:55+00:00`
- dataset: `/workspace/benchmarks/external/longmemeval/data/longmemeval_s_cleaned.json`
- samples: **50**
- granularity: `turn`
- metric_scope: `external_positioning`
- product_kpi: `True`
- embedding: `ngram`

| Mode | R@5 | R@10 | NDCG@10 | avg_ms | questions |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical_baseline | 0.840 | 0.920 | 0.714 | 57.754 | 50 |
| embedding_baseline | 0.620 | 0.680 | 0.479 | 404.704 | 50 |
| weighted_graph | 0.780 | 0.820 | 0.670 | 114.396 | 50 |
| hybrid | 0.840 | 0.920 | 0.714 | 552.518 | 50 |
| activation_spreading_v1 | 0.760 | 0.800 | 0.631 | 108.098 | 50 |

## Notes

- Product KPI baseline for Layer A public recall.
- granularity=session aggregates each session; granularity=turn stores drawer-like turn units.
- hybrid mode blends lexical+embedding seeds then graph-expands.
- embedding=ngram (default) or fastembed/sentence via --embedding / MPE_EMBEDDING.
- Full LongMemEval-S: download the cleaned file and run with --label full.
- Layer B path/contradiction metrics remain the architecture proof surface.
