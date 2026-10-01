# LongMemEval baseline (tiny-turn)

- generated_at: `2026-10-01T15:18:42+00:00`
- dataset: `/workspace/benchmarks/external/longmemeval/longmemeval_tiny_fixture.json`
- samples: **2**
- granularity: `turn`
- metric_scope: `external_positioning`
- product_kpi: `True`

| Mode | R@5 | R@10 | NDCG@10 | avg_ms | questions |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical_baseline | 1.000 | 1.000 | 0.728 | 0.454 | 2 |
| embedding_baseline | 1.000 | 1.000 | 0.751 | 1.004 | 2 |
| weighted_graph | 1.000 | 1.000 | 0.741 | 2.406 | 2 |
| hybrid | 1.000 | 1.000 | 0.728 | 3.655 | 2 |
| activation_spreading_v1 | 1.000 | 1.000 | 0.741 | 0.954 | 2 |

## Notes

- Product KPI baseline for Layer A public recall.
- granularity=session aggregates each session; granularity=turn stores drawer-like turn units.
- hybrid mode blends lexical+embedding seeds then graph-expands.
- Full LongMemEval-S: download the cleaned file and run with --label full.
- Layer B path/contradiction metrics remain the architecture proof surface.
