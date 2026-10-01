# LongMemEval baseline (tiny)

- generated_at: `2026-10-01T08:51:44+00:00`
- dataset: `/workspace/benchmarks/external/longmemeval/longmemeval_tiny_fixture.json`
- samples: **2**
- granularity: `session`
- metric_scope: `external_positioning`
- product_kpi: `True`

| Mode | R@5 | R@10 | NDCG@10 | avg_ms | questions |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical_baseline | 1.000 | 1.000 | 0.785 | 0.222 | 2 |
| embedding_baseline | 1.000 | 1.000 | 0.785 | 0.276 | 2 |
| weighted_graph | 1.000 | 1.000 | 0.847 | 1.316 | 2 |
| hybrid | 1.000 | 1.000 | 0.847 | 1.183 | 2 |
| activation_spreading_v1 | 1.000 | 1.000 | 0.825 | 0.643 | 2 |

## Notes

- Product KPI baseline: session-level retrieval-only.
- hybrid mode blends lexical+embedding seeds then graph-expands.
- Turn-level drawers and full LongMemEval-S KPI land in M3.
- Layer B path/contradiction metrics remain the architecture proof surface.
