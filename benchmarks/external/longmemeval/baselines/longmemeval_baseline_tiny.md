# LongMemEval baseline (tiny)

- generated_at: `2026-10-01T08:37:37+00:00`
- dataset: `/workspace/benchmarks/external/longmemeval/longmemeval_tiny_fixture.json`
- samples: **2**
- granularity: `session`
- metric_scope: `external_positioning`
- product_kpi: `True`

| Mode | R@5 | R@10 | NDCG@10 | avg_ms | questions |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical_baseline | 1.000 | 1.000 | 0.785 | 0.214 | 2 |
| embedding_baseline | 1.000 | 1.000 | 0.785 | 0.245 | 2 |
| weighted_graph | 1.000 | 1.000 | 0.847 | 1.129 | 2 |
| activation_spreading_v1 | 1.000 | 1.000 | 0.825 | 0.557 | 2 |

## Notes

- Product M1 baseline skeleton: session-level retrieval-only.
- Turn-level drawers and hybrid recall land in later milestones.
- Layer B path/contradiction metrics remain the architecture proof surface.
