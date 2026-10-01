# LongMemEval KPI (medium50, embedding=fastembed)

- generated_at: `2026-10-01T16:04:26+00:00`
- source: `longmemeval_s_cleaned.json`
- embedding: `fastembed` (`BAAI/bge-small-en-v1.5`)
- compare_to: `longmemeval_kpi_medium50.md` (default `ngram`)

| Mode | R@5 | R@10 | NDCG@10 | avg_ms | questions |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical_baseline | 0.980 | 1.000 | 0.948 | 76.218 | 50 |
| embedding_baseline | 0.800 | 0.920 | 0.752 | 2344.392 | 50 |
| weighted_graph | 0.900 | 0.960 | 0.752 | 78.698 | 50 |
| hybrid | 0.980 | 1.000 | 0.872 | 208.331 | 50 |
| activation_spreading_v1 | 0.920 | 0.940 | 0.858 | 55.006 | 50 |

## Notes

- Product M5: optional dense backend via `--embedding fastembed` / `MPE_EMBEDDING=fastembed`.
- hybrid R@5 improves vs ngram medium50 (0.960 → 0.980); embedding_baseline 0.680 → 0.800.
- Default install remains ngram (no fastembed dependency). Do not treat as Layer B proof.
