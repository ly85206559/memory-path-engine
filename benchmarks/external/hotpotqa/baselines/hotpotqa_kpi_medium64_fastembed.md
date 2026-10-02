# HotpotQA KPI (medium64_fastembed)

- generated_at: `2026-10-02T08:40:31+00:00`
- source: `hotpot_dev_distractor_v1.json`
- embedding: `fastembed`
- samples: `64`
- metric_scope: `external_positioning`

| Mode | evidence_hit | evidence_recall | avg_ms | questions |
| --- | ---: | ---: | ---: | ---: |
| lexical_baseline | 0.406 | 0.406 | 2.006 | 64 |
| hybrid | 0.859 | 0.859 | 340.117 | 64 |
| embedding_baseline | 0.531 | 0.531 | 0.570 | 64 |

## By type

### lexical_baseline

| Type | evidence_hit | questions |
| --- | ---: | ---: |
| bridge | 0.354 | 48 |
| comparison | 0.562 | 16 |

### hybrid

| Type | evidence_hit | questions |
| --- | ---: | ---: |
| bridge | 0.833 | 48 |
| comparison | 0.938 | 16 |

### embedding_baseline

| Type | evidence_hit | questions |
| --- | ---: | ---: |
| bridge | 0.438 | 48 |
| comparison | 0.812 | 16 |

## Notes

- Product M10 dense comparison on the same 64q mid-slice.
- fastembed embedding_baseline beats ngram especially on comparison (paraphrase-like) questions.
- hybrid+fastembed edges hybrid+ngram overall (0.859 vs 0.844).
- Do not treat as Layer B architecture proof.
