# HotpotQA KPI (medium64)

- generated_at: `2026-10-02T08:40:31+00:00`
- source: `hotpot_dev_distractor_v1.json`
- embedding: `ngram`
- samples: `64`
- metric_scope: `external_positioning`

| Mode | evidence_hit | evidence_recall | avg_ms | questions |
| --- | ---: | ---: | ---: | ---: |
| lexical_baseline | 0.406 | 0.406 | 2.026 | 64 |
| embedding_baseline | 0.234 | 0.234 | 6.031 | 64 |
| weighted_graph | 0.609 | 0.609 | 4.478 | 64 |
| hybrid | 0.844 | 0.844 | 17.628 | 64 |
| activation_spreading_v1 | 0.594 | 0.594 | 4.193 | 64 |

## By type

### lexical_baseline

| Type | evidence_hit | questions |
| --- | ---: | ---: |
| bridge | 0.354 | 48 |
| comparison | 0.562 | 16 |

### embedding_baseline

| Type | evidence_hit | questions |
| --- | ---: | ---: |
| bridge | 0.208 | 48 |
| comparison | 0.312 | 16 |

### weighted_graph

| Type | evidence_hit | questions |
| --- | ---: | ---: |
| bridge | 0.604 | 48 |
| comparison | 0.625 | 16 |

### hybrid

| Type | evidence_hit | questions |
| --- | ---: | ---: |
| bridge | 0.812 | 48 |
| comparison | 0.938 | 16 |

### activation_spreading_v1

| Type | evidence_hit | questions |
| --- | ---: | ---: |
| bridge | 0.583 | 48 |
| comparison | 0.625 | 16 |

## Notes

- Product M10 mid-slice (64q) public HotpotQA evidence-hit KPI.
- hybrid (ngram) leads overall; retrieval-only (not official EM/F1).
- Do not treat as Layer B architecture proof.
