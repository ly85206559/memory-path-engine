# Ablation and Latency Report

Fixtures: structure_ablation_benchmark.json, multi_hop_chain_benchmark.json, exception_override_path_benchmark.json, contradiction_tension_benchmark.json
Modes: lexical_baseline, embedding_baseline, structure_only, weighted_graph, activation_spreading_v1

Ablation families follow `docs/evaluation.md`: remove structure, remove weights, remove path expansion.

## Overall Latency by Mode

| Mode | avg_ms | median_ms | p95_ms | max_ms | avg_evidence_hit_rate | fixtures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lexical_baseline | 0.167 | 0.128 | 0.258 | 0.258 | 1.000 | 4 |
| embedding_baseline | 0.282 | 0.136 | 0.666 | 0.666 | 1.000 | 4 |
| structure_only | 1.345 | 0.869 | 2.445 | 2.445 | 1.000 | 4 |
| weighted_graph | 1.753 | 1.174 | 3.143 | 3.143 | 1.000 | 4 |
| activation_spreading_v1 | 0.925 | 0.755 | 1.460 | 1.460 | 1.000 | 4 |

## Ablation Family Summary

| Family | baseline → full | primary metric | fixtures met | pass rate | avg primary Δ | avg latency Δ ms |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| no_structure | embedding_baseline → structure_only | semantic_hit_rate | 4/4 | 1.000 | +1.000 | +1.217 |
| no_weight | structure_only → weighted_graph | evidence_hit_rate | 4/4 | 1.000 | +0.000 | +0.461 |
| no_path_expansion | embedding_baseline → weighted_graph | path_hit_rate | 4/4 | 1.000 | +1.000 | +1.678 |

## Per Fixture

### structure_ablation_benchmark.json

| Mode | evidence_hit_rate | path_hit_rate | semantic_hit_rate | contradiction_hit_rate | avg_ms | median_ms | p95_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lexical_baseline | 1.000 | 0.000 | 0.000 | 0.000 | 0.258 | 0.258 | 0.258 |
| embedding_baseline | 1.000 | 0.000 | 0.000 | 0.000 | 0.666 | 0.666 | 0.666 |
| structure_only | 1.000 | 1.000 | 1.000 | 0.000 | 2.445 | 2.445 | 2.445 |
| weighted_graph | 1.000 | 1.000 | 1.000 | 0.000 | 3.143 | 3.143 | 3.143 |
| activation_spreading_v1 | 1.000 | 1.000 | 1.000 | 0.000 | 1.460 | 1.460 | 1.460 |

Ablation deltas:

| Family | primary Δ | evidence Δ | path Δ | semantic Δ | latency Δ ms | direction ok |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| no_structure | +1.000 | +0.000 | +1.000 | +1.000 | +1.779 | yes |
| no_weight | +0.000 | +0.000 | +0.000 | +0.000 | +0.698 | yes |
| no_path_expansion | +1.000 | +0.000 | +1.000 | +1.000 | +2.477 | yes |

### multi_hop_chain_benchmark.json

| Mode | evidence_hit_rate | path_hit_rate | semantic_hit_rate | contradiction_hit_rate | avg_ms | median_ms | p95_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lexical_baseline | 1.000 | 0.000 | 0.000 | 0.000 | 0.246 | 0.246 | 0.246 |
| embedding_baseline | 1.000 | 0.000 | 0.000 | 0.000 | 0.524 | 0.524 | 0.524 |
| structure_only | 1.000 | 1.000 | 1.000 | 0.000 | 2.366 | 2.366 | 2.366 |
| weighted_graph | 1.000 | 1.000 | 1.000 | 0.000 | 3.049 | 3.049 | 3.049 |
| activation_spreading_v1 | 1.000 | 1.000 | 1.000 | 0.000 | 1.340 | 1.340 | 1.340 |

Ablation deltas:

| Family | primary Δ | evidence Δ | path Δ | semantic Δ | latency Δ ms | direction ok |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| no_structure | +1.000 | +0.000 | +1.000 | +1.000 | +1.842 | yes |
| no_weight | +0.000 | +0.000 | +0.000 | +0.000 | +0.683 | yes |
| no_path_expansion | +1.000 | +0.000 | +1.000 | +1.000 | +2.525 | yes |

### exception_override_path_benchmark.json

| Mode | evidence_hit_rate | path_hit_rate | semantic_hit_rate | contradiction_hit_rate | avg_ms | median_ms | p95_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lexical_baseline | 1.000 | 0.000 | 0.000 | 1.000 | 0.113 | 0.113 | 0.113 |
| embedding_baseline | 1.000 | 0.000 | 0.000 | 1.000 | 0.155 | 0.155 | 0.155 |
| structure_only | 1.000 | 1.000 | 1.000 | 1.000 | 0.649 | 0.649 | 0.649 |
| weighted_graph | 1.000 | 1.000 | 1.000 | 1.000 | 0.807 | 0.807 | 0.807 |
| activation_spreading_v1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.488 | 0.488 | 0.488 |

Ablation deltas:

| Family | primary Δ | evidence Δ | path Δ | semantic Δ | latency Δ ms | direction ok |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| no_structure | +1.000 | +0.000 | +1.000 | +1.000 | +0.494 | yes |
| no_weight | +0.000 | +0.000 | +0.000 | +0.000 | +0.158 | yes |
| no_path_expansion | +1.000 | +0.000 | +1.000 | +1.000 | +0.652 | yes |

### contradiction_tension_benchmark.json

| Mode | evidence_hit_rate | path_hit_rate | semantic_hit_rate | contradiction_hit_rate | avg_ms | median_ms | p95_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lexical_baseline | 1.000 | 0.000 | 0.000 | 1.000 | 0.128 | 0.128 | 0.137 |
| embedding_baseline | 1.000 | 0.000 | 0.000 | 1.000 | 0.116 | 0.083 | 0.185 |
| structure_only | 1.000 | 1.000 | 1.000 | 1.000 | 0.869 | 0.852 | 0.939 |
| weighted_graph | 1.000 | 1.000 | 1.000 | 1.000 | 1.174 | 1.154 | 1.252 |
| activation_spreading_v1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.755 | 0.774 | 0.848 |

Ablation deltas:

| Family | primary Δ | evidence Δ | path Δ | semantic Δ | latency Δ ms | direction ok |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| no_structure | +1.000 | +0.000 | +1.000 | +1.000 | +0.753 | yes |
| no_weight | +0.000 | +0.000 | +0.000 | +0.000 | +0.305 | yes |
| no_path_expansion | +1.000 | +0.000 | +1.000 | +1.000 | +1.058 | yes |
