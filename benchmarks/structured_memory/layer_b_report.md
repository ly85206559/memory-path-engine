# Layer B Report

Fixtures: spatial_recall_benchmark.json, route_replay_benchmark.json, state_transition_benchmark.json, encoding_recall_benchmark.json, consolidation_gain_benchmark.json, exception_override_benchmark.json, exception_override_path_benchmark.json, multi_hop_chain_benchmark.json, activation_snapshot_benchmark.json
Modes: weighted_graph, activation_spreading_v1

## Overall

| Mode | path_hit_rate | path_cases | route_hit_rate | route_cases | space_hit_rate | space_cases | lifecycle_hit_rate | lifecycle_cases | activation_trace_hit_rate | trace_cases | activation_snapshot_hit_rate | snapshot_cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weighted_graph | 1.000 | 2 | 1.000 | 2 | 1.000 | 1 | 1.000 | 1 | 1.000 | 1 | 1.000 | 1 |
| activation_spreading_v1 | 1.000 | 2 | 1.000 | 2 | 1.000 | 1 | 1.000 | 1 | 1.000 | 1 | 1.000 | 1 |

## Per Fixture

### spatial_recall_benchmark.json

| Mode | path_hit_rate | path_cases | route_hit_rate | route_cases | space_hit_rate | space_cases | lifecycle_hit_rate | lifecycle_cases | activation_trace_hit_rate | trace_cases | activation_snapshot_hit_rate | snapshot_cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weighted_graph | 0.000 | 0 | 0.000 | 0 | 1.000 | 1 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 |
| activation_spreading_v1 | 0.000 | 0 | 0.000 | 0 | 1.000 | 1 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 |

### route_replay_benchmark.json

| Mode | path_hit_rate | path_cases | route_hit_rate | route_cases | space_hit_rate | space_cases | lifecycle_hit_rate | lifecycle_cases | activation_trace_hit_rate | trace_cases | activation_snapshot_hit_rate | snapshot_cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weighted_graph | 0.000 | 0 | 1.000 | 1 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 |
| activation_spreading_v1 | 0.000 | 0 | 1.000 | 1 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 |

### state_transition_benchmark.json

| Mode | path_hit_rate | path_cases | route_hit_rate | route_cases | space_hit_rate | space_cases | lifecycle_hit_rate | lifecycle_cases | activation_trace_hit_rate | trace_cases | activation_snapshot_hit_rate | snapshot_cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weighted_graph | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 1.000 | 1 | 0.000 | 0 | 0.000 | 0 |
| activation_spreading_v1 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 1.000 | 1 | 0.000 | 0 | 0.000 | 0 |

### encoding_recall_benchmark.json

| Mode | path_hit_rate | path_cases | route_hit_rate | route_cases | space_hit_rate | space_cases | lifecycle_hit_rate | lifecycle_cases | activation_trace_hit_rate | trace_cases | activation_snapshot_hit_rate | snapshot_cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weighted_graph | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 |
| activation_spreading_v1 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 |

### consolidation_gain_benchmark.json

| Mode | path_hit_rate | path_cases | route_hit_rate | route_cases | space_hit_rate | space_cases | lifecycle_hit_rate | lifecycle_cases | activation_trace_hit_rate | trace_cases | activation_snapshot_hit_rate | snapshot_cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weighted_graph | 0.000 | 0 | 1.000 | 1 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 |
| activation_spreading_v1 | 0.000 | 0 | 1.000 | 1 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 |

### exception_override_benchmark.json

| Mode | path_hit_rate | path_cases | route_hit_rate | route_cases | space_hit_rate | space_cases | lifecycle_hit_rate | lifecycle_cases | activation_trace_hit_rate | trace_cases | activation_snapshot_hit_rate | snapshot_cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weighted_graph | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 1.000 | 1 | 0.000 | 0 |
| activation_spreading_v1 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 1.000 | 1 | 0.000 | 0 |

### exception_override_path_benchmark.json

| Mode | path_hit_rate | path_cases | route_hit_rate | route_cases | space_hit_rate | space_cases | lifecycle_hit_rate | lifecycle_cases | activation_trace_hit_rate | trace_cases | activation_snapshot_hit_rate | snapshot_cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weighted_graph | 1.000 | 1 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 |
| activation_spreading_v1 | 1.000 | 1 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 |

### multi_hop_chain_benchmark.json

| Mode | path_hit_rate | path_cases | route_hit_rate | route_cases | space_hit_rate | space_cases | lifecycle_hit_rate | lifecycle_cases | activation_trace_hit_rate | trace_cases | activation_snapshot_hit_rate | snapshot_cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weighted_graph | 1.000 | 1 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 |
| activation_spreading_v1 | 1.000 | 1 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 |

### activation_snapshot_benchmark.json

| Mode | path_hit_rate | path_cases | route_hit_rate | route_cases | space_hit_rate | space_cases | lifecycle_hit_rate | lifecycle_cases | activation_trace_hit_rate | trace_cases | activation_snapshot_hit_rate | snapshot_cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weighted_graph | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 1.000 | 1 |
| activation_spreading_v1 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0 | 1.000 | 1 |
