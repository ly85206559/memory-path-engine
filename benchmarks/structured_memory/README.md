# Structured Memory Benchmarks

This directory contains repository-owned benchmark fixtures for `memory-path-engine`.

These benchmarks are different from broad conversational-memory benchmarks such as LongMemEval or LoCoMo. Their purpose is to measure the architectural properties that matter for this project:

- evidence recall on structured documents
- multi-hop retrieval quality
- path-oriented reasoning support
- semantic-role and edge-type correctness
- latency trade-offs across retrieval modes

## Why repository-owned fixtures exist

External benchmarks are useful for comparing generic memory retrieval quality, but they do not fully capture the design goals of this project.

`memory-path-engine` needs its own benchmark layer because it cares about:

- typed nodes
- explicit edges
- weighted retrieval
- replayable memory paths
- domain-pack-specific evidence expectations

## Dataset format

Each dataset file is validated through the `structured_benchmark` bounded context and uses strong pydantic models.

Top-level fields:

- `dataset_id`
- `dataset_name`
- `domain_pack_name`
- `document_directory`
- `cases`

Each case defines:

- `case_id`
- `query`
- `tags`
- `expectation`

Current expectation fields:

- `evidence_node_ids`
- `minimum_evidence_matches`
- `path`
- `path_scope`
- `required_edge_types`
- `required_semantic_roles`
- `required_contradiction_pairs`
- `activation_trace`
- `required_trace_stop_reasons`
- `min_activation_trace_length`
- `max_activation_trace_length`

Layer B reports should now also surface fixed aggregates for:

- `path_hit_rate`
- `route_hit_rate`
- `space_hit_rate`
- `lifecycle_hit_rate`
- `activation_snapshot_hit_rate`

## Current fixtures

- `example_contract_benchmark.json`
- `example_runbook_benchmark.json`
- `contract_exception_priming_benchmark.json`
- `dynamic_override_sequence_benchmark.json`
- `dynamic_memory_priming_benchmark.json`
- `exception_override_benchmark.json`
- `exception_override_path_benchmark.json`
- `multi_hop_chain_benchmark.json`
- `structure_ablation_benchmark.json`

These fixtures are intentionally small. They are meant to support TDD and architectural iteration before larger benchmark suites are introduced.

To generate a fixed-format Layer B report across the current palace-oriented fixtures:

```bash
python scripts/generate_layer_b_report.py --output "benchmarks/structured_memory/layer_b_report.json" --markdown-output "benchmarks/structured_memory/layer_b_report.md"
```

## Dynamic Memory Priming

`dynamic_memory_priming_benchmark.json` is the first repository-owned fixture designed to show a real sequential-memory effect across cases.

It uses repeated `prime-*` cases to reinforce one branch of a small runbook graph, then uses a final `probe-*` case to compare `activation_spreading_static` and `activation_spreading_dynamic` on the same graph.

How to read it:

- `activation_spreading_static` keeps memory state frozen across queries
- `activation_spreading_dynamic` reinforces visited path nodes and decays unvisited nodes between queries
- the `prime-*` cases should stay aligned between static and dynamic
- the final `probe-*` case is expected to diverge, showing that dynamic memory state now changes later retrieval outcomes

In the calibrated version of this fixture, the divergence is visible at more than one level:

- case-level: `evidence_hit`, `path_hit`, and `hit`
- summary-level: `evidence_hit_rate` and `path_hit_rate`

The source document for this benchmark lives in `examples/priming_pack/runbooks`.

`contract_exception_priming_benchmark.json` adds a second priming-oriented fixture on top of the contract pack. It keeps evidence recall stable on the final probe, but expects `activation_spreading_dynamic` to replay a more sequentially correct beta termination path than `activation_spreading_static` after repeated alpha exception priming.

`dynamic_override_sequence_benchmark.json` is a smaller contract-oriented dynamic sequence fixture that isolates a single probe path and expects `activation_spreading_dynamic` to resolve it more faithfully than `activation_spreading_static`.

`exception_override_path_benchmark.json` makes the exception override case path-aware: activation spreading should satisfy evidence, path, semantic-role, and contradiction expectations on the same sample.

`structure_ablation_benchmark.json` is a compact regression target for no-structure / no-weight / no-path-expansion comparisons. It is intended to show that graph-aware modes surface the right edge types and semantic roles more reliably than flat retrieval.
