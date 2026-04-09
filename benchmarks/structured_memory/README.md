# Structured Memory Benchmarks

This directory contains repository-owned benchmark fixtures for `memory-path-engine`.

These benchmarks are different from broad conversational-memory benchmarks such as LongMemEval or LoCoMo. Their purpose is to measure the architectural properties that matter for this project:

- evidence recall on structured documents
- multi-hop retrieval quality
- path-oriented reasoning support
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

## Current fixtures

- `example_contract_benchmark.json`
- `example_runbook_benchmark.json`

These fixtures are intentionally small. They are meant to support TDD and architectural iteration before larger benchmark suites are introduced.
