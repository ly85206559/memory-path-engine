# Evaluation Plan

The first version of `memory-path-engine` is only useful if it can be evaluated against explicit baselines.

For the higher-level benchmark portfolio strategy across public datasets, repository-owned fixtures, and private real-world gold sets, see [`benchmark-strategy.md`](benchmark-strategy.md).

## Dataset format

The initial example benchmark pack lives in [`examples/contract_pack`](../examples/contract_pack).

Repository-owned benchmark fixtures now live in [`../benchmarks/structured_memory`](../benchmarks/structured_memory) and are modeled through strong pydantic types in the benchmark bounded context.

Inputs:

- synthetic markdown contract-like documents in `examples/contract_pack/contracts`
- annotated evaluation questions in `examples/contract_pack/eval/questions.json`

Each question includes:

- `id`
- `query`
- `answer`
- `evidence_node_ids`
- `tags`

## Metrics

### `answer_recall`

Whether the retriever returns a path whose final answer matches the expected answer pattern.

### `evidence_recall`

Whether at least one expected evidence node is present in the returned path set.

### `path_validity`

Whether the path crosses edges that are semantically allowed by the domain pack.

### `latency_ms`

Elapsed time for a single query under local execution.

## Baselines

### `lexical_baseline`

- retrieve by lexical similarity only
- no graph expansion
- no weight-aware reranking

### `embedding_baseline`

- retrieve by embedding similarity through a pluggable `EmbeddingProvider`
- no graph expansion
- no weight-aware reranking

### `structure_only`

- retrieve by embedding similarity
- expand across explicit edges
- no risk or anomaly boosts

### `weighted_graph`

- embedding-based candidate retrieval
- edge-aware expansion
- weighted scoring across semantic, structural, anomaly, and importance signals
- replayable `MemoryPath` output

### `activation_spreading_v1`

- embedding-based seed selection
- explicit activation propagation with decay and threshold controls
- edge-type-aware traversal
- semantic bonuses for exception, remedy, and escalation nodes
- replayable path output with propagation-oriented diagnostics

### Static vs dynamic memory experiment modes

The repository also exposes paired experiment modes that keep the retriever logic fixed while changing whether memory state is updated across queries:

- `weighted_graph_static`
- `weighted_graph_dynamic`
- `activation_spreading_static`
- `activation_spreading_dynamic`

Interpretation:

- `*_static` uses `StaticMemoryStatePolicy`, so query order should not change node memory state
- `*_dynamic` uses `MemoryStatePolicy`, so repeated queries can reinforce some nodes and decay others before later cases run

The primary repository-owned fixture for this comparison is `benchmarks/structured_memory/dynamic_memory_priming_benchmark.json`, which is intentionally ordered as repeated `prime-*` cases followed by a final `probe-*` case.

## Required ablations

- remove structure
- remove weights
- remove path expansion

If these ablations produce no meaningful change, the core design assumptions need to be revisited.

## What success looks like

- graph-aware retrieval wins on multi-hop structured-document questions in the example benchmark
- weighted retrieval improves critical clause discovery
- path output makes failures easy to inspect

## Detailed reports

The evaluation runner can now emit detailed per-question diagnostics in addition to summary scores.

With `detailed=True`, each mode includes:

- `avg_latency_ms`
- per-question hit or miss
- expected vs returned evidence node ids
- best answer text for inspection
- surfaced semantic roles
- best-path edge types
- activated node count and propagation depth

The suite output also includes a cross-mode comparison report so you can quickly spot:

- questions missed only by one mode
- modes that win on the same question
- latency trade-offs between lexical, embedding, structure-only, and weighted retrieval
- path-hit and semantic-hit rates for graph-oriented cases
- activation breadth and propagation depth for spreading-based retrieval

## Benchmark bounded context

The structured benchmark workflow is separated into its own bounded context:

- `benchmarking.domain`: typed dataset, case, expectation, and report models
- `benchmarking.infrastructure`: JSON fixture loading
- `benchmarking.application`: runner and end-to-end evaluation service

This keeps evaluation logic explicit and strongly typed instead of spreading anonymous `dict` payloads through the codebase.

Repository-owned graph fixtures now also cover:

- exception override cases
- multi-hop chain cases
- path-shape expectations
- semantic-role and edge-type expectations
- dynamic priming cases for static vs dynamic memory comparison
