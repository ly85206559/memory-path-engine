# Evaluation Plan

The first version of `memory-path-engine` is only useful if it can be evaluated against explicit baselines.

## Dataset format

The initial example benchmark pack lives in [`examples/contract_pack`](../examples/contract_pack).

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

### `baseline_topk`

- retrieve by lexical similarity only
- no graph expansion
- no weight-aware reranking

### `embedding_topk`

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

The suite output also includes a cross-mode comparison report so you can quickly spot:

- questions missed only by one mode
- modes that win on the same question
- latency trade-offs between lexical, embedding, structure-only, and weighted retrieval
