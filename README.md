# Memory Path Engine

A structured memory engine for AI agents with weighted retrieval and path replay.

`Memory Path Engine` is a research prototype for moving beyond plain `top-k` chunk retrieval. Instead of treating memory as a flat vector index, it represents memory as nodes, edges, weights, and replayable evidence paths.

The long-term goal is a general memory substrate that can support different domain packs. The first validation pack in this repository is `contract_pack`, because contracts provide strong structure, clear risk signals, and natural multi-hop reasoning tasks.

## Why this exists

Most RAG systems still look like this:

1. Split documents into chunks.
2. Embed chunks.
3. Return top-k matches.
4. Ask an LLM to improvise the reasoning.

This repo explores a different question:

> Can we retrieve, traverse, and replay a memory path instead of only returning similar chunks?

The prototype focuses on three ideas:

- `structure`: memory is not flat; it has typed nodes and edges
- `weight`: not every memory should be treated equally
- `path`: the system should expose how it moved from query to evidence

## Research hypotheses

The first milestone tests three hypotheses:

- `H1`: graph-aware retrieval beats vanilla top-k retrieval on multi-hop questions
- `H2`: anomaly and importance weighting improve recall of critical evidence
- `H3`: replayable memory paths improve explainability without unacceptable latency

See [`docs/hypotheses.md`](docs/hypotheses.md) for the success criteria.

## Repository layout

- [`src/memory_engine`](src/memory_engine): core schema, store, ingestion, retrieval, replay
- [`docs`](docs): vision, architecture, hypotheses, evaluation plan
- [`examples/contract_pack`](examples/contract_pack): first domain validation pack
- [`tests`](tests): initial unit tests for schema and retrieval behavior

## Read This First

- [`docs/vision.md`](docs/vision.md): why this project exists, how memory palace ideas map to AI memory, and where the architecture is heading
- [`docs/architecture.md`](docs/architecture.md): what the current implementation looks like
- [`docs/evaluation.md`](docs/evaluation.md): how retrieval modes are compared

## Quick start

Install the project in editable mode:

```bash
python -m pip install --no-build-isolation -e .
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

Try the contract-pack example:

```python
from pathlib import Path

from memory_engine.ingest import ingest_contract_markdown
from memory_engine.retrieve import WeightedGraphRetriever
from memory_engine.store import MemoryStore

store = MemoryStore()
contracts_dir = Path("examples/contract_pack/contracts")

for path in contracts_dir.glob("*.md"):
    ingest_contract_markdown(path, store, domain_pack="contract_pack")

retriever = WeightedGraphRetriever(store)
result = retriever.search(
    "What happens if delivery is late and the supplier also misses the cure period?",
    top_k=3,
)

print(result.best_path().final_answer)
for step in result.best_path().steps:
    print(step.node_id, step.reason, round(step.score, 3))
```

## What is in scope for v0

- minimal `MemoryNode`, `MemoryEdge`, `MemoryPath`, and `EvidenceRef` schema
- an in-memory store for fast iteration
- a simple ingestion path for contract-style markdown
- three retrieval modes:
  - a naive lexical top-k baseline
  - an embedding top-k baseline with a pluggable `EmbeddingProvider`
  - a weighted graph retriever with neighbor expansion, configurable scoring, and replayable paths
- a small synthetic contract evaluation set for end-to-end experiments

## What is explicitly out of scope for now

- production infrastructure
- MCP integration
- multi-modal memory encoding
- online reinforcement and forgetting policies
- large-scale benchmarks
- full UI

## Why contracts first

This repository does not assume memory is only useful for contract intelligence. The core is designed to stay domain-agnostic. Contracts are simply the first proving ground because they contain:

- hierarchical structure
- exception and dependency chains
- critical risk-bearing clauses
- strong need for evidence-backed reasoning

If the retrieval and replay ideas cannot survive this setting, they are unlikely to generalize.

## Experimental framework

The retrieval stack now separates:

- candidate generation
- semantic similarity backend
- scoring strategy
- path replay

This makes it possible to compare lexical baseline, embedding baseline, structure-only traversal, and weighted graph retrieval without rewriting the main search loop.

The evaluation layer can also emit detailed per-question reports, which makes miss analysis and ablation debugging much easier than relying on a single aggregate score.

## Planned next steps

- add explicit anomaly detectors and contradiction edges
- expand the evaluation runner with ablation reports and latency summaries
- extract a `domain_pack` interface for contracts, code, and research notes
- add stronger embedding backends behind the same `EmbeddingProvider` interface

## License

MIT. See [`LICENSE`](LICENSE).
