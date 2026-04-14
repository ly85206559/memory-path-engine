# Vision

`Memory Path Engine` is not positioned as "another RAG implementation." The long-term goal is to build a structured memory substrate for AI systems that can organize, activate, and replay knowledge rather than only retrieve similar text spans.

This document explains the design intent behind the repository. See [`architecture.md`](architecture.md) for the current implementation architecture.

## Why this project exists

Most retrieval systems still rely on a flat pattern:

1. split text into chunks
2. embed chunks
3. retrieve top-k matches
4. let the model improvise the rest

That works for many narrow tasks, but it breaks down when the target answer depends on:

- hierarchical structure
- exceptions and dependencies
- conflict resolution
- multi-hop reasoning
- evidence-backed explanation

This project explores a different premise:

> memory quality depends more on organization and activation than on storage alone.

## From memory palace to AI memory

The inspiration comes from the human "memory palace" technique, but not in a superficial visual-branding sense. The point is not to make AI "remember with pictures." The point is to engineer the same functional properties that make memory palaces effective.

| Memory palace property | Functional meaning | AI design translation |
|---|---|---|
| spatial structure | place information in stable locations | hierarchical and graph memory structure |
| vivid encoding | make items easy to distinguish | embeddings, tags, typed attributes, symbolic signals |
| strong association | use salience and surprise to bind memory | weighting, anomaly detection, conflict signals |
| sequential recall | walk through a route instead of random lookup | multi-hop retrieval and path replay |
| chunking | remember grouped units, not atomic noise | clause-level, obligation-level, and risk-level memory units |

The takeaway is simple:

> AI memory should not be treated as a flat similarity lookup problem.

## Four design laws

Everything in this repository should be judged against four rules.

### 1. Memory must have structure

Memory cannot be only a vector index. It needs explicit organization, such as:

- section trees
- typed nodes
- relationship edges
- document and topic boundaries

### 2. Memory must have weight

Not all memory items matter equally. Retrieval should be able to prefer:

- high-risk items
- high-importance items
- unusual or exceptional items
- frequently reinforced items

### 3. Memory must support associative jumps

Relevant evidence often lives next to the answer rather than inside the answer. Retrieval must support:

- dependency traversal
- exception traversal
- cause and effect links
- conflict and contradiction links

### 4. Memory must support path replay

The system should not only return a result. It should be able to show:

- where the query landed
- which nodes were activated
- which edges were traversed
- which evidence supported the final answer

## Strategic direction

This project has a staged roadmap.

### Stage 1: Memory-Augmented RAG

Short-term goal:

- introduce structure into retrieval
- add weighting and anomaly signals
- support path-oriented retrieval

This is the "weak memory palace" phase: still retrieval-driven, but no longer a flat chunk lookup.

### Stage 2: Graph Memory System

Mid-term goal:

- move from chunk-centric storage to node and edge memory
- treat relationships as first-class retrieval signals
- support controlled activation across a memory graph

This is where the system stops looking like classic RAG and starts looking like a memory graph.

### Stage 3: Brain-like memory mechanisms

Long-term goal:

- dynamic reinforcement and decay
- multiple memory representations for the same source
- query to memory-path to reasoning to answer

At this stage, the main innovation is not "better retrieval." It is an architecture for memory evolution and activation.

## Why the current example packs span different document types

The long-term core should stay domain-agnostic, so the repository now keeps multiple example packs. Contract-like data is useful because it contains:

- natural hierarchy
- dense dependency chains
- explicit exceptions and remedies
- critical risk-bearing clauses
- strong demand for explainability

Runbook-like data is also useful because it contains:

- ordered action sequences
- escalation paths
- operational branching logic
- process-heavy evidence chains

If structure, weighting, and path replay do not help across these example document benchmarks, they are unlikely to become broadly useful elsewhere.

## What "not plain RAG" means here

Plain RAG usually means:

- chunk retrieval
- top-k context assembly
- answer synthesis

This project is moving toward:

- typed nodes instead of anonymous chunks
- explicit edges instead of implicit similarity only
- weighted activation instead of uniform recall
- replayable paths instead of opaque answer generation

## Near-term development priorities

The next iterations should focus on:

1. stronger evaluation and ablation tooling
2. explicit anomaly and contradiction modeling
3. domain-pack abstractions that keep the core general
4. better semantic backends behind stable interfaces
5. **Memory Palace v1**: a first-class palace graph (spaces, episodic / semantic / route memories, lifecycle state) mapped onto the legacy `MemoryStore` for backward-compatible retrieval and benchmarks

## Relationship to the rest of the docs

- [`architecture.md`](architecture.md): current implementation structure and runtime components
- [`hypotheses.md`](hypotheses.md): measurable research claims
- [`evaluation.md`](evaluation.md): baseline modes and evaluation outputs
