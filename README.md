# Memory Path Engine

[![CI](https://github.com/ly85206559/memory-path-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/ly85206559/memory-path-engine/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status: Productizing](https://img.shields.io/badge/status-productizing%20(M4)-0e7c86.svg)](docs/ROADMAP.md)

Local memory with **replayable evidence paths** — not only `top-k` chunks. Structured graph retrieval for agents, with a local palace CLI, MCP closed loop, and public-benchmark KPIs.

`Memory Path Engine` models memory as typed nodes, edges, weights, and `MemoryPath` objects so a system can retrieve, traverse, and explain how it reached an answer. Product M1 adds an on-disk palace (SQLite) and the `mpe` CLI; Layer B fixtures remain the architecture proof surface.

### System shape (v0 + Memory Palace v1 + Product M1)

Bundled markdown packs are ingested into a graph (`MemoryNode` / `MemoryEdge`). Retrievers return a `MemoryPath`. The CLI persists that graph under `.mpe/` and prints answer + hops.

**Memory Palace v1** adds a parallel domain (`memory_engine.memory`) mapped through `palace_to_store`. See [`docs/architecture.md`](docs/architecture.md) and [`docs/ROADMAP.md`](docs/ROADMAP.md).

```text
 examples/*_pack  ──▶  mpe ingest  ──▶  .mpe/store.sqlite (typed graph)
                                              │
                               ┌──────────────┼──────────────┐
                               ▼              ▼              ▼
                         BaselineTopK    other modes    WeightedGraph
                         (flat answers)  in `retrieve`  (path + scores)
                                              │
                                              ▼
                               mpe search/path → answer + hop list
```

## Why this project is different

Most RAG systems still look like this:

1. Split documents into chunks.
2. Embed chunks.
3. Return `top-k` matches.
4. Ask the LLM to improvise the reasoning.

This project asks:

> Can we retrieve a memory path instead of only retrieving similar chunks?

Three product bets:

- `structure`: typed nodes and edges, not only a flat vector index
- `weight`: importance, risk, novelty, reinforcement / forgetting
- `path`: replayable evidence chains as the default search output

## What you can do here

- compare multiple retrieval modes in one codebase
- inspect replayable evidence paths instead of only final answers
- test graph-aware retrieval on contract-like and operational documents
- run repository-owned structured benchmarks instead of toy snippets

## Quick start

Maintainers: configure the GitHub link-card image using [docs/social-preview.md](docs/social-preview.md) (`docs/assets/open-graph-cover.png`).

Install the project in editable mode (or see [`docs/install.md`](docs/install.md) for pipx / uv / Docker):

```bash
python -m pip install --no-build-isolation -e .
mpe doctor
```

### Product CLI (`mpe`) — local palace

```bash
mpe init --mode hybrid
mpe ingest examples/runbook_pack/runbooks --pack example_runbook_pack
mpe search "What if rollback does not recover the API?" --mode hybrid
mpe path "What if rollback does not recover the API?"
mpe status
mpe backup
```

Palace files live in `./.mpe/` (or `$MPE_PALACE`). Search always prints an answer plus hop citations.

5-minute agent closed loop (MCP + hooks): see [`docs/getting-started.md`](docs/getting-started.md).

```bash
mpe hooks install
mpe mcp   # stdio MCP server for Cursor / Claude
# docker build -t mpe-mcp . && docker run -i --rm -v "$PWD/.mpe:/data/palace" -e MPE_PALACE=/data/palace mpe-mcp
```

Acceptance checklist: [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md).  
Current progress vs vision stages: [`docs/progress.md`](docs/progress.md).

### LongMemEval product KPI baseline

```bash
mpe bench longmemeval --label tiny --granularity session
mpe bench longmemeval --label tiny --granularity turn
```

Checked-in tiny reports live under `benchmarks/external/longmemeval/baselines/`.

**Latest medium-slice public recall (session, LongMemEval-S cleaned):**

| Slice | Embedding | Mode | R@5 | R@10 | NDCG@10 |
| --- | --- | --- | ---: | ---: | ---: |
| 30q | ngram | hybrid | 1.000 | 1.000 | 0.932 |
| 30q | ngram | lexical_baseline | 1.000 | 1.000 | 0.988 |
| 50q | ngram | hybrid | 0.960 | 1.000 | 0.871 |
| 50q | ngram | lexical_baseline | 0.980 | 1.000 | 0.948 |
| 50q | fastembed | hybrid | 0.980 | 1.000 | 0.872 |
| 50q | fastembed | embedding_baseline | 0.800 | 0.920 | 0.752 |

Artifacts: `benchmarks/external/longmemeval/baselines/longmemeval_kpi_medium{30,50}*.{json,md}`.

**Optional dense embeddings (Product M5):** default remains dependency-free `ngram`. Install `pip install 'memory-path-engine[embed]'` then:

```bash
mpe bench longmemeval --label tiny --embedding fastembed
# or: export MPE_EMBEDDING=fastembed
```

**Reproduce full LongMemEval-S (public KPI):**

```bash
python scripts/download_longmemeval.py
mpe bench longmemeval \
  --dataset benchmarks/external/longmemeval/data/longmemeval_s_cleaned.json \
  --label full --granularity session --limit 0
```

| Slice | Granularity | How to run | Notes |
| --- | --- | --- | --- |
| tiny (2q, checked in) | session / turn | `mpe bench longmemeval --label tiny [--granularity turn]` | CI smoke + committed baselines |
| medium (50q) | session | nightly / local with `--limit 50` | positioning |
| full (LongMemEval-S) | session / turn | download + `--label full --limit 0` | headline public KPI |

See [`benchmarks/external/longmemeval/README.md`](benchmarks/external/longmemeval/README.md) and [`docs/ROADMAP.md`](docs/ROADMAP.md).

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

Run the runbook demo:

```bash
python -m memory_engine.demo --scenario runbook
```

Terminal-style capture of real stdout (refresh with `python scripts/generate_runbook_demo_terminal_svg.py`; `latency_ms` may differ run to run):

![Runbook demo terminal output](docs/assets/runbook-demo-terminal.svg)

Run the research-notes demo:

```bash
python -m memory_engine.demo --scenario research
```

Run the HotpotQA tiny benchmark sanity check:

```bash
python scripts/run_hotpotqa_benchmark.py
```

Run the LongMemEval tiny benchmark sanity check:

```bash
python scripts/run_longmemeval_benchmark.py
```

Print compact v1 palace metadata (spaces, routes, memory kinds) per case:

```bash
python scripts/run_longmemeval_benchmark.py --v1-recall-summary
```

Generate a fixed-format Layer B report (path/route/space/lifecycle/activation snapshot):

```bash
python scripts/generate_layer_b_report.py --output "benchmarks/structured_memory/layer_b_report.json" --markdown-output "benchmarks/structured_memory/layer_b_report.md"
```

Generate a fixed-format ablation matrix and latency summary (no-structure / no-weight / no-path-expansion):

```bash
python scripts/generate_ablation_report.py --output "benchmarks/structured_memory/ablation_report.json" --markdown-output "benchmarks/structured_memory/ablation_report.md"
```

Generate a Layer A positioning report (external metrics only; keeps architecture claims in Layer B):

```bash
python scripts/generate_layer_a_report.py --slice-profile tiny --markdown-output "benchmarks/external/layer_a_report.md"
```

Run Layer C transferable stand-in benchmarks:

```bash
python scripts/run_layer_c_benchmark.py --markdown-output "benchmarks/layer_c_minimal/layer_c_report.md"
```

Download the official HotpotQA dev distractor file for local benchmark runs:

```bash
python scripts/download_hotpotqa.py
```

Download the cleaned LongMemEval-S file for local benchmark runs:

```bash
python scripts/download_longmemeval.py
```

### What you will see

`python -m memory_engine.demo` prints a small banner, the query, then path-aware output: a **BEST ANSWER** line built from the winning walk, and a **REPLAY PATH** with one line per hop (`node id`, `score`, `via=<edge type>`) plus short scoring reasons on the following lines. With `--scenario contract`, a **BASELINE** block (flat top-k answers) appears above the path-aware section for the same query.

Representative runbook excerpt (answer line shortened; latency and hop scores can vary slightly between runs):

```text
========================================================================
  Memory Path Engine  |  demo
  scenario: runbook
========================================================================
-------------------------------- QUERY ---------------------------------
  What should we do if rollback does not recover the API after a
  deployment incident?
----------------- PATH-AWARE  weighted graph retrieval -----------------
  BEST ANSWER
    … stitched runbook units … [latency_ms=…]

  REPLAY PATH
    1. 01_api_incident_runbook:5  |  score=0.500  |  via=seed
       seed hit semantic=0.501
    2. 01_api_incident_runbook:4  |  score=0.299  |  via=next_unit
       expanded at hop 1 total=0.299 exception=0.450 contradiction=0.000
========================================================================
```

## What the demos show

### Runbook demo

The runbook demo loads incident and recovery procedures, then asks a multi-step operational question:

```text
What should we do if rollback does not recover the API after a deployment incident?
```

The output includes:

- a **BEST ANSWER** line composed from the graph walk
- a **REPLAY PATH** with per-step scores, `via` edge types, and short reasons

For a representative stdout excerpt, see **What you will see** (under Quick start).

### Contract demo

The contract demo runs the same query through a baseline retriever and the weighted graph retriever. Stdout shows flat top-k answers first, then the path-aware best answer and replay steps, so you can compare shapes of evidence without relying on a single aggregate metric.

## Retrieval modes in this repo


| Retriever                | What it emphasizes                  | Useful for                                   |
| ------------------------ | ----------------------------------- | -------------------------------------------- |
| lexical baseline         | keyword overlap                     | simple lookups and sanity checks             |
| embedding baseline       | semantic similarity                 | paraphrases and fuzzy matches                |
| structure-only traversal | graph connectivity                  | linked evidence exploration                  |
| weighted graph retrieval | structure plus importance weighting | multi-hop retrieval with replayable evidence |
| activation spreading v1  | explicit propagation with decay     | graph diffusion experiments                  |


## Why the examples span multiple document types

The core is meant to stay domain-agnostic. The current examples use both contract-like documents and runbooks because together they stress:

- hierarchical structure
- exception and dependency chains
- critical risk-bearing units
- procedural and operational steps
- strong need for evidence-backed reasoning

If the retrieval and replay ideas cannot survive across these document types, they are unlikely to generalize well to other structured knowledge domains.

## Repository layout

- [`src/memory_engine`](src/memory_engine): schema, storage, ingestion, retrieval, scoring, replay
- [`examples/contract_pack`](examples/contract_pack): contract-like demo pack with dense dependencies and exceptions
- [`examples/runbook_pack`](examples/runbook_pack): operational runbook pack for procedural retrieval
- [`benchmarks/structured_memory`](benchmarks/structured_memory): typed benchmark fixtures and evaluation assets
- [`docs`](docs): architecture, evaluation, hypotheses, and project vision
- [`tests`](tests): unit tests for schema, retrieval behavior, and benchmark support

## Read this first

- [`docs/vision.md`](docs/vision.md): why this project exists and where it is heading
- [`docs/architecture.md`](docs/architecture.md): how the current system is structured
- [`docs/api-tracks.md`](docs/api-tracks.md): legacy vs palace recall entry points
- [`docs/evaluation.md`](docs/evaluation.md): how retrieval modes are compared
- [`docs/benchmark-strategy.md`](docs/benchmark-strategy.md): how public, repo-owned, and private benchmarks should be used
- [`docs/private-contract-dataset-guide.md`](docs/private-contract-dataset-guide.md): how to build and annotate a private contract golden set
- [`docs/hypotheses.md`](docs/hypotheses.md): milestone hypotheses and success criteria

## Research hypotheses

The first milestone tests three claims:

- `H1`: graph-aware retrieval beats vanilla `top-k` retrieval on multi-hop questions
- `H2`: anomaly and importance weighting improve recall of critical evidence
- `H3`: replayable memory paths improve explainability without unacceptable latency

## Experimental framework

The retrieval stack separates:

- candidate generation
- semantic similarity backend
- scoring strategy
- path replay

That separation makes it possible to compare lexical baseline, embedding baseline, structure-only traversal, and weighted graph retrieval without rewriting the main search loop.

The evaluation layer can emit detailed per-question reports, which is useful for miss analysis and ablation debugging instead of relying only on a single aggregate score.

The repository also includes a dedicated structured benchmark bounded context with:

- strong pydantic dataset models
- a JSON repository for benchmark fixtures
- application services that load datasets, build stores, and run retrievers end to end

## Benchmarks

The benchmark story is intentionally split into three layers:

- **External positioning:** LongMemEval retrieval-only recall (`R@5`, `R@10`, `NDCG@10`) at **session** or **turn** granularity
- **Public retrieval sanity:** HotpotQA evidence retrieval on distractor-style multi-document questions
- **Mechanism validation:** repository-owned structured fixtures for path, semantic, contradiction, and dynamic-memory behavior

Current run matrix:

- `benchmarks/structured_memory/*.json`: CI
- `benchmarks/structured_memory/spatial_recall_benchmark.json`, `route_replay_benchmark.json`, `consolidation_gain_benchmark.json`, `state_transition_benchmark.json`, `contradiction_tension_benchmark.json`: Layer B checks for palace-oriented expectations (space, route shape, diffusion gain, lifecycle) and explicit contradiction / rule-tension pairs
- `benchmarks/external/hotpotqa/hotpot_tiny_fixture.json`: CI sanity
- `benchmarks/external/hotpotqa/data/*.json`: local / nightly (`medium` default 64 samples, `full` optional)
- `benchmarks/external/longmemeval/longmemeval_tiny_fixture.json`: local sanity
- `benchmarks/external/longmemeval/data/*.json`: local / nightly (`medium` default 50 samples, `full` optional)
- `benchmarks/layer_c_minimal/*`: runnable Layer C transfer stand-ins + private annotation templates

## What is in scope for v0.4 (Product M3)

- typed `MemoryNode` / `MemoryEdge` / `MemoryPath` graph
- SQLite-backed local palace (`.mpe/`) and `mpe` CLI
- stdio MCP server + Cursor hook templates (`mpe hooks install`)
- `hybrid` retriever mode (lexical + embedding blend → graph expand)
- LongMemEval **session + turn** KPI baselines and full-corpus reproduce recipe
- domain packs for contract / runbook / research documents

## What is out of scope for now

- Multi-backend vector zoo / hosted embedding services
- LLM-backed answer synthesis (path reasoning stays deterministic)
- multi-modal memory encoding
- full UI

## Planned next steps

See [`docs/ROADMAP.md`](docs/ROADMAP.md). Product **M1–M5** are done; next levers:

- Organization-side Layer C private gold labels
- Optional full LongMemEval-S public table (with `--embedding fastembed` if desired)
- Optional PyPI publish

For suggested GitHub topic tags (About section), see [`docs/github-topics.md`](docs/github-topics.md).

## License

MIT. See [`LICENSE`](LICENSE).