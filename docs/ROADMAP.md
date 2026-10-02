# Product roadmap

This repository is moving from a research harness to a **local memory product**
with replayable paths. Architecture claims stay in Layer B; public retrieval
recall becomes a first-class product KPI (Layer A).

## Positioning

> Local memory with replayable paths — public-benchmark recall, agent-ready
> closed loop, and explainable evidence paths.

| Surface | Role |
| --- | --- |
| Layer A (LongMemEval / HotpotQA) | Product KPI and external credibility |
| Layer B (structured fixtures) | Differentiation: path, contradiction, dynamics |
| Layer C | Real-document transfer |

## Milestones

### M1 — Persist + CLI + baseline skeleton

- [x] SQLite-backed `MemoryStore` persistence
- [x] Local palace directory (`.mpe/` or `$MPE_PALACE`)
- [x] `mpe` CLI: `init` / `ingest` / `search` / `path` / `status`
- [x] `mpe bench longmemeval` writes JSON + Markdown baseline artifacts
- [x] Commit tiny-fixture baseline under `benchmarks/external/longmemeval/baselines/`

### M2 — Agent closed loop

- [x] MCP server (`mpe mcp` / `mpe-mcp`): status, ingest, memo, search, path, reinforce
- [x] Cursor hook templates + `mpe hooks install`
- [x] 5-minute guide: [`getting-started.md`](getting-started.md)
- [x] Acceptance checklist: [`ACCEPTANCE.md`](ACCEPTANCE.md)

### M3 — Stronger public KPI (current)

- [x] Hybrid retrieve mode (`hybrid`) — lexical+embedding blend then graph expand
- [x] Turn-level session units for LongMemEval (`--granularity turn`)
- [x] Full LongMemEval-S reproducible recipe documented in README + baseline artifacts for tiny session/turn

### M4 — Distribution

- [x] `pipx` / `uv tool` install path polish ([`install.md`](install.md))
- [x] Docker stdio MCP image (`Dockerfile`, `mpe-mcp`)
- [x] Backup / repair / doctor basics (`mpe backup` / `mpe repair` / `mpe doctor`)

### M5 — Pluggable dense embeddings + KPI

- [x] `EmbeddingProvider` registry: `ngram` (default) / `hash` / `fastembed` / `sentence`
- [x] Select via `--embedding` or `MPE_EMBEDDING` (optional model via `MPE_EMBEDDING_MODEL`)
- [x] Optional extras: `pip install 'memory-path-engine[embed]'` (fastembed) / `[embed-st]`
- [x] Default stack stays dependency-light; Layer B hashing defaults unchanged unless embedding is explicit
- [x] LongMemEval medium KPI comparison artifacts (ngram vs fastembed)

### M6 — Batch dense encode + full public KPI

- [x] `embed_many` + retriever prefetch for dense backends
- [x] Long-text head+tail truncation / adaptive batch sizing for ONNX context
- [x] Commit full LongMemEval-S (500q) public KPI table (`longmemeval_kpi_full_ngram.*`)
- [x] Optional full fastembed comparison artifact when runtime permits

### M7 — Hybrid public ranking / NDCG

- [x] Diagnose NDCG gap: seed order was correct; path scoring buried gold seeds
- [x] `HybridRetriever` re-aligns palace ranking to BM25/blend seed scores after expansion
- [x] Layer B `WeightedGraphRetriever` unchanged
- [x] Refresh medium50 / full LongMemEval KPI tables

### M8 — PyPI distribution

- [x] PyPI-ready `pyproject.toml` metadata (classifiers, keywords, URLs, extras)
- [x] Trusted Publisher workflow (`.github/workflows/publish.yml`) + CI package build job
- [x] Publish guide [`publish.md`](publish.md) + `CHANGELOG.md`
- [x] `scripts/release.sh` one-shot tag/publish helper (no GitHub Environment required)
- [x] First upload: **`memory-path-engine==0.8.0`** on PyPI

### M9 — Release polish

- [x] ROADMAP/progress/README reflect published status + PyPI badge
- [x] `mpe doctor` install tips prefer PyPI
- [x] GitHub Release for `v0.8.0` / follow-up `v0.8.1` polish tag

## Non-goals (for now)

- Hosted multi-vendor vector zoo as a hard requirement
- LLM answer synthesis as the primary path (PathReasoner stays deterministic)
- Replacing Layer B with public-only metrics

## Progress snapshot

See [`progress.md`](progress.md) for the Stage 0–6 / Layer A–C / Product M1–M9 scorecard against the original vision + benchmark strategy.
