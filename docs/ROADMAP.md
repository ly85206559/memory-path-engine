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

### M4 — Distribution (next)

- [ ] `pipx` / `uv tool` install path polish
- [ ] Docker stdio MCP image
- [ ] Backup / repair basics

## Non-goals (for now)

- Multi-backend vector zoo before hybrid + CLI are stable
- LLM answer synthesis as the primary path (PathReasoner stays deterministic)
- Replacing Layer B with public-only metrics
