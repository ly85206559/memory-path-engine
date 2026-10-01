# Dual-track API guide

`Memory Path Engine` currently exposes two cooperating tracks. Prefer the
unified helpers in [`src/memory_engine/api.py`](../src/memory_engine/api.py)
instead of wiring both stacks by hand.

## When to use which track

| Goal | Prefer | Entry point |
| --- | --- | --- |
| Fast graph path replay demos and Layer B fixtures | **legacy** | `recall_from_store` / `recall_from_documents(..., track="legacy")` |
| Space / seed / lifecycle-aware recall | **palace** | `recall_from_palace` / `recall_from_documents(..., track="palace")` |
| Public adapters (HotpotQA / LongMemEval) | legacy store + optional palace projection | existing adapters; summaries stay Layer A |

## Shared contracts

- Retriever mode names are shared (`weighted_graph`, `activation_spreading_v1`, …) via `build_legacy_retriever`.
- `palace_to_store` / `store_to_palace` keep both tracks on the same graph facts.
- Domain packs remain the only place for domain-specific ingest/edge/weight rules.

## Domain packs today

- `example_contract_pack` (`contract_pack` alias)
- `example_runbook_pack`
- `example_research_pack` (`research_pack` alias)
- adapter placeholders: `hotpotqa_sentence_pack`, `longmemeval_session_pack`

Add a new pack by subclassing `RuleBasedSectionedDocumentPack` (or implementing `DomainPack`) and registering it with `register_domain_pack`.

## Migration note

Legacy `MemoryPath` / `RetrievalResult` remain stable. Palace APIs are additive.
New application code should start at `memory_engine.api` so track choice stays explicit and testable.

## Stage 6 brain-like mechanisms

Optional helpers on the same dual-track surface:

| Capability | Entry point | Notes |
| --- | --- | --- |
| Dual episodic / semantic views | `project_dual_views(store)` | Linked `recalls` / `summarizes` edges from one source node |
| Online reinforce / forget | `apply_online_memory_step(..., policy="mild"\|"aggressive")` | Named policies for falsifiable decay curves |
| query → path → answer | `reason_from_recall` / `recall_and_reason` | Deterministic hop citations; not an LLM caller |

These stay additive: existing Layer B fixtures and retriever modes do not require them.

## Product M1 local palace

On-disk workspace helpers (see [`palace_workspace.py`](../src/memory_engine/palace_workspace.py)):

| Command | Purpose |
| --- | --- |
| `mpe init` | Create `.mpe/config.json` + `store.sqlite` |
| `mpe ingest <paths>` | Domain-pack ingest into the SQLite store |
| `mpe search` / `mpe path` | Recall + deterministic path answer |
| `mpe memo` | Append freeform session memo |
| `mpe reinforce` | Search + online mild/aggressive forgetting step |
| `mpe status` | Node/edge counts and palace location |
| `mpe mcp` | Stdio MCP server for Cursor / Claude |
| `mpe hooks install` | Copy hook + MCP templates into `.cursor/mpe-hooks/` |
| `mpe bench longmemeval` | Layer A KPI baseline JSON/Markdown |

Prefer the CLI for product flows; keep `memory_engine.api` for library embedding.
See [`getting-started.md`](getting-started.md) and [`ACCEPTANCE.md`](ACCEPTANCE.md).

