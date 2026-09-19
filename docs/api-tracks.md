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
