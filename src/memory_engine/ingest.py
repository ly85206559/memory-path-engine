from __future__ import annotations

from pathlib import Path

from memory_engine.domain_pack import DomainPack, get_domain_pack
from memory_engine.store import MemoryStore


def ingest_document(
    path: Path,
    store: MemoryStore,
    domain_pack: str | DomainPack = "contract_pack",
) -> None:
    resolved_pack = domain_pack if not isinstance(domain_pack, str) else get_domain_pack(domain_pack)
    resolved_pack.ingest_document(path, store)


def ingest_contract_markdown(path: Path, store: MemoryStore, domain_pack: str = "contract_pack") -> None:
    if domain_pack != "contract_pack":
        raise ValueError(
            "ingest_contract_markdown only supports the 'contract_pack' domain pack. "
            "Use ingest_document(...) for generic domain-pack ingestion."
        )
    ingest_document(path, store, domain_pack=domain_pack)
