from pathlib import Path

from memory_engine.ingest import ingest_document
from memory_engine.retrieve import WeightedGraphRetriever
from memory_engine.store import MemoryStore


def build_store() -> MemoryStore:
    store = MemoryStore()
    for path in Path(__file__).parent.joinpath("runbooks").glob("*.md"):
        ingest_document(path, store, domain_pack="example_runbook_pack")
    return store


if __name__ == "__main__":
    store = build_store()
    query = "What should we do if rollback does not recover the API after a deployment incident?"

    result = WeightedGraphRetriever(store).search(query, top_k=3)

    for step in result.best_path().steps:
        print(step.node_id, step.reason, round(step.score, 3), step.via_edge_type)
    print(result.best_path().final_answer)
