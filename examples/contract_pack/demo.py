from pathlib import Path

from memory_engine.ingest import ingest_contract_markdown
from memory_engine.retrieve import BaselineTopKRetriever, WeightedGraphRetriever
from memory_engine.store import MemoryStore


def build_store() -> MemoryStore:
    store = MemoryStore()
    for path in Path(__file__).parent.joinpath("contracts").glob("*.md"):
        ingest_contract_markdown(path, store)
    return store


if __name__ == "__main__":
    store = build_store()
    query = "What happens if delivery is late and the supplier does not cure in time?"

    baseline = BaselineTopKRetriever(store).search(query, top_k=2)
    weighted = WeightedGraphRetriever(store).search(query, top_k=2)

    print("BASELINE")
    for path in baseline.paths:
        print(path.final_answer)

    print("\nWEIGHTED GRAPH")
    for step in weighted.best_path().steps:
        print(step.node_id, step.reason, round(step.score, 3), step.via_edge_type)
    print(weighted.best_path().final_answer)
