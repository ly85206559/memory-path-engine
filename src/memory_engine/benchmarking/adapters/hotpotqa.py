from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from memory_engine.benchmarking.application.runner import StructuredBenchmarkRunner
from memory_engine.benchmarking.application.service import (
    build_comparison_report,
    build_retriever,
)
from memory_engine.benchmarking.domain.models import (
    StructuredBenchmarkCase,
    StructuredBenchmarkDataset,
    StructuredBenchmarkExpectation,
    StructuredBenchmarkReport,
    StructuredBenchmarkSuiteReport,
)
from memory_engine.schema import EvidenceRef, MemoryEdge, MemoryNode, MemoryWeight
from memory_engine.store import MemoryStore

_ADJACENT_EDGE_TYPE = "next_unit"


def normalize_hotpot_title_for_match(title: str) -> str:
    """Normalize titles for matching `context` entries to `supporting_facts` titles."""
    t = unicodedata.normalize("NFKC", title).strip().casefold()
    t = re.sub(r"\s+", " ", t)
    return t


def hotpot_title_stem(title: str) -> str:
    """Filesystem-safe stem used inside node ids (per paragraph title)."""
    t = unicodedata.normalize("NFKC", title).strip().lower()
    t = t.replace(" ", "_")
    t = re.sub(r"[^a-z0-9_]+", "_", t, flags=re.IGNORECASE)
    t = re.sub(r"_+", "_", t).strip("_")
    return t or "title"


def assign_paragraph_title_stems(context: list) -> dict[str, str]:
    """
    Map each raw paragraph title from `context` to a unique stem.

    HotpotQA `context` is `[[title, [sentences...]], ...]`. Titles must stay
    stable for `supporting_facts` lookup.
    """
    counts: dict[str, int] = {}
    raw_to_stem: dict[str, str] = {}
    for entry in context:
        raw_title = entry[0]
        base = hotpot_title_stem(raw_title)
        n = counts.get(base, 0) + 1
        counts[base] = n
        if n == 1:
            raw_to_stem[raw_title] = base
        else:
            raw_to_stem[raw_title] = f"{base}__{n}"
    return raw_to_stem


def resolve_context_title(supporting_title: str, context: list) -> str:
    """Return the raw `context` title string that matches a supporting_fact title."""
    target = normalize_hotpot_title_for_match(supporting_title)
    for entry in context:
        raw_title = entry[0]
        if raw_title == supporting_title:
            return raw_title
        if normalize_hotpot_title_for_match(raw_title) == target:
            return raw_title
    raise KeyError(f"No context paragraph matches supporting_fact title {supporting_title!r}")


def hotpot_sentence_node_id(stem: str, sent_idx: int) -> str:
    return f"{stem}:{sent_idx}"


def supporting_facts_to_evidence_node_ids(sample: dict) -> list[str]:
    """Map HotpotQA `supporting_facts` to adapter node ids (deduped, order preserved)."""
    context = sample["context"]
    title_to_stem = assign_paragraph_title_stems(context)
    seen: set[str] = set()
    out: list[str] = []
    for sf_title, sent_id in sample["supporting_facts"]:
        raw_title = resolve_context_title(sf_title, context)
        stem = title_to_stem[raw_title]
        node_id = hotpot_sentence_node_id(stem, int(sent_id))
        if node_id not in seen:
            seen.add(node_id)
            out.append(node_id)
    return out


def build_hotpot_memory_store(sample: dict) -> MemoryStore:
    """
    Build a per-question MemoryStore: one node per sentence, chained within each paragraph.

    This matches HotpotQA **distractor** setting: one isolated graph per JSON object.
    """
    store = MemoryStore()
    context = sample["context"]
    title_to_stem = assign_paragraph_title_stems(context)
    sample_ref = str(sample.get("_id", sample.get("id", "hotpot")))

    for entry in context:
        raw_title = entry[0]
        sentences = entry[1]
        stem = title_to_stem[raw_title]
        n = len(sentences)
        for sent_idx, sentence in enumerate(sentences):
            node_id = hotpot_sentence_node_id(stem, sent_idx)
            node = MemoryNode(
                id=node_id,
                type="sentence",
                content=sentence.strip(),
                attributes={
                    "hotpot_title": raw_title,
                    "hotpot_stem": stem,
                    "sentence_index": sent_idx,
                    "sample_id": sample_ref,
                },
                weights=MemoryWeight(
                    importance=0.5,
                    risk=0.2,
                    novelty=0.25,
                    confidence=0.95,
                ),
                source_ref=EvidenceRef(
                    source_path=f"hotpotqa:{sample_ref}",
                    section_id=stem,
                    metadata={"sentence_index": sent_idx, "title": raw_title},
                ),
            )
            store.add_node(node)
            if sent_idx + 1 < n:
                store.add_edge(
                    MemoryEdge(
                        from_id=node_id,
                        to_id=hotpot_sentence_node_id(stem, sent_idx + 1),
                        edge_type=_ADJACENT_EDGE_TYPE,
                        weight=0.85,
                        confidence=0.95,
                        bidirectional=True,
                    )
                )
    return store


def hotpot_sample_to_benchmark_case(sample: dict, *, case_id: str | None = None) -> StructuredBenchmarkCase:
    evidence = supporting_facts_to_evidence_node_ids(sample)
    cid = case_id or str(sample.get("_id", sample.get("id", ""))) or "hotpot-case"
    return StructuredBenchmarkCase(
        case_id=cid,
        query=sample["question"],
        tags=["hotpotqa", "public", str(sample.get("type") or "unknown")],
        expectation=StructuredBenchmarkExpectation(
            evidence_node_ids=evidence,
            minimum_evidence_matches=len(evidence),
        ),
    )


def hotpot_samples_to_dataset(
    samples: list[dict],
    *,
    dataset_id: str = "hotpotqa-minimal",
    dataset_name: str = "HotpotQA (adapter)",
) -> StructuredBenchmarkDataset:
    """Build a typed dataset. Do not use `build_store_for_dataset` with this pack."""
    cases = [
        hotpot_sample_to_benchmark_case(s, case_id=str(s.get("_id", s.get("id", f"hotpot-{i}"))))
        for i, s in enumerate(samples)
    ]
    return StructuredBenchmarkDataset(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        domain_pack_name="hotpotqa_sentence_pack",
        document_directory="../../benchmarks/external/hotpotqa",
        cases=cases,
    )


def load_hotpotqa_json_array(path: Path) -> list[dict]:
    """Load a HotpotQA-style JSON file (array of question objects)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data)}")
    return data


def run_hotpotqa_benchmark(
    samples: list[dict],
    *,
    retriever_modes: tuple[str, ...] = ("lexical_baseline", "embedding_baseline"),
    top_k: int = 10,
    dataset_id: str = "hotpotqa-adapter-eval",
) -> StructuredBenchmarkSuiteReport:
    """
    Run each sample in isolation (fresh MemoryStore per question), then aggregate by mode.

    This is required for HotpotQA distractor: graphs must not be merged across questions.
    """
    runner = StructuredBenchmarkRunner()
    mode_reports: dict[str, StructuredBenchmarkReport] = {}

    for mode in retriever_modes:
        case_reports = []
        for sample in samples:
            store = build_hotpot_memory_store(sample)
            retriever = build_retriever(mode, store)
            case = hotpot_sample_to_benchmark_case(sample)
            single = StructuredBenchmarkDataset(
                dataset_id=dataset_id,
                dataset_name="HotpotQA single",
                domain_pack_name="hotpotqa_sentence_pack",
                document_directory="../../benchmarks/external/hotpotqa",
                cases=[case],
            )
            report = runner.run(
                dataset=single,
                retriever_name=mode,
                retriever=retriever,
                top_k=top_k,
            )
            case_reports.append(report.case_reports[0])

        n = len(case_reports)
        evidence_hit_rate = (
            sum(1 for r in case_reports if r.evidence_hit) / n if n else 0.0
        )
        evidence_recall = sum(1 for r in case_reports if r.hit) / n if n else 0.0
        avg_latency_ms = (
            round(sum(r.latency_ms for r in case_reports) / n, 3) if n else 0.0
        )
        mode_reports[mode] = StructuredBenchmarkReport(
            dataset_id=dataset_id,
            retriever_name=mode,
            questions=n,
            evidence_hit_rate=evidence_hit_rate,
            evidence_recall=evidence_recall,
            avg_latency_ms=avg_latency_ms,
            case_reports=case_reports,
        )

    return StructuredBenchmarkSuiteReport(
        dataset_id=dataset_id,
        modes=mode_reports,
        comparison=build_comparison_report(mode_reports),
    )
