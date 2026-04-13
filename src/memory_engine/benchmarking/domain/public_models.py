from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PublicBenchmarkCaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    query: str
    gold_items: list[str] = Field(default_factory=list)
    retrieved_items: list[str] = Field(default_factory=list)
    matched_ranks: list[int] = Field(default_factory=list)
    hit_at_5: bool = False
    hit_at_10: bool = False
    ndcg_at_10: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PublicBenchmarkModeReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    benchmark_name: str
    dataset_id: str
    retriever_name: str
    questions: int = Field(ge=0)
    recall_at_5: float = Field(default=0.0, ge=0.0, le=1.0)
    recall_at_10: float = Field(default=0.0, ge=0.0, le=1.0)
    ndcg_at_10: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_latency_ms: float = Field(ge=0.0)
    case_reports: list[PublicBenchmarkCaseResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PublicBenchmarkSuiteReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    benchmark_name: str
    dataset_id: str
    modes: dict[str, PublicBenchmarkModeReport]


class BenchmarkBucketSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    questions: int = Field(ge=0)
    evidence_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_latency_ms: float = Field(ge=0.0)


class HotpotQAPerQuestionModeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_hit: bool
    hit: bool
    matched_evidence: list[str] = Field(default_factory=list)
    latency_ms: float = Field(ge=0.0)


class HotpotQAPerQuestionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    question_type: str
    difficulty_level: str
    modes: dict[str, HotpotQAPerQuestionModeResult]


class HotpotQAModeSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    overall: BenchmarkBucketSummary
    breakdown_by_type: dict[str, BenchmarkBucketSummary] = Field(default_factory=dict)
    breakdown_by_level: dict[str, BenchmarkBucketSummary] = Field(default_factory=dict)


class HotpotQASummaryReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    benchmark_name: str
    dataset_id: str
    modes: dict[str, HotpotQAModeSummary]
    per_question_matrix: list[HotpotQAPerQuestionSummary] = Field(default_factory=list)
