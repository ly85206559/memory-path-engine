from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PathStepExpectation(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: str = Field(min_length=1)
    via_edge_type: str | None = None

    @field_validator("via_edge_type")
    @classmethod
    def empty_edge_type_is_not_allowed(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("via_edge_type cannot be empty")
        return value


class PathShapeExpectation(BaseModel):
    model_config = ConfigDict(frozen=True)

    match_mode: Literal["subsequence", "prefix", "exact"] = "subsequence"
    steps: list[PathStepExpectation] = Field(min_length=1)


class StructuredBenchmarkExpectation(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_node_ids: list[str] = Field(min_length=1)
    minimum_evidence_matches: int = Field(default=1, ge=1)
    path: PathShapeExpectation | None = None
    path_scope: Literal["any_path", "best_path"] = "any_path"

    @field_validator("minimum_evidence_matches")
    @classmethod
    def minimum_matches_must_not_exceed_expected_evidence(cls, value: int, info):
        evidence_node_ids = info.data.get("evidence_node_ids", [])
        if evidence_node_ids and value > len(evidence_node_ids):
            raise ValueError("minimum_evidence_matches cannot exceed evidence_node_ids length")
        return value


class StructuredBenchmarkCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    expectation: StructuredBenchmarkExpectation


class StructuredBenchmarkDataset(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    domain_pack_name: str = Field(min_length=1)
    document_directory: str = Field(min_length=1)
    cases: list[StructuredBenchmarkCase] = Field(min_length=1)


class StructuredBenchmarkCaseReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    query: str
    tags: list[str] = Field(default_factory=list)
    hit: bool
    path_hit: bool | None = None
    expected_evidence: list[str]
    matched_evidence: list[str]
    missing_evidence: list[str]
    returned_node_ids: list[str]
    best_answer: str = ""
    latency_ms: float = Field(ge=0.0)


class StructuredBenchmarkReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: str
    retriever_name: str
    questions: int = Field(ge=0)
    evidence_recall: float = Field(ge=0.0, le=1.0)
    avg_latency_ms: float = Field(ge=0.0)
    case_reports: list[StructuredBenchmarkCaseReport] = Field(default_factory=list)


class StructuredBenchmarkModeSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_recall: float = Field(ge=0.0, le=1.0)
    avg_latency_ms: float = Field(ge=0.0)


class StructuredBenchmarkModeCaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    hit: bool
    matched_evidence: list[str] = Field(default_factory=list)
    latency_ms: float = Field(ge=0.0)


class StructuredBenchmarkComparisonCaseReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    modes: dict[str, StructuredBenchmarkModeCaseResult]
    best_modes: list[str] = Field(default_factory=list)
    missed_by_all: bool = False


class StructuredBenchmarkComparisonReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    per_question: list[StructuredBenchmarkComparisonCaseReport] = Field(default_factory=list)
    mode_summary: dict[str, StructuredBenchmarkModeSummary] = Field(default_factory=dict)


class StructuredBenchmarkSuiteReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: str
    modes: dict[str, StructuredBenchmarkReport]
    comparison: StructuredBenchmarkComparisonReport
